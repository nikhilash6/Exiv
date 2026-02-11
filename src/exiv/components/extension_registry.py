import os
import sys
import json
import importlib.util
import traceback
from typing import Dict, List, Type, Any, Union
from .extensions import Extension
from ..utils.logging import app_logger
from ..utils.file import find_file_path, CONFIG_FILENAME

class ExtensionRegistry:
    _instance = None
    
    def __init__(self):
        """
        - patches: All registered capabilities from extensions. 
                   If callable, they are executed at startup.
                   If objects, they are just stored here.
        """
        # maps ID -> extension manifest instance
        self.extensions: Dict[str, Extension] = {}
        
        # capabilities
        self.patches: List[Any] = []
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ExtensionRegistry()
        return cls._instance

    def load_extensions_from_path(self, directory: str):
        if not os.path.exists(directory):
            app_logger.warning(f"Extension directory not found: {directory}")
            return

        # ensuring the directory is importable
        if directory not in sys.path:
            sys.path.append(directory)
            
        app_logger.debug(f"Scanning for extensions in: {directory}")
        for item in os.listdir(directory):
            ext_path = os.path.join(directory, item)
            # NOTE: only folders containing __init__.py are scanned
            if os.path.isdir(ext_path) and os.path.exists(os.path.join(ext_path, "__init__.py")):
                self._load_single_extension(item, ext_path)

    def _load_single_extension(self, module_name: str, path: str):
        try:
            # using spec_from_file_location is safer for arbitrary paths
            init_path = os.path.join(path, "__init__.py")
            spec = importlib.util.spec_from_file_location(module_name, init_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                extension_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, Extension) and attr is not Extension:
                        extension_class = attr
                        break
                
                if extension_class:
                    self._register_extension(extension_class)
                else:
                    app_logger.debug(f"No Extension subclass found in {module_name}")
                    
        except Exception as e:
            app_logger.error(f"Failed to load extension {module_name}: {e}")
            traceback.print_exc()

    def _register_extension(self, cls: Type[Extension]):
        try:
            instance = cls()
            self.extensions[cls.ID] = instance
            app_logger.info(f"Loaded Extension: {cls.DISPLAY_NAME} ({cls.ID}) v{cls.VERSION}")
            
            # get capabilities
            capabilities = instance.register()
            
            for item in capabilities:
                # Treat everything as a patch/capability
                self.patches.append(item)
                if callable(item):
                    app_logger.debug(f"  - Registered Callable Patch")
                else:
                    app_logger.debug(f"  - Registered Object Patch: {item}")

        except Exception as e:
            app_logger.error(f"Error registering extension {cls.ID}: {e}")
            traceback.print_exc()

    def run_patches(self):
        """
        Executes all registered callable patches.
        """
        if not self.patches: return
        
        count = 0
        for handler in self.patches:
            if callable(handler) and not isinstance(handler, Extension):
                # We generally don't "call" the Extension object itself unless it implements __call__
                # But usually patches are separate functions.
                # If an Extension object IS callable, we call it? 
                # Let's assume 'patches' are meant to be executed if they are functions.
                try:
                    handler()
                    count += 1
                except Exception as e:
                    app_logger.error(f"Patch failed: {e}")
                    traceback.print_exc()
        
        if count > 0:
            app_logger.info(f"Applied {count} system patches.")

    def initialize(self, run_patches: bool = True):
        """
        Loads built-in extensions and user-registered extensions.
        """
        if self.extensions:
            return

        # loading built-in extensions
        current_dir = os.path.dirname(os.path.abspath(__file__))
        builtin_dir = os.path.abspath(os.path.join(current_dir, "..", "extensions"))
        
        if os.path.exists(builtin_dir):
            self.load_extensions_from_path(builtin_dir)

        # loading registered extensions from exiv_config.json
        config_file, config_dir = find_file_path(CONFIG_FILENAME, recursive=True)
        
        if config_file and config_dir:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
                registered_paths = config.get("extensions", [])
                for ext_path in registered_paths:
                    # resolve relative paths against the config file location
                    if not os.path.isabs(ext_path):
                        abs_path = os.path.abspath(os.path.join(config_dir, ext_path))
                    else:
                        abs_path = ext_path

                    if os.path.exists(abs_path):
                        self.load_extensions_from_path(abs_path)
                    else:
                        app_logger.warning(f"Registered extension path not found: {abs_path}")
            except Exception as e:
                app_logger.error(f"Failed to load exiv_config.json: {e}")
        else:
            app_logger.debug("No exiv_config.json found")

        if run_patches:
            self.run_patches()

    def get_all_extensions_metadata(self):
        """
        Returns metadata for ALL registered extensions.
        """
        meta = []
        for ext_id, ext in self.extensions.items():
            meta.append({
                "id": ext.ID,
                "name": ext.DISPLAY_NAME,
                "version": ext.VERSION,
            })
        return meta
