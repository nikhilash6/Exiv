import os
from pathlib import Path
import sys
import json
import importlib.util
import traceback
from typing import Dict, List, Type, Any, Union
from .extensions import Extension
from ..utils.logging import app_logger
from ..utils.file import find_file_path, CONFIG_FILENAME

EXTENSION_ENTRYPOINT = "extension.py"
DEFAULT_CONFIG = {"extensions": []}

class ExtensionRegistry:
    _instance = None
    
    def __init__(self):
        # maps ID -> extension manifest instance
        self.extensions: Dict[str, Extension] = {}
        self._initialized = False
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ExtensionRegistry()
            cls._instance.initialize()
            cls._instance.execute_all_extensions()
        return cls._instance

    def _setup_extension_namespace(self, full_module_name: str, path: str):
        """
        Ensures the parent namespaces exist as packages and returns a module spec.
        """
        import types
        # ensure parent packages exist so the dotted name is valid (e.g., 'exiv.ext.inspect')
        parts = full_module_name.split('.')[:-1]
        for i in range(len(parts)):
            part_name = ".".join(parts[:i+1])
            if part_name not in sys.modules:
                sys.modules[part_name] = types.ModuleType(part_name)
                sys.modules[part_name].__path__ = []
            elif not hasattr(sys.modules[part_name], "__path__"):
                # ensure it's treated as a package even if it already exists
                sys.modules[part_name].__path__ = []

        entry_path = os.path.join(path, EXTENSION_ENTRYPOINT)
        spec = importlib.util.spec_from_file_location(full_module_name, entry_path)
        if spec:
            spec.submodule_search_locations = [path]
        return spec, entry_path

    @classmethod
    def load_config(cls, config_file: Path) -> dict:
        if not config_file.exists():
            return DEFAULT_CONFIG.copy()

        try:
            with config_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data.get("extensions", []), list):
                    data["extensions"] = []
                return data
        except (OSError, json.JSONDecodeError) as e:
            app_logger.warning(f"Could not read {config_file}, creating new one. Error: {e}")
            return DEFAULT_CONFIG.copy()

    @classmethod
    def save_config(cls, config_file: Path, config: dict) -> None:
        with config_file.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")

    def load_extensions_from_directory(self, directory: str):
        if not os.path.exists(directory):
            app_logger.warning(f"Extension directory not found: {directory}")
            return

        # ensuring the directory is importable
        if directory not in sys.path: sys.path.append(directory)
        app_logger.debug(f"Scanning for extensions in: {directory}")
        for item in os.listdir(directory):
            ext_path = os.path.join(directory, item)
            # NOTE: only folders containing EXTENSION_ENTRYPOINT are scanned
            if os.path.isdir(ext_path) and os.path.exists(os.path.join(ext_path, EXTENSION_ENTRYPOINT)):
                self.load_single_extension(ext_path)

    def load_single_extension(self, path: str):
        """
        Creates a consistent namespace and imports the extension class in the main code
        """
        module_name = os.path.basename(path)
        try:
            # consistent namespace: exiv.ext.<module_name>
            full_module_name = f"exiv.ext.{module_name}"
            spec, _ = self._setup_extension_namespace(full_module_name, path)
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # registering so imports like 'import exiv.ext.foo' work
                sys.modules[full_module_name] = module
                try:
                    spec.loader.exec_module(module)
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, Extension) and attr is not Extension:
                            if attr.ID in self.extensions:
                                app_logger.debug(f"Extension {attr.ID} already registered. Skipping reload from {path}")
                                return
                            self.extensions[attr.ID] = attr
                            app_logger.debug(f"Registered extension: {attr.ID}")
                            break
                            
                except Exception as e:
                    # Clean up if execution fails
                    if full_module_name in sys.modules:
                        del sys.modules[full_module_name]
                    raise e
        except Exception as e:
            app_logger.error(f"Failed to load extension {module_name}: {e}")
            traceback.print_exc()

    def _execute_extension(self, cls: Type[Extension]):
        """
        Executes all callables in the given extension
        """
        try:
            capabilities = cls().register()
            for item in capabilities: 
                if callable(item): item()
        except Exception as e:
            app_logger.error(f"Error executing extension {cls}: {e}")
            traceback.print_exc()

    def execute_all_extensions(self):
        """
        Executes all callables returned by the registered extensions
        """
        if not self.extensions: return
        for e_id, ext in self.extensions.items(): self._execute_extension(ext)
        app_logger.debug(f"Applied {len(self.extensions)} system patches.")

    def initialize(self):
        """
        Loads built-in extensions and user-registered extensions.
        """
        if self._initialized:
            return
            
        # loading built-in extensions
        current_dir = os.path.dirname(os.path.abspath(__file__))
        builtin_dir = os.path.abspath(os.path.join(current_dir, "..", "extensions"))
        if os.path.exists(builtin_dir): self.load_extensions_from_directory(builtin_dir)

        # loading registered extensions from exiv_config.json
        abs_paths = self.get_all_registered_paths(absolute_path=True)
        for p in abs_paths: self.load_single_extension(p)
        
        self._initialized = True
    def get_all_registered_paths(self, absolute_path=False) -> List[str]:
        """
        Returns the paths to all the extensions registered / installed.
        Always reads from the config file to ensure paths are returned.
        """
        config_file_str, config_dir_str = find_file_path(CONFIG_FILENAME, recursive=True)
        if not config_file_str:
            app_logger.debug("No exiv_config.json found, returning empty extension list.")
            return []

        config = self.load_config(Path(config_file_str))
        paths = config.get("extensions", [])
        if absolute_path:
            abs_paths = []
            for p in paths:
                if not os.path.isabs(p): p = os.path.abspath(os.path.join(config_dir_str, p))
                abs_paths.append(p)
            paths = abs_paths
        return paths
    
    def get_extension_info_from_path(self, path: str) -> Dict[str, str]:
        """
        Reads metadata from an extension at a specific path without registering it.
        """
        entry_path = os.path.join(path, EXTENSION_ENTRYPOINT)
        default_return = None
        if not os.path.exists(entry_path): return default_return

        try:
            module_name = f"exiv.ext.inspect.{os.path.basename(path)}"
            spec, _ = self._setup_extension_namespace(module_name, path)

            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module   # temporarily register in sys.modules to support imports
                try:
                    spec.loader.exec_module(module)
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, Extension) and attr is not Extension:
                            return {
                                "id": attr.ID,
                                "name": attr.DISPLAY_NAME,
                                "version": attr.VERSION,
                                "path": path
                            }
                finally:
                    # clean up
                    if module_name in sys.modules:
                        del sys.modules[module_name]
        except Exception as e:
            app_logger.debug(f"Error inspecting extension at {path}: {e}")

        return default_return

    def get_all_extensions_metadata(self):
        """
        Returns metadata for ALL registered extensions.
        """
        abs_paths = self.get_all_registered_paths(absolute_path=True)
        meta = []
        for p in abs_paths:
            res = self.get_extension_info_from_path(p)
            if res: meta.append(res)
        return meta
