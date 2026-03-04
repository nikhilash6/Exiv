import os
import json
import glob
from dataclasses import dataclass
from typing import List, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)

# TODO: not properly tested in a multi root paths scenario

# going with what everyone is using
DEFAULT_MAPPING = {
        "checkpoint":      ["models/checkpoints"],
        "unet":            ["models/unet", "models/diffusion_models"],
        "lora":            ["models/loras"],
        "vae":             ["models/vae"],
        "text_encoder":    ["models/clip"],
        "vision_encoder":  ["models/clip_vision"],
        "style_model":     ["models/style_models"],
        "embedding":       ["models/embeddings"],
        "hypernetwork":    ["models/hypernetworks"],
        "controlnet":      ["models/controlnet"],
        "upscale_model":   ["models/upscale_models"],
        "gligen":          ["models/gligen"],
        "input":           ["input"],
        "output":          ["output"]
}

# package source root (where the code lives)
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# user workspace root (where models/outputs should go)
USER_ROOT = os.getcwd()

def load_download_map() -> Dict[str, Dict[str, str]]:
    """
    Loads model download metadata from JSON files in the src/exiv/data/registry/ directory.
    """
    # Registry is in the src/exiv/data/registry/ folder
    registry_dir = os.path.join(PACKAGE_ROOT, "src", "exiv", "data", "registry")
    
    # Fallback: check if it's in the current working directory (useful for dev/hub)
    if not os.path.exists(registry_dir):
        registry_dir = os.path.join(USER_ROOT, "data", "registry")

    download_map = {}
    if os.path.exists(registry_dir):
        json_files = glob.glob(os.path.join(registry_dir, "*.json"))
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    download_map.update(data)
            except Exception as e:
                # Using print as fallback if logger is not yet configured or causes issues
                print(f"Error loading registry file {json_file}: {e}")
                
    return download_map

DOWNLOAD_MAP = load_download_map()

@dataclass
class FilePathData:
    name: str | None = None
    path: str | None = None
    is_present: bool = False
    url: str | None = None

class FilePaths:
    # stores registered search paths
    # Format: [{'path': '/abs/path', 'map': {'type': ['folder1', 'folder2']}}]
    _search_roots = []
    
    # cache: root_path -> [list of all absolute file paths found recursively]
    _file_cache = {}
    
    @classmethod
    def get_output_directory(cls) -> str:
        """Returns the absolute path to the output directory."""
        return cls.get_save_folder("output")

    @classmethod
    def resolve_path(cls, path: str) -> str:
        """
        Attempts to resolve a path (possibly relative) to an absolute path
        by checking registered search roots.
        """
        if not path:
            return ""
        
        # 1. returning absolute
        if os.path.isabs(path):
            return path
            
        # 2. check relative to search roots (including subfolders like 'output', 'input', 'models')
        # we check direct first, then subfolders
        for entry in cls._search_roots:
            root = entry["path"]
            mapping = entry["map"]
            
            # 2a. check direct relative to root
            potential = os.path.abspath(os.path.join(root, path))
            if os.path.exists(potential):
                return potential
                
            # 2b. check inside type-specific subfolders
            for file_type, folders in mapping.items():
                for folder in folders:
                    potential = os.path.abspath(os.path.join(root, folder, path))
                    if os.path.exists(potential):
                        return potential
        
        # 3. check relative to USER_ROOT (fallback)
        potential = os.path.abspath(os.path.join(USER_ROOT, path))
        if os.path.exists(potential):
            return potential
            
        # 4. if nothing found, return original (or could raise error, but let caller handle)
        return path

    @classmethod
    def set_extra_search_paths(cls, paths: List[str]):
        """Sets external search paths, replacing any previously set extra paths."""
        core_paths = [os.path.abspath(USER_ROOT), os.path.abspath(PACKAGE_ROOT)]
        cls._search_roots = [r for r in cls._search_roots if r["path"] in core_paths]
        
        for p in paths:
            if p:
                abs_p = os.path.abspath(p)
                if os.path.exists(abs_p):
                    cls.add_search_path(abs_p)

    @classmethod
    def add_search_path(cls, path: str, mapping: Dict[str, Union[str, List[str]]] = None):
        """
        Registers a root directory to search for files.
        
        Args:
            path: The root directory path.
            mapping: A dict mapping a 'type' to one or more folder names.
                     e.g. {"lora": ["loras", "finetuned/lora"], "checkpoints": "models"}
        """
        if mapping is None:
            mapping = DEFAULT_MAPPING

        clean_map = {}
        for key, val in mapping.items():
            if isinstance(val, str):
                clean_map[key] = [val]
            else:
                clean_map[key] = val
                
        abs_path = os.path.abspath(path)
        if any(r["path"] == abs_path for r in cls._search_roots):
            return
            
        cls._search_roots.append({
            "path": abs_path,
            "map": clean_map
        })
        
        cls._file_cache = {}    # force a rescan when the files are fetched next

    @classmethod
    def init_cache(cls):
        """
        scans all registered search paths and caches the file structure
        should be called on app startup
        """
        cls._file_cache = {}
        
        for entry in cls._search_roots:
            root = entry["path"]
            if not os.path.exists(root):
                continue
            
            files_in_root = []
            for dirpath, dirnames, filenames in os.walk(root):
                # optimization: ignoring hidden directories and common large folders that don't contain models
                dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in ('venv', '__pycache__', 'node_modules', 'dist', 'build', 'exiv.egg-info')]
                for f in filenames:
                    files_in_root.append(os.path.join(dirpath, f))
            
            cls._file_cache[root] = files_in_root

    @classmethod
    def get_files(cls, file_type: str, extensions: List[str] = None) -> List[FilePathData]:
        """
        Retrieves files matching the given type, including missing downloadable ones
        Returns a list of dicts: {'name': str, 'path': str|None, 'is_present': bool, 'url': str|None}
        """
        if not cls._file_cache:
            cls.init_cache()
            
        results = []
        found_names = set()
        
        # existing files on disk
        for entry in cls._search_roots:
            root = entry["path"]
            mapping = entry["map"]
            
            target_folders = mapping.get(file_type, [file_type])
            cached_files = cls._file_cache.get(root, [])
            
            for f_path in cached_files:
                rel_path = os.path.relpath(f_path, root)
                
                is_in_folder = False
                for tf in target_folders:
                    tf = os.path.normpath(tf)
                    if rel_path == tf or rel_path.startswith(tf + os.sep):
                        is_in_folder = True
                        break
                
                if is_in_folder:
                    if extensions and not any(f_path.lower().endswith(ext.lower()) for ext in extensions):
                        continue
                    
                    name = os.path.basename(f_path)
                    found_names.add(name)
                    
                    url = DOWNLOAD_MAP.get(name, {}).get("url") if name in DOWNLOAD_MAP and DOWNLOAD_MAP[name].get("type") == file_type else None
                    results.append(
                        FilePathData(
                            name=name,
                            path=f_path,
                            is_present=True,
                            url=url
                        )
                    )

        # missing downloadable files
        save_folder = cls.get_save_folder(file_type)
        for name, info in DOWNLOAD_MAP.items():
            if info.get("type") == file_type and name not in found_names:
                if extensions and not any(name.lower().endswith(ext.lower()) for ext in extensions):
                    continue
                
                target_path = os.path.join(save_folder, name)
                results.append(
                    FilePathData(
                        name=name,
                        path=target_path,
                        is_present=False,
                        url=info.get("url")
                    )
                )
                        
        return sorted(results, key=lambda x: x.name)

    @classmethod
    def get_path(cls, filename: str, file_type: str) -> FilePathData:
        """
        Resolves a file by name and type.
        Prioritizes:
        1. Direct File Path (Absolute or Relative) -> returns immediately if found.
        2. Local Exact match (in registered roots)
        3. Local Stem match (in registered roots)
        4. Download Map Exact match
        5. Download Map Stem match
        
        Returns a dict: {'name': str, 'path': str, 'is_present': bool, 'url': str|None}
        Raises FileNotFoundError if not found in local or downloadable.
        """
        
        # user specific path is directly used (e.g. "/tmp/custom.safetensors" or "./model.ckpt")
        potential_path = os.path.abspath(filename)
        if os.path.isfile(potential_path):
            name = os.path.basename(potential_path)
            # check if we happen to have a URL for this filename in our map, 
            # just in case they are manually pointing to a known file.
            url = None
            if name in DOWNLOAD_MAP and DOWNLOAD_MAP[name].get("type") == file_type:
                url = DOWNLOAD_MAP[name].get("url")

            return FilePathData(
                name=name,
                path=potential_path,
                is_present=True,
                url=url
            )
        
        if not cls._file_cache:
            cls.init_cache()

        # local files
        candidates = []
        for entry in cls._search_roots:
            root = entry["path"]
            mapping = entry["map"]
            
            target_folders = mapping.get(file_type, [file_type])
            cached_files = cls._file_cache.get(root, [])
            
            for f_path in cached_files:
                rel_path = os.path.relpath(f_path, root)
                
                # check folder membership
                is_in_folder = False
                for tf in target_folders:
                    tf = os.path.normpath(tf)
                    if rel_path == tf or rel_path.startswith(tf + os.sep):
                        is_in_folder = True
                        break
                
                if not is_in_folder:
                    continue

                f_name = os.path.basename(f_path)
                
                # exact match
                if f_name == filename:
                    if os.path.exists(f_path):
                        url = DOWNLOAD_MAP.get(f_name, {}).get("url") if f_name in DOWNLOAD_MAP and DOWNLOAD_MAP[f_name].get("type") == file_type else None
                        return FilePathData(
                            name=f_name,
                            path=f_path,
                            is_present=True,
                            url=url
                        )
                
                # stem match (if input has no extension)
                if os.path.splitext(f_name)[0] == filename:
                     if os.path.exists(f_path):
                        candidates.append(f_path)

        # local stem candidates
        if candidates:
            # Pick first candidate
            f_path = candidates[0]
            f_name = os.path.basename(f_path)
            url = DOWNLOAD_MAP.get(f_name, {}).get("url") if f_name in DOWNLOAD_MAP and DOWNLOAD_MAP[f_name].get("type") == file_type else None
            return FilePathData(
                    name=f_name,
                    path=f_path,
                    is_present=True,
                    url=url
                )

        save_folder = cls.get_save_folder(file_type)

        # exact download map
        if filename in DOWNLOAD_MAP:
            info = DOWNLOAD_MAP[filename]
            if info.get("type") == file_type:
                target_path = os.path.join(save_folder, filename)
                return FilePathData(
                        name=filename,
                        path=target_path,
                        is_present=False,
                        url=info.get("url")
                    )

        # steam download map
        for name, info in DOWNLOAD_MAP.items():
            if info.get("type") == file_type:
                if os.path.splitext(name)[0] == filename:
                    target_path = os.path.join(save_folder, name)
                    return FilePathData(
                        name=name,
                        path=target_path,
                        is_present=False,
                        url=info.get("url")
                    )

        # raise FileNotFoundError(f"File '{filename}' of type '{file_type}' not found locally or in download map.")
        return FilePathData()
    
    @classmethod
    def get_save_folder(cls, file_type: str) -> str:
        """
        Returns the absolute path to the default save directory for a given file type.
        Uses the first registered search root (primary).
        Creates the directory if it does not exist.
        """
        if not cls._search_roots:
            # fallback to USER_ROOT instead of CWD
            root = USER_ROOT
            mapping = DEFAULT_MAPPING
        else:
            root = cls._search_roots[0]["path"]
            mapping = cls._search_roots[0]["map"]

        folders = mapping.get(file_type)
        if not folders:
            raise ValueError(f"Unknown file type: '{file_type}'")

        save_dir = os.path.join(root, folders[0])
        os.makedirs(save_dir, exist_ok=True)
        return os.path.abspath(save_dir)


# NOTE: the first registered root is treated as the primary and 
# will be used to save models and the outputs
# adding USER_ROOT first so that it is the primary for outputs/downloads
FilePaths.add_search_path(USER_ROOT)
if PACKAGE_ROOT != USER_ROOT:
    # NOTE: although search paths are primarily added for finding models, they could also contain
    # other built-in assets
    FilePaths.add_search_path(PACKAGE_ROOT)

# TODO: for backward compatibility (remove this)
FilePaths.OUTPUT_DIRECTORY = FilePaths.get_output_directory()