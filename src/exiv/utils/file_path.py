import os
from typing import List, Dict, Union, Optional


# going with what everyone is using
DEFAULT_MAPPING = {
        "checkpoint":       ["models/checkpoints"],
        "unet":            ["models/unet", "models/diffusion_models"],
        "lora":            ["models/loras"],
        "vae":             ["models/vae"],
        "clip":            ["models/clip"],
        "clip_vision":     ["models/clip_vision"],
        "style_model":     ["models/style_models"],
        "embedding":       ["models/embeddings"],
        "hypernetwork":    ["models/hypernetworks"],
        "controlnet":      ["models/controlnet"],
        "upscale_model":   ["models/upscale_models"],
        "gligen":          ["models/gligen"],
        "configs":         ["models/configs"],
        "photomaker":      ["models/photomaker"],
        "input":           ["input"],
        "output":          ["output"]
}


class FilePaths:
    # stores registered search paths
    # Format: [{'path': '/abs/path', 'map': {'type': ['folder1', 'folder2']}}]
    _search_roots = []
    
    # cache: root_path -> [list of all absolute file paths found recursively]
    _file_cache = {}
    
    OUTPUT_DIRECTORY = DEFAULT_MAPPING["output"][0]
    
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
                
        cls._search_roots.append({
            "path": os.path.abspath(path),
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
            for dirpath, _, filenames in os.walk(root):
                for f in filenames:
                    files_in_root.append(os.path.join(dirpath, f))
            
            cls._file_cache[root] = files_in_root

    @classmethod
    def get_files(cls, file_type: str, extensions: List[str] = None) -> List[str]:
        """
        Retrieves files matching the given type from all registered paths
        
        Args:
            file_type: The type of file to find (e.g., "lora", "embedding")
            extensions: Optional list of allowed extensions (e.g. ['.pt', '.safetensors'])
        """
        if not cls._file_cache:
            cls.init_cache()
            
        results = []
        
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
                    if extensions:
                        if any(f_path.lower().endswith(ext.lower()) for ext in extensions):
                            results.append(f_path)
                    else:
                        results.append(f_path)
                        
        return sorted(results)

    @classmethod
    def get_path(cls, filename: str, file_type: str) -> str:
        """
        - resolves the full path for a specific file name and type
        - prioritizes exact matches, then stem matches (ignoring extension)
        - validates that the file exists on disk
        """
        if not cls._file_cache:
            cls.init_cache()

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
                        return f_path
                
                # stem match (if input has no extension)
                if os.path.splitext(f_name)[0] == filename:
                     if os.path.exists(f_path):
                        candidates.append(f_path)

        if candidates:
            return candidates[0]

        raise FileNotFoundError(f"File '{filename}' of type '{file_type}' not found.")


FilePaths.add_search_path(".")