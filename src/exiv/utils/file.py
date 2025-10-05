import os
import re
import glob
import urllib.parse
import requests

from .logging import app_logger

def create_sanitized_path(file_path):
    # make sure directory exists
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # pattern to search existing images
    pattern = os.path.join(file_path, "img_*.jpg")
    fns = [fn for fn in glob.iglob(pattern) if re.search(r"img_[0-9]+\.jpg$", fn)]

    if fns:
        # extract highest index and increment
        idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
    else:
        idx = 0

    return os.path.join(file_path, f"img_{idx}.jpg")


def ensure_model_available(model_path: str, download_path: str = None, force_download: bool = False) -> str:
    """
    - Downloads model if a URL is provided, else verifies the local path.
    - Works with absolute paths (internally converts relative to absolute)
    - Store stuff in .cache if download path is not provided
    """
    parsed = urllib.parse.urlparse(model_path)

    if parsed.scheme in ("http", "https"):  # It's a URL
        if download_path is None:
            # default download path: same filename under ~/.cache/my_package/
            cache_dir = os.path.expanduser("~/.cache/my_package")
            os.makedirs(cache_dir, exist_ok=True)
            download_path = os.path.join(cache_dir, os.path.basename(parsed.path))

        if force_download or not os.path.exists(download_path):
            print(f"Downloading model from {model_path} to {download_path} ...")
            response = requests.get(model_path, stream=True)
            response.raise_for_status()
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            print(f"Model already exists at {download_path} (use force_download=True to overwrite)")
        return os.path.abspath(download_path)

    else: 
        abs_path = os.path.abspath(os.path.expanduser(model_path))
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Model file not found at {abs_path}")
        return abs_path

