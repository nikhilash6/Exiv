import os
import iglob
import re
import urllib.parse
import requests

from .logging import app_logger

def create_sanitized_path(file_path):
    filename = os.path.join(file_path, "img_{idx}.jpg")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        idx = 0
    else:
        # if same filename is already present then increase the idx
        fns = [fn for fn in iglob(filename.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0
            
    return idx

def ensure_model_available(model_path: str, download_path: str, force_download: bool = False) -> str:
    # downloads model if a url is provided else verifies the local path
    parsed = urllib.parse.urlparse(model_path)

    if parsed.scheme in ("http", "https"):  # It's a URL
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
        return download_path
    
    else:  # Local file path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return model_path
