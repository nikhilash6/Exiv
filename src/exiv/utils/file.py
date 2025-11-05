import torch
import numpy as np

import os
import re
import glob
import urllib.parse
import requests
from typing import List

from tqdm import tqdm

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

def ensure_model_available(model_path: str, download_url: str = None, force_download: bool = False) -> str:
    """
    - Downloads model if a URL is provided, else verifies the local path.
    - Works with absolute paths (internally converts relative to absolute)
    - Store stuff in .cache if download path is not provided
    """
    
    if download_url:  # It's a URL
        parsed = urllib.parse.urlparse(download_url)
        assert parsed.scheme in ("http", "https"), "invalid download link"
    
    # TODO: fix this, have proper model paths defined
    if model_path is None:
        if not download_url:
            raise ValueError("Must provide either a local 'model_path' or a 'download_url'.")
        # default download path: same filename under ~/.cache/my_package/
        cache_dir = os.path.expanduser("~/.cache/my_package")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, os.path.basename(parsed.path))

    if download_url and (force_download or not os.path.exists(model_path)):
        app_logger.info(f"Downloading model from {download_url} to {model_path} ...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(model_path)) as pbar:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
    abs_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Model file not found at {abs_path}")
    return abs_path
        

class ImageProcessor:
    @staticmethod
    def load_image_list(image_path_list: List[str]):
        from PIL import Image
        
        # loads in the torch tensor format
        res = []
        for img_path in image_path_list:
            try:
                pil_img = Image.open(img_path)
            except Exception as e:
                app_logger.warning(str(e))
                continue
            
            np_img = np.array(pil_img).astype(np.float32) / 255.0
            pt_img = torch.from_numpy(np_img.transpose(0, 3, 1, 2))
            res.append(pt_img)
        
        return res