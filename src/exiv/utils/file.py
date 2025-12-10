import torch
import torchvision
import numpy as np

import os
import re
import glob
import urllib.parse
import requests
from typing import List
from tqdm import tqdm

from .file_path import FilePaths


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

def get_numbered_filename(folder: str, filename: str) -> str:
    """
    Returns a unique full path. If 'folder/filename' exists, 
    it appends a number (e.g., '_1', '_2') to the base name
    """
    base, ext = os.path.splitext(filename)
    full_path = os.path.join(folder, filename)
    
    counter = 1
    while os.path.exists(full_path):
        new_filename = f"{base}_{counter}{ext}"
        full_path = os.path.join(folder, new_filename)
        counter += 1
        
    return full_path

def ensure_model_availability(model_path: str, download_url: str = None, force_download: bool = False) -> str:
    """
    - Downloads model if a URL is provided, else verifies the local path.
    - Works with absolute paths (internally converts relative to absolute)
    - Store stuff in .cache if download path is not provided
    """
    from .logging import app_logger
    
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
        

class MediaProcessor:
    @staticmethod
    def load_image_list(image_path_list: List[str]):
        from .logging import app_logger
        
        from PIL import Image
        
        if isinstance(image_path_list, str):
            image_path_list = [image_path_list]

        res = []
        for img_path in image_path_list:
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                app_logger.warning(str(e))
                continue
            
            # Converts H x W x C (0-255) to H x W x C (0.0-1.0)
            np_img = np.array(pil_img).astype(np.float32) / 255.0

            # Transposes H x W x C -> C x H x W
            pt_img = torch.from_numpy(np_img.transpose(2, 0, 1)) 
            res.append(pt_img)
        
        if not res:
            # returning empty tensor for now, (needs changing?)
            return torch.empty(0) 
        
        return torch.stack(res, dim=0)
    
    @staticmethod
    def save_latents_to_media(out):
        # TODO: make this a generic method, allowing saving images/audio/3d as well
        # rn it is only for video
        video_tensor = out.sample if hasattr(out, "sample") else out

        # rescale from [-1, 1] to [0, 255] and cast to uint8
        video_tensor = ((video_tensor.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

        output_paths = []
        # current shape: (Batch, Channels, Time, Height, Width) -> e.g., (1, 3, 121, 512, 768)
        for i, video in enumerate(video_tensor):
            # (C, T, H, W) -> (T, H, W, C), for torchvision
            video_formatted = video.permute(1, 2, 3, 0).cpu()
            
            save_path = f"output_video_{i}.mp4"
            save_path = get_numbered_filename(FilePaths.OUTPUT_DIRECTORY, save_path)
            torchvision.io.write_video(
                save_path,
                video_formatted,
                fps=24,
                options={"crf": "5"}  # 'Constant Rate Factor' for quality (lower is better)
            )
            output_paths.append(save_path)
            
        return output_paths