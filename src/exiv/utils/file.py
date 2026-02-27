from __future__ import annotations
import os
import re
import glob
import urllib.parse
import requests
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:   # mainly for IDE suggestions
    import torch
    import numpy as np
    import av


CONFIG_FILENAME = "exiv_config.json"

from .file_path import FilePaths
from ..utils.common import is_ffmpeg_present


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

def find_file_path(filename: str, start_path: str = None, recursive: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    - Searches for a file starting from start_path (or cwd)
    - If recursive is True, it walks up the directory tree until it finds the file or hits root
    Returns (abs_path_to_file, directory_containing_file)
    """
    if start_path is None:
        start_path = os.getcwd()
        
    current = os.path.abspath(start_path)
    
    while True:
        check_path = os.path.join(current, filename)
        if os.path.exists(check_path):
            return check_path, current
        
        if not recursive:
            break

        parent = os.path.dirname(current)
        if parent == current:  # reached root
            break
        current = parent
        
    return None, None

def _interactive_download_check(model_path: str, download_url: str) -> bool:
    from .logging import app_logger
    from ..config import global_config
    from .logging import app_logger
    from ..config import global_config
    import requests

    if global_config.auto_download:
        return True

    # fetch file size
    size_label = "Unknown size"
    try:
        head_response = requests.head(download_url, allow_redirects=True, timeout=5)
        if 'content-length' in head_response.headers:
            size_bytes = int(head_response.headers['content-length'])
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_bytes < 1024.0:
                    size_label = f"{size_bytes:.2f} {unit}"
                    break
                size_bytes /= 1024.0
    except Exception as e:
        app_logger.warning(f"Could not retrieve file size: {e}")

    # CLI prompt
    print(f"\n[Exiv] File missing: {os.path.basename(model_path)}")
    while True:
        user_input = input(f"Do you want to auto download this file ({size_label})? [yes/no/always]: ").strip().lower()
        
        if user_input in ("yes", "y"):
            return True
        
        elif user_input == "always":
            global_config.auto_download = True
            app_logger.info("Auto-download enabled for this session.")
            return True
        
        elif user_input in ("no", "n"):
            return False

def ensure_model_availability(model_path: str, download_url: str = None, force_download: bool = False) -> str:
    """
    - Downloads model if a URL is provided, else verifies the local path.
    - Works with absolute paths (internally converts relative to absolute)
    - Store stuff in .cache if download path is not provided
    """
    from .logging import app_logger
    
    assert model_path is not None and model_path != "", "model_path provided can't be None or empty"
    
    if download_url:  # It's a URL
        parsed = urllib.parse.urlparse(download_url)
        assert parsed.scheme in ("http", "https"), "invalid download link"

    if download_url and (force_download or not os.path.exists(model_path)):
        should_download = _interactive_download_check(model_path, download_url)
        if not should_download:
            raise FileNotFoundError(f"Download cancelled by user. Model {model_path} is required.")
        
        app_logger.info(f"Downloading model from {download_url} to {model_path} ...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        from tqdm import tqdm
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
    def load_image_list(image_path_list: List[str] | str):
        from PIL import Image
        from PIL import Image
        from .logging import app_logger
        import torch
        import numpy as np
        
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

        return res
    
    @staticmethod
    def _frame_to_tensor(frame) -> torch.Tensor:
        import numpy as np
        import torch
        np_frame = frame.to_ndarray(format='rgb24')
        np_frame = np_frame.astype(np.float32) / 255.0
        return torch.from_numpy(np_frame.transpose(2, 0, 1))

    @staticmethod
    def _resample_frames(container, fps: float | None, limit_frame_count: int | None, orig_fps: float) -> List[torch.Tensor]:
        frames = []
        if fps is None:
            for i, frame in enumerate(container.decode(video=0)):
                if limit_frame_count is not None and len(frames) >= limit_frame_count:
                    break
                frames.append(MediaProcessor._frame_to_tensor(frame))
            return frames

        target_frame_time = 1.0 / fps
        next_target_time = None
        start_time = None
        last_frame_tensor = None

        for i, frame in enumerate(container.decode(video=0)):
            # Use PTS or fallback to index-based timing
            current_time = frame.time if frame.time is not None else i / orig_fps 

            if start_time is None:
                start_time = current_time
                next_target_time = start_time

            current_tensor = MediaProcessor._frame_to_tensor(frame)

            # Fill frames for target times before current frame (Sample and Hold)
            while next_target_time < current_time - 1e-5:
                if last_frame_tensor is not None:
                    frames.append(last_frame_tensor)
                    if limit_frame_count is not None and len(frames) >= limit_frame_count:
                        return frames
                
                next_target_time += target_frame_time

            last_frame_tensor = current_tensor
            
        # Emit last frame if needed to catch up
        if last_frame_tensor is not None and next_target_time is not None:
             frames.append(last_frame_tensor)
        
        return frames

    @staticmethod
    def load_video(video_path: str, output_frames: bool = True, limit_frame_count: int | None = None, fps: float | None = None):
        """
        Loads a video and returns (frames, metadata)
        output_frames: return frame list if True, the video tensor otherwise
        limit_frame_count: Optional integer to stop loading after N frames
        fps: Optional float to specify the target frames per second.
             If provided, frames will be sampled (or duplicated) to match this FPS.
        
        Returns: 
            video_tensor: (C, T, H, W) float32 tensor in [0, 1] range, or None
            metadata: Dict containing fps, resolution, duration, etc
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        import av
        import numpy as np
        import torch

        container = av.open(video_path)
        stream = container.streams.video[0]
        
        orig_fps = float(stream.average_rate)
        metadata = {
            "fps": fps if fps is not None else orig_fps,
            "original_fps": orig_fps,
            "resolution": (stream.width, stream.height),
            "duration": float(stream.duration * stream.time_base) if stream.duration else 0.0,
            "total_frames_in_file": stream.frames
        }

        frames = MediaProcessor._resample_frames(container, fps, limit_frame_count, orig_fps)

        video_tensor = None
        if frames:
            metadata["loaded_frames"] = len(frames)
            if not output_frames:
                video_tensor = torch.stack(frames)                  # stack -> (T, C, H, W)
                video_tensor = video_tensor.permute(1, 0, 2, 3)     # permute -> (C, T, H, W)
            else:
                video_tensor = frames
        else:
            metadata["loaded_frames"] = 0

        container.close()
        return video_tensor, metadata
    
    @staticmethod
    def _draw_debug_frame_numbers(video_formatted, start_frame: int):
        import torch
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # font size = 5% of height or min 16
        frame_height = video_formatted.shape[1]
        font_size = max(16, int(frame_height * 0.05))
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            # for older PIL versions
            font = ImageFont.load_default()
            
        stroke_width = max(1, font_size // 15)
        text_color = (0, 0, 0)
        border_color = (255, 255, 255)

        for f_idx in range(len(video_formatted)):
            frame_np = video_formatted[f_idx].numpy()
            img = Image.fromarray(frame_np)
            draw = ImageDraw.Draw(img)
            text = str(f_idx + start_frame)
            
            draw.text(
                (10, 10), 
                text, 
                fill=text_color, 
                font=font, 
                stroke_width=stroke_width, 
                stroke_fill=border_color
            )
            video_formatted[f_idx] = torch.from_numpy(np.array(img))
        return video_formatted

    @staticmethod
    def save_latents_to_media(out, metadata: Dict | None = None, subfolder: str | None = None, start_frame = 0, end_frame = None, media_type = "video", fps=16, debug: bool = False):
        # TODO: make this a generic method, allowing saving images/audio/3d as well
        import torch
        video_tensor = out.sample if hasattr(out, "sample") else out

        # IMPORTANT: VAE outputs should always be in [0, 1]
        # rescale to [0, 255] and cast to uint8
        video_tensor = (video_tensor.clamp(0, 1) * 255).to(torch.uint8)
        output_paths = []
        # current shape: (Batch, Channels, Time, Height, Width) -> e.g., (1, 3, 121, 512, 768)
        for i, video in enumerate(video_tensor):
            # (C, T, H, W) -> (T, H, W, C), for torchvision
            video_formatted = video.permute(1, 2, 3, 0).cpu()
            if end_frame is None or end_frame == -1:
                video_formatted = video_formatted[start_frame:]
            else:
                video_formatted = video_formatted[start_frame:end_frame]

            save_dir = FilePaths.OUTPUT_DIRECTORY
            if subfolder:
                save_dir = os.path.join(save_dir, subfolder)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            
            if media_type == "image":
                save_path = f"output_image_{i}.png"
                save_path = get_numbered_filename(save_dir, save_path)
                from PIL import Image
                import numpy as np
                # video_formatted is (T, H, W, C). For image, T should be 1 or we take the first frame.
                img_np = video_formatted[0].numpy()
                Image.fromarray(img_np).save(save_path)
            else:
                save_path = f"output_video_{i}.mp4"
                save_path = get_numbered_filename(save_dir, save_path)
                import torchvision

                if debug:
                    video_formatted = MediaProcessor._draw_debug_frame_numbers(video_formatted, start_frame)

                torchvision.io.write_video(
                    save_path,
                    video_formatted,
                    fps=fps,
                    options={"crf": "25"}  # 'Constant Rate Factor' for quality (lower is better)
                )
            
            try:
                if metadata:
                    MediaProcessor.save_metadata(save_path, metadata)
            except Exception as e:
                from ..utils.logging import app_logger
                app_logger.warning(f"Unable to write metadata in {save_path}: {e}")
            
            rel_path = os.path.relpath(save_path, FilePaths.OUTPUT_DIRECTORY)
            output_paths.append(rel_path)
            
        return output_paths

    # NOTE: maybe a pure pythonic way will be more robust
    @staticmethod
    def save_metadata(file_path: str, metadata: Dict):
        ext = os.path.splitext(file_path)[1].lower()
        
        # standardize metadata
        clean_metadata = {k: str(v) for k, v in metadata.items()}

        # image support (e.g. PNG via Pillow)
        if ext == '.png':
            try:
                from PIL import Image, PngImagePlugin
                with Image.open(file_path) as img:
                    info = PngImagePlugin.PngInfo()
                    # preserve existing info
                    for k, v in img.info.items():
                        info.add_text(k, str(v))
                    # add new metadata
                    for k, v in clean_metadata.items():
                        info.add_text(k, v)
                    img.save(file_path, pnginfo=info)
                return
            except Exception as e:
                print(f"Pillow PNG metadata save failed: {e}")

        temp_path = file_path + f".temp{ext}"
        
        options = {}
        if ext in ('.mp4', '.mov', '.m4v'):
            options = {'movflags': 'use_metadata_tags'}

        try:
            import av
            with av.open(file_path) as input_container:
                with av.open(temp_path, mode='w', options=options) as output_container:
                    # update global metadata
                    full_metadata = dict(input_container.metadata)
                    full_metadata.update(clean_metadata)
                    output_container.metadata.update(full_metadata)

                    # copy streams (video/audio/subtitle)
                    stream_mapping = {}
                    for stream in input_container.streams:
                        if stream.type not in ('video', 'audio', 'subtitle'):
                            continue

                        # NOTE: PyAV v14+ uses add_stream_from_template, older versions used template=stream
                        if hasattr(output_container, 'add_stream_from_template'):
                            out_stream = output_container.add_stream_from_template(stream)
                        else:
                            # fallback for older PyAV versions
                            try:
                                out_stream = output_container.add_stream(template=stream)
                            except (TypeError, AttributeError):
                                # manual fallback if template arg fails
                                out_stream = output_container.add_stream(stream.codec_context.name)
                                out_stream.time_base = stream.time_base
                                if stream.type == 'video':
                                    out_stream.width = stream.codec_context.width
                                    out_stream.height = stream.codec_context.height
                                    out_stream.pix_fmt = stream.codec_context.pix_fmt
                        
                        stream_mapping[stream.index] = out_stream

                    # mux packets (remuxing)
                    for packet in input_container.demux():
                        if packet.stream.index in stream_mapping:
                            if packet.dts is None: continue
                            packet.stream = stream_mapping[packet.stream.index]
                            output_container.mux(packet)
            
            os.replace(temp_path, file_path)
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"Error saving metadata: {e}")

    @staticmethod
    def get_metadata(file_path: str) -> Dict:
        ext = os.path.splitext(file_path)[1].lower()
        
        # prefer Pillow for standard image formats
        if ext in ('.png', '.jpg', '.jpeg', '.webp'):
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    # return info dict, ensuring values are strings
                    return {k: str(v) for k, v in img.info.items()}
            except Exception:
                pass

        try:
            import av
            with av.open(file_path) as container:
                return dict(container.metadata)
        except Exception:
            return {}