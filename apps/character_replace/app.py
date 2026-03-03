import os
import torch
import json
import uuid

from exiv.utils.logging import app_logger
from exiv.components.extension_registry import ExtensionRegistry
from exiv.components import KSamplerType, SchedulerType, VAEType, KSampler
from exiv.components.cond_registry import preprocess_conds
from exiv.components.models.wan.constructor import get_wan_instance
from exiv.components.vae.base import get_vae
from exiv.model_patching.lora_hook import enable_lora_hook
from exiv.model_utils.common_classes import AuxConditioning, AuxCondType, Conditioning, BatchedConditioning, Latent, ModelWrapper
from exiv.model_utils.lora_mixin import LoraDefinition
from exiv.server.app_core import App, AppOutputType, Input, Output
from exiv.utils.device import MemoryManager
from exiv.utils.file import MediaProcessor, ensure_model_availability
from exiv.utils.file_path import FilePathData, FilePaths
from exiv.utils.common import fix_frame_count

VAE_DTYPE = torch.float16
USE_VAE_TILING = True

# Global cache to store temporary objects like masks between stateless API calls
SESSION_CACHE = {}

class WanAnimateMode:
    ANIMATION = "animation"
    REPLACEMENT = "replacement"

def create_dwpose_video(frames, dwpose_ext, detect_body, detect_hands, detect_face):
    out_frames = []
    for i, frame in enumerate(frames):
        app_logger.info(f"Processing DWPose frame {i+1}/{len(frames)}")
        out_tensor = dwpose_ext.process(image=frame, detect_body=detect_body, detect_hand=detect_hands, detect_face=detect_face)
        out_frames.append(out_tensor)
    out_video = torch.stack(out_frames).permute(1, 0, 2, 3).unsqueeze(0)
    output_paths = MediaProcessor.save_latents_to_media(out_video, media_type="video", subfolder="dwpose")
    return output_paths[0]

def main(**params):
    context = params.get("context")
    app_mode = params.get("app_mode", "0_init")
    input_video = params.get("input_video", "")
    
    registry = ExtensionRegistry.get_instance()
    
    def progress_callback(progress_fraction, stage): 
        app_logger.debug(f"Percent: {progress_fraction}  -- Stage: {stage}")
        if context:
            context.progress(progress_fraction, "Processing", stage=stage) 

    if app_mode == "0_init":
        if not input_video or not os.path.exists(input_video):
            raise ValueError(f"Input video not found: {input_video}")
            
        video_tensor, metadata = MediaProcessor.load_video(input_video, output_frames=False)
        first_frame = video_tensor[:, 0, :, :] # (C, H, W)
        preview_tensor = first_frame.unsqueeze(0).unsqueeze(2) # (1, C, 1, H, W)
        preview_paths = MediaProcessor.save_latents_to_media(preview_tensor, media_type="image", subfolder="temp")
        
        session_id = str(uuid.uuid4())
        return {"1": {"first_frame": preview_paths[0], "session_id": session_id}}

    elif app_mode == "1_segment":
        matanyone_cls = registry.extensions.get("matanyone")
        if not matanyone_cls:
            raise ValueError("MatAnyone extension is not available. Install it from github.com/piyushK52/exiv_matanyone")
        matanyone_ext = matanyone_cls()
            
        session_id = params.get("session_id", str(uuid.uuid4()))
        points_str = params.get("points", "[]")
        labels_str = params.get("labels", "[]")
        
        points = json.loads(points_str)
        labels = json.loads(labels_str)
        
        if not input_video or not os.path.exists(input_video):
            raise ValueError(f"Input video not found: {input_video}")
            
        video_tensor, metadata = MediaProcessor.load_video(input_video, output_frames=False)
        first_frame = video_tensor[:, 0, :, :]
        
        seg_res = matanyone_ext.process(
            mode="segment_frame", 
            reference_image=first_frame, 
            points=points, 
            labels=labels
        )
        
        mask = seg_res["mask"]
        SESSION_CACHE[session_id] = mask
        
        preview_tensor = seg_res["preview"]
        preview_tensor = preview_tensor.unsqueeze(0).unsqueeze(2)
        preview_paths = MediaProcessor.save_latents_to_media(preview_tensor, media_type="image", subfolder="temp")
        
        return {"1": {"preview": preview_paths[0], "session_id": session_id}}

    elif app_mode == "4_animate":
        if context: context.start_anchor("Setup", steps=1)
        
        pos_prompt = params.get("positive", "a man dancing in a studio, high quality")
        neg_prompt = params.get("negative", "bad quality, blurry, distorted, disfigured")
        seed = params.get("seed", 42)
        steps = params.get("steps", 4)
        cfg = params.get("cfg", 1.0)
        sampler_name = params.get("sampler_name", KSamplerType.EULER.value)
        scheduler_name = params.get("scheduler_name", SchedulerType.SIMPLE.value)
        
        height = params.get("height", 640)
        width = params.get("width", 640)
        frame_count = params.get("frame_count", 81)

        ref_img_path = params.get("reference_image", "")
        pose_video_path = params.get("pose_video", "")
        face_video_path = params.get("face_video", "")
        bg_video_path = params.get("bg_video", "")
        mask_video_path = params.get("mask_video", "")
        session_id = params.get("session_id", "")
        input_video = params.get("input_video", "")
        
        if input_video and (not bg_video_path or not mask_video_path):
            matanyone_cls = registry.extensions.get("matanyone")
            if matanyone_cls:
                mask = SESSION_CACHE.get(session_id)
                if mask is not None:
                    if context: context.start_anchor("Background Matting", steps=1)
                    matanyone_ext = matanyone_cls()
                    video_tensor, metadata = MediaProcessor.load_video(input_video, output_frames=False)
                    matte_res = matanyone_ext.process(
                        mode="matte_video", 
                        video=video_tensor, 
                        mask=mask,
                        output_type="black_background",
                        mask_padding=10,
                        blocky_mask=True,
                        binary_mask=True,
                    )
                    bg_video_path = MediaProcessor.save_latents_to_media(matte_res["foregrounds"], subfolder="matanyone_fg")[0]
                    mask_video_path = MediaProcessor.save_latents_to_media(matte_res["alphas"], subfolder="matanyone_mask")[0]

        if input_video and (not pose_video_path or not face_video_path):
            if context: context.start_anchor("Pose Detection", steps=1)
            dwpose_cls = registry.extensions.get("dwpose")
            if dwpose_cls:
                dwpose_ext = dwpose_cls()
                frames, metadata = MediaProcessor.load_video(input_video, output_frames=True, fps=16)
                pose_video_path = create_dwpose_video(frames, dwpose_ext, detect_body=True, detect_hands=True, detect_face=False)
                face_video_path = create_dwpose_video(frames, dwpose_ext, detect_body=False, detect_hands=False, detect_face=True)

        if not bg_video_path:
            bg_video_path = input_video

        print("------ ref_image_path: ", ref_img_path)
        print("------ pose_video_path: ", pose_video_path)
        print("------ face_video_path: ", face_video_path)
        print("------ bg_video_path: ", bg_video_path)
        print("------ mask_video_path: ", mask_video_path)

        if bg_video_path and mask_video_path:
            mode = WanAnimateMode.REPLACEMENT
        else:
            mode = WanAnimateMode.ANIMATION
            bg_video_path = ""
            mask_video_path = ""
        
        frame_count = fix_frame_count(frame_count, 4)
        
        if context: context.start_anchor("Preprocessing", steps=6)
        
        model_name = "wan22_animate_14b_fp8_e4m3_scaled"
        model_path_data = FilePaths.get_path(filename=model_name, file_type="checkpoint")
        wan_dit_model = get_wan_instance(model_path_data.path, model_path_data.url, force_dtype=torch.float16)
        
        model_path_data = FilePaths.get_path(filename="wan_animate_lightx_cfg_step_distill_lora.safetensors", file_type="lora")
        lora_path = ensure_model_availability(model_path=model_path_data.path, download_url=model_path_data.url)
        lora_def = LoraDefinition(path=lora_path)
        enable_lora_hook(model=wan_dit_model, lora_def=lora_def)
        model_wrapper = ModelWrapper(model=wan_dit_model)
        
        def create_cond(group_name, prompt, chunk_offset=0, temporal_latent_pixel=None, max_overlap=0):
            cond = Conditioning(
                group_name=group_name,
                input_metadata=prompt,
                extra={
                    "animate_mode": mode,
                    "background_video_path": bg_video_path,
                    "character_mask_path": mask_video_path,
                    "frame_offset": chunk_offset,
                    "temporal_latent": temporal_latent_pixel,
                    "max_overlap": max_overlap
                }
            )
            cond.aux = [
                AuxConditioning(type=AuxCondType.REF_LATENT, input_metadata=ref_img_path),
                AuxConditioning(type=AuxCondType.VISUAL_EMBEDDING, input_metadata=ref_img_path),
                AuxConditioning(type=AuxCondType.POSE_LATENTS, input_metadata=pose_video_path),
                AuxConditioning(type=AuxCondType.FACE_PIXEL_VALUES, input_metadata=face_video_path)
            ]
            return cond

        wan_vae = get_vae(VAEType.WAN21.value, VAE_DTYPE, USE_VAE_TILING)
        latent_format = model_wrapper.model.model_arch_config.latent_format
        
        if context: context.start_anchor("Sampling", steps=12)
        bs = 81
        max_overlap = 8
        generated_frames = 0
        all_pixels = []
        temporal_latent_pixel = None
        
        import math
        num_chunks = max(1, math.ceil((frame_count - max_overlap) / (bs - max_overlap))) if frame_count > bs else 1
        chunk_idx = 0

        while generated_frames < frame_count:
            chunk_frames = bs if generated_frames == 0 else min(bs, frame_count - generated_frames + max_overlap)
            chunk_frames = fix_frame_count(chunk_frames, 4)
            current_offset = generated_frames
            current_overlap = max_overlap if generated_frames > 0 else 0

            cond_list = [
                create_cond("positive", pos_prompt, current_offset, temporal_latent_pixel, current_overlap),
                create_cond("negative", neg_prompt, current_offset, temporal_latent_pixel, current_overlap)
            ]
            
            def chunk_progress_callback(p, s, stage_offset=0.0, stage_weight=1.0):
                # stage_offset: 0.0 for preprocessing, 0.2 for sampling
                # stage_weight: 0.2 for preprocessing, 0.8 for sampling
                local_p = (stage_offset + p * stage_weight) / num_chunks
                global_p = (chunk_idx / num_chunks) + local_p
                progress_callback(global_p, s)

            batched_cond: BatchedConditioning = preprocess_conds(
                model_wrapper=model_wrapper,
                cond_list=cond_list,
                height=height, 
                width=width, 
                frame_count=chunk_frames,
                cfg=cfg,
                progress_callback=lambda p, s: chunk_progress_callback(p, s, stage_offset=0.0, stage_weight=0.2)
            )
            
            latent = Latent() 
            latent.encode_keyframe_condition( 
                width, 
                height,
                chunk_frames, 
                latent_format, 
                wan_vae,
            )
            print("------- steps found: ", steps)
            sampler = KSampler(
                wrapped_model=model_wrapper,
                seed=seed,
                total_steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler_name=scheduler_name,
                batched_conditioning=batched_cond,
                latent_image=latent
            )
            out = sampler.run_sampling(callback=lambda i, s: chunk_progress_callback(i, s, stage_offset=0.2, stage_weight=0.8))
            
            decoded_chunk = wan_vae.decode(out.to(dtype=VAE_DTYPE), (width, height, chunk_frames))
            frames_to_clip = 8
            all_pixels.append(decoded_chunk[:, :, frames_to_clip:])
            if generated_frames + chunk_frames - current_overlap < frame_count:
                temporal_latent_pixel = decoded_chunk[0, :, -max_overlap:].permute(1, 0, 2, 3)

            generated_frames += chunk_frames if generated_frames == 0 else chunk_frames - max_overlap
            chunk_idx += 1

        out = torch.cat(all_pixels, dim=2)
        
        wan_dit_model.to("cpu")
        del wan_dit_model, model_wrapper
        MemoryManager.clear_memory()
        
        metadata_dict = {"positive": pos_prompt, "seed": seed, "mode": mode, "model": "Wan2.2 Animate"}
        out_paths = MediaProcessor.save_latents_to_media(out, metadata=metadata_dict, debug=False)
        return {"1": {
            "final_video": out_paths[0],
            "fg_video": bg_video_path,
            "mask_video": mask_video_path,
            "pose_video": pose_video_path,
            "face_video": face_video_path
        }}
        
    else:
        raise ValueError(f"Unknown app_mode: {app_mode}")

app = App(
    name="Character Replace",
    inputs={
        'app_mode': Input(label="App Mode", type="select", options=["0_init", "1_segment", "4_animate"], default="0_init"),
        'input_video': Input(label="Input Video", type="str", default=""),
        'session_id': Input(label="Session ID", type="str", default=""),
        'points': Input(label="Points", type="str", default="[]"),
        'labels': Input(label="Labels", type="str", default="[]"),
        
        'reference_image': Input(label="Reference Image", type="str", default=""),
        'bg_video': Input(label="Background Video", type="str", default=""),
        'mask_video': Input(label="Mask Video", type="str", default=""),
        'pose_video': Input(label="Pose Video", type="str", default=""),
        'face_video': Input(label="Face Video", type="str", default=""),
        
        'positive': Input(label="Positive Prompt", type="str", default="a girl talking", resizable=True),
        'negative': Input(label="Negative Prompt", type="str", default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", resizable=True),
        'seed': Input(label="Seed", type="number", default=-1),
        'steps': Input(label="Steps", type="number", default=4),
        'cfg': Input(label="CFG", type="number", default=1.0, increment_step=0.1),
        'height': Input(label="Height", type="number", default=640),
        'width': Input(label="Width", type="number", default=640),
        'frame_count': Input(label="Frame Count", type="number", default=81),
        'sampler_name': Input(label="Sampler", type="select", options=KSamplerType.value_list(), default=KSamplerType.EULER.value),
        'scheduler_name': Input(label="Scheduler", type="select", options=SchedulerType.value_list(), default=SchedulerType.SIMPLE.value),
    },
    outputs=[Output(id=1, type=AppOutputType.JSON.value)],
    extra_metadata={'preserve_state': True},
    handler=main
)

if __name__ == "__main__":
    app.run_standalone()