import torch
import numpy as np
import os
import sys

from exiv.components.extension_registry import ExtensionRegistry
from exiv.server.app_core import App, AppOutputType, Input, Output
from exiv.utils.logging import app_logger
from exiv.utils.file import MediaProcessor

def main(**params):
    """
    Main handler for the Extension Tester App.
    This app demonstrates how to load and use the DWPose and MatAnyone extensions.
    Note: Actual processing is skipped as per user request for cloud execution.
    """
    # 1. Initialize the extension registry
    # This will load extensions defined in exiv_config.json
    registry = ExtensionRegistry.get_instance()
    registry.initialize()
    
    # 2. Access the extensions by their IDs
    dwpose_ext = registry.extensions.get("dwpose")
    matanyone_ext = registry.extensions.get("matanyone")
    status_report = []

    # Inputs from UI/CLI
    image_path = "./tests/test_utils/assets/media/boy_anime.jpg"
    video_path = "input.mp4"
    run_dwpose = params.get("run_dwpose", False)
    run_matanyone = params.get("run_matanyone", False)

    results = {
        "extension_status": status_report,
        "actions_simulated": []
    }

    # Example of how DWPose would be called
    if run_dwpose and dwpose_ext:
        if image_path and os.path.exists(image_path):
            app_logger.info(f"[SIMULATION] Would run DWPose on: {image_path}")
            image = MediaProcessor.load_image_list(image_path)[0]
            out_tensor = dwpose_ext.process(image=image, detect_hand=True, detect_face=True)
            out_tensor = out_tensor.unsqueeze(0).unsqueeze(2)
            output_paths = MediaProcessor.save_latents_to_media(out_tensor, media_type="image")
            results["actions_simulated"].append(f"DWPose saved to: {output_paths[0]}")
            app_logger.info(f"Pose map saved to {output_paths[0]}") 
        else:
            app_logger.info("[SIMULATION] DWPose selected but no valid image_path provided.")

    # Actual MatAnyone execution
    if run_matanyone and matanyone_ext:
        video_path = "extensions/exiv_matanyone/sample_videos/sample_input.mp4"
        if os.path.exists(video_path):
            app_logger.info(f"Running MatAnyone on: {video_path}")
            
            # 1. Load video and get first frame
            video_tensor, metadata = MediaProcessor.load_video(video_path, output_frames=False)
            first_frame = video_tensor[:, 0, :, :] # (C, H, W)
            
            # 2. Segment first frame (Center click)
            h, w = first_frame.shape[1:]
            points = [[w // 2, h // 2]]
            labels = [1]
            
            app_logger.info("Step 1: Segmenting first frame...")
            seg_res = matanyone_ext.process(
                mode="segment_frame", 
                reference_image=first_frame, 
                points=points, 
                labels=labels
            )
            
            mask = seg_res["mask"]
            preview_tensor = seg_res["preview"]
            
            # Save segmentation preview using save_latents_to_media
            # (C, H, W) -> (1, C, 1, H, W)
            preview_tensor = preview_tensor.unsqueeze(0).unsqueeze(2)
            preview_paths = MediaProcessor.save_latents_to_media(preview_tensor, media_type="image")
            app_logger.info(f"Segmentation preview saved to: {preview_paths[0]}")

            # 3. Video Matting (limit to 10 frames for speed)
            app_logger.info("Step 2: Running Video Matting (10 frames)...")
            short_video = video_tensor[:, :10, :, :]
            matte_res = matanyone_ext.process(
                mode="matte_video", 
                video=short_video, 
                mask=mask
            )
            
            # 4. Save results (tensors already in correct format)
            fg_tensor = matte_res["foregrounds"]
            al_tensor = matte_res["alphas"]

            fg_paths = MediaProcessor.save_latents_to_media(fg_tensor, subfolder="matanyone")
            al_paths = MediaProcessor.save_latents_to_media(al_tensor, subfolder="matanyone")
            
            results["actions_simulated"].append(f"MatAnyone foreground saved to: {fg_paths[0]}")
            results["actions_simulated"].append(f"MatAnyone alpha saved to: {al_paths[0]}")
        else:
            app_logger.info(f"MatAnyone selected but video not found at: {video_path}")

    app_logger.info("Extension Tester App execution complete.")
    return {"1": results}

# Define the App interface
app = App(
    name="Extension Tester",
    inputs={
        'run_dwpose': Input(
            label="Run DWPose", 
            type="boolean", 
            default=True
        ),
        'run_matanyone': Input(
            label="Run MatAnyone", 
            type="boolean", 
            default=True
        ),
    },
    outputs=[Output(id=1, type=AppOutputType.JSON.value)],
    handler=main
)

if __name__ == "__main__":
    app.run_standalone()
