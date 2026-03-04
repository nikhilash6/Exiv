import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image
from typing import Callable, Dict, List, Optional

from exiv.utils.file import MediaProcessor

from ...latent_format import Wan21VAELatentFormat, Wan22VAELatentFormat
from ....model_utils.model_mixin import ModelArchConfig
from ....utils.device import VRAM_DEVICE, MemoryManager
from ....utils.tensor import common_upscale, get_tensor_weak_hash
from ...enum import Model, VAEType
from ...cond_registry import fix_keyframe_indexing, get_image_tensor, get_text_embeddings, get_vision_embeddings, register_preprocessor
from ...text_vision_encoder.common import TextEncoderOutput, VisionEncoderOutput
from ...vae.base import VAEBase, get_vae
from ....model_utils.common_classes import AuxCondType, AuxConditioning, BatchedConditioning, Conditioning, ExtraCond, Latent, ModelWrapper
from ....utils.common import fix_frame_count, null_func

VAE_DTYPE = torch.float16

is_vace_model = lambda model_type: model_type in [Model.WAN21_VACE_1_3B_R2V.value, Model.WAN21_VACE_14B_R2V.value]
is_text_model = lambda model_type: model_type in [Model.WAN21_1_3B_T2V.value, Model.WAN22_5B_T2V.value, Model.WAN22_14B_TI2V.value] or is_vace_model(model_type)
is_img_model = lambda model_type: not is_text_model(model_type)

class Wan21ModelArchConfig(ModelArchConfig):
    def __init__(self, model_type):
        self.model_type = model_type
        self.latent_format = Wan21VAELatentFormat()
        
        # default models
        self.default_vae_type = VAEType.WAN21.value
        self.default_text_encoder = "umt5_xxl_fp16.safetensors"
        self.default_vision_encoder = "CLIP-ViT-H-fp16.safetensors"
        
    def get_ref_latent(self, image, vae, length, width, height):
        # 16 ch VAE latent + 4 ch mask
        if is_text_model(self.model_type) and self.model_type != Model.WAN22_14B_TI2V.value: return None
        image = common_upscale(image, width, height, "bilinear", "center")[0]
        video = torch.ones((1, 3, length, height, width), device=image.device, dtype=image.dtype) * 0.5
        video[:, :, 0, :, :] = image
        video = video.to(dtype=vae.dtype)
        concat_latent_image = vae.encode(video)
        concat_latent_image = self.latent_format.process_in(concat_latent_image)
        mask = torch.zeros(
            (
                1, 
                4, 
                ((length - 1) // vae.temporal_compression_ratio) + 1, 
                concat_latent_image.shape[-2], 
                concat_latent_image.shape[-1]
            ), 
            device=image.device,
            dtype=image.dtype
        )
        mask[:, :, :((image.shape[0] - 1) // vae.temporal_compression_ratio) + 1] = 1.0
        mask = mask.to(VRAM_DEVICE)
        concat_latent_image = concat_latent_image.to(VRAM_DEVICE)
        conditioning = torch.cat([mask, concat_latent_image], dim=1)
        return conditioning

class Wan22ModelArchConfig(Wan21ModelArchConfig):
    def __init__(self, model_type=Model.WAN22_5B_T2V.value):
        super().__init__(model_type)
        self.model_type = model_type
        
        if model_type == Model.WAN22_14B_TI2V.value:
            self.latent_format = Wan21VAELatentFormat() 
            self.default_vae_type = VAEType.WAN21.value
        else:
            # 5B uses the new 48-channel latents
            self.latent_format = Wan22VAELatentFormat()
            self.default_vae_type = VAEType.WAN22.value

class WanAnimateModelArchConfig(Wan21ModelArchConfig):
    def __init__(self, model_type=Model.WAN22_14B_ANIMATE.value):
        super().__init__(model_type)
        self.model_type = model_type
        self.latent_format = Wan21VAELatentFormat()
        self.default_vae_type = VAEType.WAN21.value
        self.default_text_encoder = "umt5_xxl_fp16.safetensors"
        self.default_vision_encoder = "CLIP-ViT-H-fp16.safetensors"
        
    def get_ref_latent(self, image, vae, length, width, height, **kwargs):
        # processing reference image input
        # mask -- 1.0 = keep, 0.0 = generate
        if image is None: image = torch.zeros(1, 3, 1, height, width)     # ref ch can't be left out
        image = common_upscale(image[:length, :, :], width, height, "bilinear", "center")[0]
        image = image.unsqueeze(2).to(VRAM_DEVICE)      # [B, C, T, H, W], T=1
        ref_latent = vae.encode(image)
        ref_latent = self.latent_format.process_in(ref_latent)
        _, lat_ch, lat_t, lat_h, lat_w = ref_latent.shape  # [B, 16, t, h, w], t=1
        ref_mask = torch.ones(1, 4, lat_t, lat_h, lat_w, device=image.device, dtype=image.dtype)   # [B, 4, t, h, w]; lat_t = length // 4 
        # NOTE: mixing the ref_img with the pixel_video and jointly encoding can lead to the img
        # bleeding into the subsequent frames (although wan vae handles this pretty well)
        
        # main video tensor (initialized to gray 0.5)
        length -= image.shape[2]
        pixel_video = torch.ones((length, 3, height, width), device=VRAM_DEVICE, dtype=VAE_DTYPE) * 0.5
        pixel_mask = torch.zeros((length, 1, height, width), device=VRAM_DEVICE, dtype=VAE_DTYPE)  # [T, 1, H, W]
        
        # ----- temporal latents from the previous run
        # NOTE: normal sliding ctx is a not applicable here, as the model requires
        # pure temporal latents from the previous run
        temporal_latent = kwargs.get("temporal_latent", None)   # [T1, C, H, W]
        t_len = 0
        if temporal_latent is not None:
            t_len = min(length, temporal_latent.shape[0])
            pixel_video[:t_len] = temporal_latent[:t_len]
            pixel_mask[:t_len] = 1.0
        
        background_video_path, character_mask_path = kwargs.get("background_video_path", None), kwargs.get("character_mask_path", None)
        start_idx = t_len
        global_start_frame = kwargs.get("frame_offset", 0) + image.shape[2]  # image.shape[2] is 1 (ref frame)
        if background_video_path:
            bg_video, _ = MediaProcessor.load_video(background_video_path, output_frames=False) # [C, T, H, W]
            bg_video = common_upscale(bg_video.permute(1, 0, 2, 3), width, height, "area", "center")[0].permute(1, 0, 2, 3)
            
            if bg_video.shape[1] > global_start_frame:
                frames_to_copy = min(length - start_idx, bg_video.shape[1] - global_start_frame)
                if frames_to_copy > 0:
                    pixel_video[start_idx:start_idx + frames_to_copy] = \
                        bg_video[:, global_start_frame:global_start_frame + frames_to_copy].permute(1, 0, 2, 3).to(VRAM_DEVICE, dtype=VAE_DTYPE)

        if character_mask_path:
            mask_video, _ = MediaProcessor.load_video(character_mask_path, output_frames=False) # [C, T, H, W]
            mask_video = common_upscale(mask_video.permute(1, 0, 2, 3), width, height, "nearest", "center")[0].permute(1, 0, 2, 3)
            
            if mask_video.shape[1] > global_start_frame:
                frames_to_copy = min(length - start_idx, mask_video.shape[1] - global_start_frame)
                if frames_to_copy > 0:
                    m = mask_video[0:1, global_start_frame:global_start_frame + frames_to_copy].permute(1, 0, 2, 3)
                    # invert mask: input 1.0 (character) -> 0.0 (generate), input 0.0 (bg) -> 1.0 (keep)
                    pixel_mask[start_idx:start_idx + frames_to_copy] = 1.0 - m.to(VRAM_DEVICE, dtype=VAE_DTYPE)
        
        # encode latent
        pixel_video_b = pixel_video.permute(1, 0, 2, 3).unsqueeze(0) # [1, 3, T, H, W]
        latents = vae.encode(pixel_video_b) # [1, 16, t, h, w]
        latents = self.latent_format.process_in(latents)
        
        # reshape mask, [1, 1, T*4, H, W] -> [1, 4, T, H, W]
        mask_resized = common_upscale(pixel_mask, lat_w, lat_h, upscale_method="nearest-exact", crop="center")[0]   # [T, 1, h, w]
        target_frames = latents.shape[2] * 4
        current_frames = mask_resized.shape[0]
        if current_frames < target_frames:
            padding_needed = target_frames - current_frames
            last_frame = mask_resized[-1:].repeat(padding_needed, 1, 1, 1)
            mask_resized = torch.cat([mask_resized, last_frame], dim=0)
        mask_final = mask_resized.view(1, 1, latents.shape[2], 4, lat_h, lat_w)   # [1, 1, t, 4, h, w]
        mask_final = mask_final.permute(0, 1, 3, 2, 4, 5).squeeze(1)            # [1, 1, 4, t, h, w] -> [1, 4, t, h, w]
        
        full_mask = torch.cat([ref_mask, mask_final], dim=2)
        full_latent = torch.cat([ref_latent, latents], dim=2)
        concat = torch.cat([full_mask, full_latent], dim=1)
        return concat
                
def _process_vace_keyframes(cond: Conditioning, height: int, width: int, frame_count: int):
    found_idx = -1      # first keyframe aux conditioning index
    for i, aux_c in enumerate(cond.aux):
        if aux_c.type == AuxCondType.KEYFRAMES:
            found_idx = i
            break
    
    if found_idx == -1: return
    aux_c = cond.aux[found_idx]
    kf_data = fix_keyframe_indexing(aux_c.input_metadata, frame_count)
    
    # control_video with 0.5 (gray), mask with 1.0 (all masked)
    control_video = torch.ones((3, frame_count, height, width), dtype=torch.float32) * 0.5
    control_mask = torch.ones((frame_count, height, width), dtype=torch.float32)
    for idx, img_path in kf_data.items():
        if 0 <= idx < frame_count:
            img_tensor = get_image_tensor(img_path, height, width) # (1, 3, H, W)
            if img_tensor is not None:
                img_tensor = img_tensor.squeeze(0) # (3, H, W)
                control_video[:, idx] = img_tensor
                control_mask[idx] = 0.0 # 0.0 means unmasked (use this content)
    
    aux_c.input_metadata = {
        "control_video_tensor": control_video,
        "control_mask_tensor": control_mask
    }
    aux_c.type = AuxCondType.VACE_CTX
    # removing all OTHER keyframe or vace_ctx aux conds
    cond.aux = [a for i, a in enumerate(cond.aux) if i == found_idx or (a.type != AuxCondType.KEYFRAMES and a.type != AuxCondType.VACE_CTX)]

def _process_vace_context(cond_list: List[Conditioning], wrapper: ModelWrapper, vae: VAEBase, height: int, width: int, frame_count: int, progress_callback):
    _vace_cache = {}
    for c in cond_list:
        extra_frames = 0
        for aux_c in c.aux:
            if aux_c.type == AuxCondType.VACE_CTX:
                latent_length = ((frame_count - 1) // vae.temporal_compression_ratio) + 1
                reference_image, control_masks, control_video = None, None, None
                
                # check for direct tensor inputs (e.g. from _process_vace_keyframes)
                control_video = aux_c.input_metadata.get("control_video_tensor", None)
                control_masks = aux_c.input_metadata.get("control_mask_tensor", None)
                if control_video is None:
                    control_video_path = aux_c.input_metadata.get("control_video_path", None)
                    reference_image_path = aux_c.input_metadata.get("reference_image_path", None)   # TODO: extend to a list of ref images
                    if control_video_path: control_video, _ = MediaProcessor.load_video(control_video_path, output_frames=False)    # (C, T, H, W)
                    if reference_image_path: reference_image = get_image_tensor(reference_image_path, height, width)    # (B, C, W, H)
                
                key_video = get_tensor_weak_hash(control_video)
                key_mask = get_tensor_weak_hash(control_masks)
                key_ref = get_tensor_weak_hash(reference_image)
                cache_key = (key_video, key_mask, key_ref, height, width, frame_count)
                
                if cache_key in _vace_cache:
                    final = _vace_cache[cache_key]
                else:
                    if control_video is not None:
                        control_video = common_upscale(control_video[:, :frame_count], width, height, "bilinear", "center")[0]
                        control_video = control_video.permute(1, 2, 3, 0)   # (T, H, W, C)
                        if control_video.shape[0] < frame_count:
                            control_video = torch.nn.functional.pad(control_video, (0, 0, 0, 0, 0, 0, 0, frame_count - control_video.shape[0]), value=0.5)
                    else:
                        control_video = torch.ones((frame_count, height, width, 3)) * 0.5

                    if reference_image is not None:
                        reference_image = common_upscale(reference_image[:1], width, height, "bilinear", "center")[0]   # ([1, 3, 720, 720])
                        reference_image = vae.encode(reference_image[:, :3, :, :].unsqueeze(2))  # encoding (B, 3, 1, H, W) for (b, c, t, h, w)
                        reference_image = torch.cat([
                                reference_image, 
                                wrapper.model.model_arch_config.latent_format.process_out(torch.zeros_like(reference_image))    # dummy / no-mask background
                            ], dim=1)

                    if control_masks is None:
                        mask = torch.ones((frame_count, height, width, 1))
                    else:
                        mask = control_masks
                        if mask.ndim == 3: mask = mask.unsqueeze(1)     # (T, 1, H, W)
                        mask = common_upscale(mask[:frame_count], width, height, "bilinear", "center")[0]
                        mask = mask.permute(0, 2, 3, 1) # (T, H, W, 1)
                        if mask.shape[0] < frame_count:
                            mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, frame_count - mask.shape[0]), value=1.0)

                    control_video = control_video - 0.5
                    inactive = (control_video * (1 - mask)) + 0.5
                    reactive = (control_video * mask) + 0.5
                    # (C, T, H, W) -> (3, 0, 1, 2) on (T, H, W, C)
                    inactive = inactive.permute(3, 0, 1, 2)     # (B, C, T, H, W) vae format
                    reactive = reactive.permute(3, 0, 1, 2)
                    inactive = vae.encode(inactive[:3, :, :, :].unsqueeze(0))
                    reactive = vae.encode(reactive[:3, :, :, :].unsqueeze(0))
                    control_video_latent = torch.cat((inactive, reactive), dim=1)   # adding along the channel dim
                    if reference_image is not None:
                        control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)
                    
                    # pixel shuffle the mask (no vae encode)
                    vae_stride = vae.spatial_compression_ratio
                    height_mask = height // vae_stride
                    width_mask = width // vae_stride
                    mask = mask.view(frame_count, height_mask, vae_stride, width_mask, vae_stride)
                    mask = mask.permute(2, 4, 0, 1, 3)
                    mask = mask.reshape(vae_stride * vae_stride, frame_count, height_mask, width_mask)
                    mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

                    if reference_image is not None:
                        mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
                        mask = torch.cat((mask_pad, mask), dim=1)
                        extra_frames = max(extra_frames, reference_image.shape[2])  # NOTE: the above mask only handles the video input

                    mask = mask.unsqueeze(0).to(VRAM_DEVICE)
                    for i in range(0, control_video_latent.shape[1], 16):
                        control_video_latent[:, i:i + 16] = wrapper.model.process_latent_in(control_video_latent[:, i:i + 16])
                    final = torch.cat([control_video_latent, mask], dim=1)
                    final = final.unsqueeze(0)      # to support multi vace ctx inputs
                    final = (final, aux_c.input_metadata.get("strength", 1.0))      # ctx, strength tuple
                    _vace_cache[cache_key] = final
                    
                aux_c.data = final
        c.extra[ExtraCond.EXTRA_LATENT_FRAMES] = extra_frames       
    
def _process_visual_embeddings(cond_list, model_wrapper, height, width, progress_callback, clip_model_name=None):
    pending_embeds = []
    for c in cond_list:
        for aux_c in c.aux:
            if aux_c.type == AuxCondType.VISUAL_EMBEDDING and aux_c.data is None:
                pending_embeds.append(aux_c)
    
    if not pending_embeds: return
    progress_callback(0.3, "Generating CLIP embeddings")
    images = [get_image_tensor(aux.input_metadata, height, width) for aux in pending_embeds]
    clip_embed_list: List[VisionEncoderOutput] = get_vision_embeddings(
        images, 
        ve_model_filename=clip_model_name or model_wrapper.model.model_arch_config.default_vision_encoder
    )
    for aux_c, embed in zip(pending_embeds, clip_embed_list):
        aux_c.data = embed.intermediate_hidden_states
                
def _process_ref_latents(cond_list, model_wrapper, wan_vae, height, width, frame_count, progress_callback):
    progress_callback(0.4, "Generating referece latents")
    for c in cond_list:
        for aux_c in c.aux:
            if aux_c.type == AuxCondType.REF_LATENT and aux_c.data is None:
                img = get_image_tensor(img_path, height, width) if (img_path:=aux_c.input_metadata) is not None else None
                data = model_wrapper.model.model_arch_config.get_ref_latent(
                    image=img,
                    vae=wan_vae,
                    length=frame_count,
                    height=height,
                    width=width,
                    **c.extra
                )
                aux_c.data = data

def _process_wan_animate_aux(cond_list: List[Conditioning], model_wrapper, wan_vae, height, width, frame_count, progress_callback):
    for c in cond_list:
        current_frame_offset = c.extra.get("frame_offset", 0)       # 0, 10, 20 ...
        temporal_latent = c.extra.get("temporal_latent", None)
        max_overlap = c.extra.get("max_overlap", 10)
        if temporal_latent is not None:
            temporal_latent = temporal_latent[-max_overlap:]
            current_frame_offset -= temporal_latent.shape[0]
            current_frame_offset = max(0, current_frame_offset)
        
        for aux_c in c.aux:
            if aux_c.data is not None: continue
            if aux_c.type == AuxCondType.POSE_LATENTS:
                video_path = aux_c.input_metadata
                pose_video, _ = MediaProcessor.load_video(video_path, output_frames=False)
                pose_latents = None
                
                # pose_video is [C, T, H, W]
                if pose_video.shape[1] > current_frame_offset:
                    pose_video = pose_video[:, current_frame_offset:]
                    
                    if pose_video.shape[1] < frame_count:
                         last_frame = pose_video[:, -1:, :, :]
                         repeats = frame_count - pose_video.shape[1]
                         padding = last_frame.repeat(1, repeats, 1, 1)
                         pose_video = torch.cat([pose_video, padding], dim=1)
                    
                    pose_video = pose_video[:, :frame_count]
                    pose_video = common_upscale(pose_video.permute(1, 0, 2, 3), width, height, "area", "center")[0].permute(1, 0, 2, 3)
                    pose_video_b = pose_video.unsqueeze(0).to(VRAM_DEVICE, dtype=torch.float16) # TODO: dynamic dtype
                    pose_latents = wan_vae.encode(pose_video_b)
                    if model_wrapper.model.model_arch_config.latent_format:
                        pose_latents = model_wrapper.model.model_arch_config.latent_format.process_in(pose_latents)
                aux_c.data = pose_latents

            elif aux_c.type == AuxCondType.FACE_PIXEL_VALUES:
                if aux_c.data is not None: continue
                video_path = aux_c.input_metadata
                face_video, _ = MediaProcessor.load_video(video_path, output_frames=False)
                
                if face_video.shape[1] > current_frame_offset:
                    face_video = face_video[:, current_frame_offset:]
                    
                    if face_video.shape[1] < frame_count:
                         last_frame = face_video[:, -1:, :, :]
                         repeats = frame_count - face_video.shape[1]
                         padding = last_frame.repeat(1, repeats, 1, 1)
                         face_video = torch.cat([face_video, padding], dim=1)
                         
                    face_video = face_video[:, :frame_count]
                    face_video = common_upscale(face_video.permute(1, 0, 2, 3), 512, 512, "area", "center")[0].permute(1, 0, 2, 3)
                    face_video = face_video.unsqueeze(0).to(VRAM_DEVICE, dtype=torch.float16)
                    scale = 2.0 if c.group_name == "positive" else 0.0      # NOTE: hardcoding group name, 0 for negative
                    face_video = face_video * scale - 1.0 # Normalize to [-1, 1]
                else:
                    face_video = None
                    
                aux_c.data = face_video

def process_auxiliaries(cond_list: List[Conditioning], wrapper: ModelWrapper, wan_vae, height, width, frame_count, progress_callback, clip_model_name=None):
    if wrapper.model.model_type in [Model.WAN21_VACE_14B_R2V.value, Model.WAN21_VACE_1_3B_R2V.value]:
        for cond in cond_list: _process_vace_keyframes(cond, height, width, frame_count)
        _process_vace_context(cond_list, wrapper, wan_vae, height, width, frame_count, progress_callback)
    elif wrapper.model.model_type == Model.WAN22_14B_ANIMATE.value:
        _process_visual_embeddings(cond_list, wrapper, height, width, progress_callback, clip_model_name)
        _process_ref_latents(cond_list, wrapper, wan_vae, height, width, frame_count, progress_callback)
        _process_wan_animate_aux(cond_list, wrapper, wan_vae, height, width, frame_count, progress_callback)
    else:
        _process_visual_embeddings(cond_list, wrapper, height, width, progress_callback, clip_model_name)
        _process_ref_latents(cond_list, wrapper, wan_vae, height, width, frame_count, progress_callback)

@register_preprocessor(Model.WAN21_VACE_14B_R2V.value)
@register_preprocessor(Model.WAN21_VACE_1_3B_R2V.value)
@register_preprocessor(Model.WAN22_14B_ANIMATE.value)
@register_preprocessor(Model.WAN22_14B_TI2V.value)
@register_preprocessor(Model.WAN22_5B_T2V.value)
@register_preprocessor(Model.WAN21_14B_TI2V.value)
@register_preprocessor(Model.WAN21_1_3B_T2V.value)
def preprocess_wan_conditionals(
        model_wrapper: ModelWrapper,
        cond_list: List[Conditioning],           # NOTE: these conditionings most probably don't have 'data' at this point
        height: int, 
        width: int, 
        frame_count: int,
        vae: Optional[VAEBase] = None,           # TODO: not a good practice, passing both vae an vae_name
        progress_callback: Callable = null_func,
        t5_model_name: Optional[str] = None,
        clip_model_name: Optional[str] = None,
        vae_model_name: Optional[str] = None,
        **kwargs
) -> BatchedConditioning:
    
    progress_callback(0.1, "Initializing")
    
    progress_callback(0.2, "Encoding prompts")
    # generate text embeddings (Heavy VRAM operation)
    prompts = [c.input_metadata for c in cond_list]
    te_embeds: List[TextEncoderOutput] = get_text_embeddings(
        prompts, te_model_filename=t5_model_name or model_wrapper.model.model_arch_config.default_text_encoder)
    for i, te_embed in enumerate(te_embeds): cond_list[i].data = te_embed.last_hidden_state

    # NOTE: we generally pass vae_model_name instead of vae object because it allows the internal
    # logic of preprocess conds to handle loading / offloading on its own. although not used now, but maybe
    # used for bigger VAE models in the future
    if vae is None:
        wan_vae = get_vae(
            vae_type=model_wrapper.model.model_arch_config.default_vae_type,
            vae_dtype=torch.float16,
            use_tiling=False,
            override_filename=vae_model_name
        )
    else:
        wan_vae = vae
        
    frame_count = fix_frame_count(frame_count, wan_vae.temporal_compression_ratio)

    process_auxiliaries(cond_list, model_wrapper, wan_vae, height, width, frame_count, progress_callback, clip_model_name)
    
    # clean up internal VAE instance if we created it
    if vae is None:
        del wan_vae
        MemoryManager.clear_memory()
        
    batched_cond = BatchedConditioning(
        execution_order=["positive", "negative"]    # TODO: generalize this order based on index rather than group_name
    )
    for cond in cond_list: batched_cond.set_cond(cond)
    return batched_cond
