import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image
from typing import Callable, Dict, List, Optional

from exiv.utils.file import MediaProcessor

from ...latent_format import Wan21VAELatentFormat, Wan22VAELatentFormat
from ....model_utils.model_mixin import ModelArchConfig
from ....utils.device import VRAM_DEVICE
from ....utils.tensor import common_upscale, get_tensor_weak_hash
from ...enum import Model, VAEType
from ...cond_registry import get_image_tensor, get_text_embeddings, get_vision_embeddings, register_preprocessor
from ...text_vision_encoder.common import TextEncoderOutput, VisionEncoderOutput
from ...vae.base import VAEBase, get_vae
from ....model_utils.common_classes import AuxCondType, AuxConditioning, BatchedConditioning, Conditioning, ExtraCond, Latent, ModelWrapper
from ....utils.common import fix_frame_count, null_func


is_text_model = lambda model_type: model_type in [Model.WAN21_1_3B_T2V.value, Model.WAN22_5B_T2V.value, Model.WAN22_14B_TI2V.value]
is_img_model = lambda model_type: not is_text_model(model_type)

class Wan21ModelArchConfig(ModelArchConfig):
    def __init__(self, model_type):
        self.model_type = model_type
        self.latent_format = Wan21VAELatentFormat()
        
        # default models
        self.default_vae_type = VAEType.WAN21.value
        self.default_text_encoder = "umt5_xxl_fp16.safetensors"
        self.default_vision_encoder = "CLIP-ViT-H-fp16.safetensors"
        
    def get_ref_latent(self, start_image, vae, length, width, height):
        if is_text_model(self.model_type) and self.model_type != Model.WAN22_14B_TI2V.value: return None
        start_image = common_upscale(start_image, width, height, "bilinear", "center")[0]
        video = torch.ones((1, 3, length, height, width), device=start_image.device, dtype=start_image.dtype) * 0.5
        video[:, :, 0, :, :] = start_image
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
            device=start_image.device,
            dtype=start_image.dtype
        )
        mask[:, :, :((start_image.shape[0] - 1) // vae.temporal_compression_ratio) + 1] = 1.0
        
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

def _process_vace_context(cond_list: List[Conditioning], wrapper: ModelWrapper, vae: VAEBase, height: int, width: int, frame_count: int, progress_callback):
    _vace_cache = {}
    for c in cond_list:
        extra_frames = 0
        for aux_c in c.aux:
            if aux_c.type == AuxCondType.VACE_CTX:
                latent_length = ((frame_count - 1) // vae.temporal_compression_ratio) + 1
                reference_image, control_masks, control_video = None, None, None
                control_video_path = aux_c.input_metadata.get("control_video_path", None)
                reference_image_path = aux_c.input_metadata.get("reference_image_path", None)
                if control_video_path: control_video = MediaProcessor.load_video(control_video_path, output_frames=True)
                if reference_image_path: reference_image = get_image_tensor(reference_image_path)
                
                key_video = get_tensor_weak_hash(control_video)
                key_mask = get_tensor_weak_hash(control_masks)
                key_ref = get_tensor_weak_hash(reference_image)
                cache_key = (key_video, key_mask, key_ref, height, width, frame_count)
                
                if cache_key in _vace_cache:
                    final = _vace_cache[cache_key]
                else:
                    if control_video is not None:
                        control_video = common_upscale(control_video[:frame_count].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                        if control_video.shape[0] < frame_count:
                            control_video = torch.nn.functional.pad(control_video, (0, 0, 0, 0, 0, 0, 0, frame_count - control_video.shape[0]), value=0.5)
                    else:
                        control_video = torch.ones((frame_count, height, width, 3)) * 0.5

                    if reference_image is not None:
                        reference_image = common_upscale(reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                        reference_image = vae.encode(reference_image[:, :, :, :3])
                        reference_image = torch.cat([
                                reference_image, 
                                wrapper.model.model_arch_config.latent_format.process_out(torch.zeros_like(reference_image))    # dummy / no-mask background
                            ], dim=1)

                    if control_masks is None:
                        mask = torch.ones((frame_count, height, width, 1))
                    else:
                        mask = control_masks
                        if mask.ndim == 3: mask = mask.unsqueeze(1)     # add channel dim
                        mask = common_upscale(mask[:frame_count], width, height, "bilinear", "center").movedim(1, -1)
                        if mask.shape[0] < frame_count:
                            mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, frame_count - mask.shape[0]), value=1.0)

                    control_video = control_video - 0.5
                    inactive = (control_video * (1 - mask)) + 0.5
                    reactive = (control_video * mask) + 0.5

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
                
                aux_c.data = final
        
        c.extra[ExtraCond.EXTRA_LATENT_FRAMES] = extra_frames       
    

def _process_visual_embeddings(cond_list, model_wrapper, height, width, progress_callback):
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
        ve_model_filename=model_wrapper.model.model_arch_config.default_vision_encoder
    )
    for aux_c, embed in zip(pending_embeds, clip_embed_list):
        aux_c.data = embed.intermediate_hidden_states
                
def _process_ref_latents(cond_list, model_wrapper, wan_vae, height, width, frame_count, progress_callback):
    progress_callback(0.4, "Generating referece latents")
    for c in cond_list:
        for aux_c in c.aux:
            if aux_c.type == AuxCondType.REF_LATENT and aux_c.data is None:
                if (img:=get_image_tensor(aux_c.input_metadata, height, width)) is not None:
                    data = model_wrapper.model.model_arch_config.get_ref_latent(
                        start_image=img,
                        vae=wan_vae,
                        length=frame_count,
                        height=height,
                        width=width,
                    )
                    aux_c.data = data

def process_auxiliaries(cond_list, wrapper, wan_vae, height, width, frame_count, progress_callback):
    _process_visual_embeddings(cond_list, wrapper, height, width, progress_callback)
    _process_ref_latents(cond_list, wrapper, wan_vae, height, width, frame_count, progress_callback)
    _process_vace_context(cond_list, wrapper, wan_vae, height, width, frame_count, progress_callback)

@register_preprocessor(Model.WAN21_VACE_1_3B_R2V.value)
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
        vae: Optional[VAEBase] = None,           # uses the default vae if not provided
        progress_callback: Callable = null_func
) -> BatchedConditioning:
    
    progress_callback(0.1, "Initializing")
    if vae is None:
        wan_vae = get_vae(
            vae_type=model_wrapper.model.model_arch_config.default_vae_type,
            vae_dtype=torch.float16,
            use_tiling=False
        )
    else:
        wan_vae = vae
    frame_count = fix_frame_count(frame_count, wan_vae.temporal_compression_ratio)
    
    progress_callback(0.2, "Encoding prompts")
    # generate text embeddings
    prompts = [c.input_metadata for c in cond_list]
    te_embeds: List[TextEncoderOutput] = get_text_embeddings(
        prompts, te_model_filename=model_wrapper.model.model_arch_config.default_text_encoder)
    for i, te_embed in enumerate(te_embeds): cond_list[i].data = te_embed.last_hidden_state
    
    process_auxiliaries(cond_list, model_wrapper, wan_vae, height, width, frame_count, progress_callback)
    batched_cond = BatchedConditioning(
        execution_order=["positive", "negative"]    # TODO: generalize this order based on index rather than group_name
    )
    for cond in cond_list: batched_cond.set_cond(cond)
    return batched_cond
