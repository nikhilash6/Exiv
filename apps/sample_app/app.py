import torch

from kirin.quantizers import TorchAOConfig, TorchAOQuantizer
from kirin.components import latent_noise_generator

# TODO: WIP
# def main():
#     text_encoder = WanTextEncoder()
#     vae_decoder = VAEDecoder()
    
#     quant_config = TorchAOConfig(quant_type=quant_type)
#     quantizer = TorchAOQuantizer(quantization_config=quant_config)
    
#     wan_model = Wan2_2(quantizer=quantizer, is_causal=True)
#     noise_generator = LatentNoiseGenerator()
#     noise = noise_generator(1, 21, 16, 60, 104)
    
#     num_blocks = 7
#     total_frames = 10 * num_blocks
#     generation_active = True
    
#     for idx, current_num_frames in enumerate(total_frames):
#         if not generation_active: break
        
#         s = current_start_frame - num_input_frames
#         e = current_start_frame + current_num_frames - num_input_frames
#         noisy_input = noise[:, s:e]
    
#         # denoising loop
#         for index, current_timestep in enumerate(denoising_step_list):
#             if not generation_active: break
            
#             timestep = torch.ones([1, current_num_frames], device=noise.device,
#                                       dtype=torch.int64) * current_timestep
            
#             if index < len(pipeline.denoising_step_list) - 1:
#                 _, denoised_pred = wan_model(
#                     noisy_image_or_video=noisy_input,
#                     conditional_dict=conditional_dict,
#                     timestep=timestep,
#                     kv_cache=pipeline.kv_cache1,
#                     crossattn_cache=pipeline.crossattn_cache,
#                     current_start=current_start_frame * pipeline.frame_seq_length
#                 )
#                 next_timestep = pipeline.denoising_step_list[index + 1]
#                 noisy_input = pipeline.scheduler.add_noise(
#                     denoised_pred.flatten(0, 1),
#                     torch.randn_like(denoised_pred.flatten(0, 1)),
#                     next_timestep * torch.ones([1 * current_num_frames], device=noise.device, dtype=torch.long)
#                 ).unflatten(0, denoised_pred.shape[:2])
#             else:
#                 _, denoised_pred = wan_model(
#                     noisy_image_or_video=noisy_input,
#                     conditional_dict=conditional_dict,
#                     timestep=timestep,
#                     kv_cache=pipeline.kv_cache1,
#                     crossattn_cache=pipeline.crossattn_cache,
#                     current_start=current_start_frame * pipeline.frame_seq_length
#                 )


def main(*args, **kwargs):
    print("running script ----- \n")
    noise = latent_noise_generator(1, 3, 32, 32)
    print("shape: ", noise.shape)