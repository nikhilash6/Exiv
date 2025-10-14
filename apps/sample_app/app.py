from exiv.components.enum import KSamplerType, ModelType, SchedulerType
from exiv.components.models.wan.main import WanModel
from exiv.components.samplers.model_sampling import KSampler
from exiv.components.samplers.sampler_types import get_model_sampling
from exiv.components.text_image_encoder.text_encoder import WanEncoder
from exiv.components.vae.wan_vae import WanVAE
from exiv.model_utils.model_wrapper import ModelWrapper


def main():
    positive_prompt = "a dog running in the park"
    negative_prompt = "blurry, bad quality"
    wan_encoder = WanEncoder()
    pos_embed = wan_encoder.encode(positive_prompt)
    neg_embed = wan_encoder.encode(negative_prompt)
    
    wan_vae = WanVAE()
    wan_dit_model = WanModel()
    model_sampling = get_model_sampling(ModelType.EDM)
    model_wrapper = ModelWrapper(
        model=wan_dit_model,
        model_sampling=model_sampling
    )
    
    main_sampler = KSampler(
        wrapped_model=model_wrapper,
        seed=123,
        steps=50,
        cfg=7.0,
        sampler_name=KSamplerType.EULER.value,
        scheduler_name=SchedulerType.SIMPLE.value,
        positive=pos_embed,
        negative=neg_embed,
    )
    
    out = main_sampler.run_sampling()
    out = wan_vae.decode(out)
    
    