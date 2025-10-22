from typing import Any, Dict
import torch
from einops import rearrange, repeat

from ....config import DEFAULT_T2I_PROMPT
from ....utils.k_math import nearest_multiple
from .configs import configs
from ...text_encoders import T5XXL, SD_ClipL
from ...base import ModelMixin


class FLUX(ModelMixin):
    def __init__(
        self,
        name: str = "flux-schnell",
        width: int = 1360,
        height: int = 768,
        seed: int | None = None,
        prompt: str = DEFAULT_T2I_PROMPT,
        num_steps: int | None = None,
        guidance: float = 2.5,
        add_sampling_metadata: bool = True,
        # other components
        t5_encoder: T5XXL = None,
        clip_l_encoder: SD_ClipL = None,
        initial_noise = None,
    ):
        self.initial_noise = initial_noise
        self.prompt = self.prompt.split("|")
        if len(self.prompt) == 1:
            self.prompt = self.prompt[0]
            additional_prompts = None
        else:
            additional_prompts = self.prompt[1:]
            self.prompt = self.prompt[0]
            
        if name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Got unknown model name: {name}, choose from {available}")
        
        num_steps = num_steps or (4 if name == "flux_schnell" else 50)
        
        # for better conversion to the latent space
        height = nearest_multiple(height, 16)
        width = nearest_multiple(width, 16)
        
        self.t5_xxl = t5_encoder
        self.clip_l_encoder = clip_l_encoder
        
    def prepare_latents(self, img) -> Dict[str, Any]:
        bs, c, h, w = img.shape
        if isinstance(self.prompt, str):
            self.prompt = [self.prompt]
        
        # NOTE: if a list of prompts is passed then batch size is reset
        bs = len(self.prompt)

        # chunking img into 2x2 patches
        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if img.shape[0] == 1 and bs > 1:
            img = repeat(img, "1 ... -> bs ...", bs=bs)

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]   # fills the second channel with y coordinate
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]   # fills the third channel with x coordinate
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)   # reshape to og size

        # generating text embeddings via T5
        txt = self.t5_xxl(self.prompt)
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = torch.zeros(bs, txt.shape[1], 3)

        # image embeds via clip
        vec = self.clip_l_encoder(self.prompt)
        if vec.shape[0] == 1 and bs > 1:
            vec = repeat(vec, "1 ... -> bs ...", bs=bs)

        return {
            "img": img,
            "img_ids": img_ids.to(img.device),
            "txt": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
            "vec": vec.to(img.device),
        }
        
    def load_model(name: str, device: str | torch.device = torch_device, verbose: bool = True) -> Flux:
        # loading Flux
        print("Init model")
        config = configs[name]
        
        ckpt_path = str(get_checkpoint_path(config.repo_id, config.repo_flow, "FLUX_MODEL"))
        
        with torch.device("meta"):
            if config.lora_repo_id is not None and config.lora_filename is not None:
                model = FluxLoraWrapper(params=config.params).to(torch.bfloat16)
            else:
                model = Flux(config.params).to(torch.bfloat16)
        
        print(f"Loading checkpoint: {ckpt_path}")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device=str(device))
        sd = optionally_expand_state_dict(model, sd)
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if verbose:
            print_load_warning(missing, unexpected)
        
        if config.lora_repo_id is not None and config.lora_filename is not None:
            print("loading LoRA")
            lora_path = str(get_checkpoint_path(config.lora_repo_id, config.lora_filename, "FLUX_LORA"))
            lora_sd = load_sft(lora_path, device=str(device))
            # loading the lora params + overwriting scale values in the norms
            missing, unexpected = model.load_state_dict(lora_sd, strict=False, assign=True)
            if verbose:
                print_load_warning(missing, unexpected)
        return model
    

    def forward(self):
        # TODO: TEs are supposed to be offloaded here, create an automatic offload mechanism
        input_latent = self.prepare_latents(self.initial_noise)
        model = self.load_model(self.name, )
        
        pred = self.model(
            img=img_input,
            img_ids=img_input_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        