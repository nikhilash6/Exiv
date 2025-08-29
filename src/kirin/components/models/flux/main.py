from ....constants import DEFAULT_T2I_PROMPT
from ...utils.k_math import nearest_multiple
from .configs import configs

def main(
    name: str = "flux-schnell",
    width: int = 1360,
    height: int = 768,
    seed: int | None = None,
    prompt: str = DEFAULT_T2I_PROMPT,
    num_steps: int | None = None,
    guidance: float = 2.5,
    offload: bool = False,
    output_file_path: str = "output",
    add_sampling_metadata: bool = True,
):
    prompt = prompt.split("|")
    if len(prompt) == 1:
        prompt = prompt[0]
        additional_prompts = None
    else:
        additional_prompts = prompt[1:]
        prompt = prompt[0]
        
    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, choose from {available}")
    
    num_steps = num_steps or (4 if name == "flux_schnell" else 50)
    
    # for better conversion to the latent space
    height = nearest_multiple(height, 16)
    width = nearest_multiple(width, 16)
    
    