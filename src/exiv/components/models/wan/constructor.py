from .main import Wan21Model
from ...enum import Model
from ....config import LOADING_MODE
from ....model_utils.helper_methods import get_state_dict
from ....utils.file import ensure_model_availability


def detect_wan_params(state_dict):
    """Detecting WAN config dynamically"""
    config = {}
    
    # 1. Dimensions
    config["dim"] = state_dict["head.modulation"].shape[-1]
    config["num_heads"] = config["dim"] // 128
    config["out_dim"] = state_dict["head.head.weight"].shape[0] // 4 # Patch size 2x2=4
    config["ffn_dim"] = state_dict["blocks.0.ffn.0.weight"].shape[0]
    
    dtype = state_dict["head.modulation"].dtype
    
    # 2. Depth (Scanning keys for the highest block index)
    max_block = 0
    for k in state_dict:
        if k.startswith("blocks."):
            # Parse "blocks.39.ffn..."
            try:
                # blocks.{id}.
                part = k.split(".")[1] 
                max_block = max(max_block, int(part))
            except: pass
    config["num_layers"] = max_block + 1
    
    # 3. Type (T2V vs I2V)
    if "img_emb.proj.0.bias" in state_dict:
        config["model_type"] = Model.WANTI2V.value
    else:
        config["model_type"] = Model.WANT2V.value
        
    return config, dtype

# NOTE: these methods detect the model arch (dims, layer counts etc.) from the 
# state dict and initialize the model cls appropriately

def get_wan_21_instance(
    model_path,
    download_url,
    force_load_mode=LOADING_MODE.LOW_VRAM.value,
    force_dtype=None
):
    model_path = ensure_model_availability(model_path, download_url)
    state_dict = get_state_dict(model_path)
    config, dict_dtype = detect_wan_params(state_dict)
    del state_dict
    
    dtype = force_dtype or dict_dtype
    wan_dit_model = Wan21Model(**config, force_load_mode=force_load_mode, dtype=dtype)
    wan_dit_model.load_model(model_path=model_path, download_url=download_url)
    return wan_dit_model