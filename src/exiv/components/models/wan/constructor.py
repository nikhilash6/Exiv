from .main import Wan21Model, Wan21ModelArchConfig, Wan22Model, Wan22ModelArchConfig
from ...enum import Model
from ....config import LOADING_MODE
from ....model_utils.helper_methods import get_state_dict
from ....utils.file import ensure_model_availability
from ....utils.logging import app_logger

def detect_wan_params(state_dict):
    """Detecting WAN config dynamically"""
    config = {}
    
    # 1. Dimensions
    config["dim"] = state_dict["head.modulation"].shape[-1]
    config["num_heads"] = config["dim"] // 128
    config["out_dim"] = state_dict["head.head.weight"].shape[0] // 4 # Patch size 2x2=4
    config["ffn_dim"] = state_dict["blocks.0.ffn.0.weight"].shape[0]
    
    dtype = state_dict["blocks.0.ffn.0.weight"].dtype
    
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
    
    # 3. Model Version Detection
    # 16 = Wan 2.1, 48 = Wan 2.2
    if config["out_dim"] == 48:
        cls = Wan22Model
    else:
        cls = Wan21Model

    # 4. Type (T2V vs I2V)
    # T2V usually has in_dim=16
    # I2V usually has in_dim=36 (16 latent + 4 mask + 16 image)
    input_channels = state_dict["patch_embedding.weight"].shape[1]
    model_arch_config = None
    if cls == Wan22Model:
        if config["dim"] == 3072:
            config["model_type"] = Model.WAN22_5B_T2V.value
            model_arch_config = Wan22ModelArchConfig(model_type=Model.WAN22_5B_T2V.value)
        elif config["dim"] == 5120:
            config["model_type"] = Model.WAN22_14B_TI2V.value
            model_arch_config = Wan22ModelArchConfig(model_type=Model.WAN22_14B_TI2V.value)
    else:
        if config["dim"] == 5120:  
            config["model_type"] = Model.WAN21_14B_TI2V.value
            model_arch_config = Wan21ModelArchConfig(model_type=Model.WAN21_14B_TI2V.value)
        else:
            config["model_type"] = Model.WAN21_1_3B_T2V.value
            model_arch_config = Wan21ModelArchConfig(model_type=Model.WAN21_1_3B_T2V.value)
    
    config["in_dim"] = input_channels
    if "ref_conv.weight" in state_dict:
        config["in_dim_ref_conv"] = state_dict["ref_conv.weight"].shape[1]
    else:
        config["in_dim_ref_conv"] = None
    
    assert model_arch_config is not None, "Model not supported or is in the wrong format. Aborting."
    app_logger.info(f"Model detected: {cls.__name__} {dtype} {model_arch_config.__class__.__name__} {config['model_type']}")
    return cls, config, dtype, model_arch_config

# NOTE: these methods detect the model arch (dims, layer counts etc.) from the 
# state dict and initialize the model cls appropriately

def get_wan_instance(
    model_path,
    download_url,
    force_load_mode=LOADING_MODE.LOW_VRAM.value,
    force_dtype=None,
    quant_type=None,
):
    model_path = ensure_model_availability(model_path, download_url)
    state_dict = get_state_dict(model_path)
    cls, config, dict_dtype, model_arch_config = detect_wan_params(state_dict)
    del state_dict
    
    dtype = force_dtype or dict_dtype
    wan_dit_model = cls(**config, force_load_mode=force_load_mode, dtype=dtype, quant_type=quant_type)
    wan_dit_model.model_arch_config = model_arch_config
    wan_dit_model.load_model(model_path=model_path, download_url=download_url)
    return wan_dit_model