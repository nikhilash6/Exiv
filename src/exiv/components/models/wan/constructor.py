from .main import Wan21Model, Wan21ModelArchConfig, Wan22Model, Wan22ModelArchConfig, Wan21VaceModel
from .animate import WanAnimateModel
from .cond_preprocessor import WanAnimateModelArchConfig
from ...enum import Model
from ....config import LOADING_MODE
from ....model_utils.helper_methods import get_state_dict
from ....utils.file import ensure_model_availability
from ....utils.logging import app_logger
from ....quantizers.base import QuantType

# TODO: streamline this ad-hoc logic
def detect_wan_params(state_dict):
    """Detecting WAN config dynamically"""
    config = {}
    
    # 1. Dimensions
    config["dim"] = state_dict["head.modulation"].shape[-1]
    config["num_heads"] = config["dim"] // 128
    config["out_dim"] = state_dict["head.head.weight"].shape[0] // 4 # Patch size 2x2=4
    config["ffn_dim"] = state_dict["blocks.0.ffn.0.weight"].shape[0]
    
    dtype = state_dict["blocks.0.ffn.0.weight"].dtype
    cls = None
    # 2. Depth (Scanning keys for the highest block index)
    max_blocks = 0
    vace_blocks = 0
    for k in state_dict:
        parts = k.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            val = int(parts[1])
            if k.startswith("blocks."):
                max_blocks = max(max_blocks, val)
            elif k.startswith("vace_blocks."):
                vace_blocks = max(vace_blocks, val)
            
    config["num_layers"] = max_blocks + 1
    if vace_blocks > 0: 
        cls = Wan21VaceModel
        config["vace_layers"] = vace_blocks + 1
        try:
            config["vace_dim"] = state_dict["vace_patch_embedding.weight"].shape[1]
        except: pass
    elif "pose_patch_embedding.weight" in state_dict:
        cls = WanAnimateModel
    
    # 3. Model Version Detection
    if cls is None:
        # 16 = Wan 2.1, 48 = Wan 2.2
        if config["out_dim"] == 48:
            cls = Wan22Model
        else:
            if config["dim"] == 1536 or any('img_emb.proj' in s for s in list(state_dict.keys())):  # wan21 1.3b + wan21 14b
                cls = Wan21Model
            else:
                cls = Wan22Model 

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
    elif cls == Wan21Model:
        if config["dim"] == 5120:
            config["model_type"] = Model.WAN21_14B_TI2V.value
            model_arch_config = Wan21ModelArchConfig(model_type=Model.WAN21_14B_TI2V.value)
        else:
            config["model_type"] = Model.WAN21_1_3B_T2V.value
            model_arch_config = Wan21ModelArchConfig(model_type=Model.WAN21_1_3B_T2V.value)
    elif cls == Wan21VaceModel:
        if config["dim"] == 1536:
            config["model_type"] = Model.WAN21_VACE_1_3B_R2V.value
            model_arch_config = Wan21ModelArchConfig(model_type=Model.WAN21_VACE_1_3B_R2V.value)
        elif config["dim"] == 5120:
            config["model_type"] = Model.WAN21_VACE_14B_R2V.value
            model_arch_config = Wan21ModelArchConfig(model_type=Model.WAN21_VACE_14B_R2V.value)
    elif cls == WanAnimateModel:
        config["model_type"] = Model.WAN22_14B_ANIMATE.value
        model_arch_config = WanAnimateModelArchConfig(model_type=Model.WAN22_14B_ANIMATE.value)
        if "face_encoder.conv1_local.weight" in state_dict:
             config["motion_encoder_dim"] = state_dict["face_encoder.conv1_local.weight"].shape[1]
    
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
    quant_type=None, #QuantType.FP8_SCALED,
):
    model_path = ensure_model_availability(model_path, download_url)
    state_dict = get_state_dict(model_path, model_type="checkpoint")    # TODO: convert these model_type strings into enums
    cls, config, dict_dtype, model_arch_config = detect_wan_params(state_dict)
    del state_dict

    dtype = force_dtype or dict_dtype
    wan_dit_model = cls(**config, force_load_mode=force_load_mode, dtype=dtype, quant_type=quant_type)
    wan_dit_model.model_arch_config = model_arch_config
    wan_dit_model.load_model(model_path=model_path, download_url=download_url, model_type="checkpoint")
    return wan_dit_model