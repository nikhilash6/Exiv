from ..base import Pipeline

class FluxPipeline(Pipeline):
    def __init__(self):
        t5 = load_t5(torch_device, max_length=256 if name == "flux_schnell" else 512)
        clip = load_clip(torch_device)
        model = load_flow_model(name, device="cpu" if offload else torch_device)
        ae = load_ae(name, device="cpu" if offload else torch_device)