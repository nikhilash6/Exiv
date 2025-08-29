from ..base import Pipeline
from ...components.text_encoders.text_encoders import Flux_Schnell_T5, SDClipL

class FluxPipeline(Pipeline):
    def __init__(self):
        t5 = Flux_Schnell_T5()
        clip = SDClipL()
        model = load_flow_model(name, device="cpu" if offload else torch_device)
        ae = load_ae(name, device="cpu" if offload else torch_device)