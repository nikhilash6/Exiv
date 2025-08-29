from ..base import Pipeline
from ...components.text_encoders.text_encoders import Flux_Schnell_T5, SD_ClipL
from ...components.vae.noise_generator import LatentNoiseGenerator
from ...client.input_bindings import InputField, OutputField, FieldType, ActionButton

class FluxPipeline(Pipeline):
    def __init__(self):
        # setup UI / Input / Output
        height = InputField("height", FieldType.INT, max_val=1024, min_val=512)
        width = InputField("width", FieldType.INT, max_val=1024, min_val=512)
        output_img = OutputField("output_img", FieldType.IMAGE, frame_height=512, frame_width=512)
        generate_btn = ActionButton("Generate", callback=self.generate)
        
        # components
        initial_noise = LatentNoiseGenerator()
        t5 = Flux_Schnell_T5()
        clip = SD_ClipL()
        model = load_flow_model(name, device="cpu" if offload else torch_device)
        vae = load_ae(name, device="cpu" if offload else torch_device)
        
    def generate(self):
        pass