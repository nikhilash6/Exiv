import os
from torch import nn

from kirin.utils.model_utils import ModelMixin

script_dir = os.path.dirname(os.path.abspath(__file__))

class SimpleModel(ModelMixin):
    # TODO: create individual model type
    SIMPLE_MODEL_PTH_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model_checkpoint.pth"))
    SIMPLE_MODEL_SAFETENSORS_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model_checkpoint.pth"))
    SIMPLE_MODEL_CKPT_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model_checkpoint.pth"))
    SIMPLE_MODEL_PT_PATH = os.path.abspath(os.path.join(script_dir, "./assets/models/simple_model_checkpoint.pth"))

    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(1024, 2048)
        self.output_layer = nn.Linear(2048, 512)

    def forward(self, x):
        return self.output_layer(self.input_layer(x))