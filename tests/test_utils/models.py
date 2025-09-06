import torch
from torch import nn

from kirin.utils.model_utils import ModelMixin

class SimpleModel(ModelMixin):
    # TODO: create individual model type
    SIMPLE_MODEL_PTH_PATH = "test_assets/models/simple_model_checkpoint.pth"
    SIMPLE_MODEL_SAFETENSORS_PATH = "test_assets/models/simple_model_checkpoint.pth"
    SIMPLE_MODEL_CKPT_PATH = "test_assets/models/simple_model_checkpoint.pth"
    SIMPLE_MODEL_PT_PATH = "test_assets/models/simple_model_checkpoint.pth"

    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(1024, 2048)
        self.output_layer = nn.Linear(2048, 512)

    def forward(self, x):
        return self.output_layer(self.input_layer(x))