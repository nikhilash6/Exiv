import torch

from typing import List

from ..components.enum import TextEncoderType
from ..components.text_vision_encoder.common import VisionEncoderOutput, TextEncoderOutput
from ..components.text_vision_encoder.te_t5 import UMT5XXL
from ..components.text_vision_encoder.text_encoder import TextPipeline, create_text_pipeline
from ..utils.device import MemoryManager

# methods for getting standard outputs on the fly

def get_text_embeddings(
    input_data: List[TextEncoderOutput] | List[str] | str,    # if encoder output is not provided, it is calculated based on the str
    te_model_filename = None,               # can be overriden to a custom model
    te_model_type = None
) -> List[TextEncoderOutput]:
    
    assert isinstance(input_data, (TextEncoderOutput, list, str)), f"{type(input_data)} not supported" 
    if isinstance(input_data, TextEncoderOutput): return input_data     # return as-is
    if isinstance(input_data, str): input_data = [input_data]

    # load the model and generate the embedding
    te_pipeline: TextPipeline = create_text_pipeline(te_model_filename, te_model_type, dtype=torch.float16)
    te_pipeline.load_model()
    res = []
    for txt in input_data:
        embed: TextEncoderOutput = te_pipeline.encode(txt)
        res.append(embed)
    
    del te_pipeline
    MemoryManager.clear_memory()
    return res