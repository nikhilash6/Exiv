from .hook_registry import FeatureType, register_hook_method
from ..model_utils.lora_mixin import LoraDefinition


@register_hook_method(FeatureType.LORA.value)
def enable_lora_hook(model: 'ModelMixin', lora_def: LoraDefinition = None, **kwargs):
    """
    NOTE: This is not exactly a hook, as all the lora state and functionality
    logic is inside the LoraMixin class. This simply acts like an entry point
    to some of its methods and a translation layer between user input and lora 
    application.
    """
    if not lora_def: lora_def = LoraDefinition.from_json(kwargs)
    model.add_lora(lora_def)