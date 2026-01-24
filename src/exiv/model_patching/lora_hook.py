from .hook_registry import FeatureType, register_hook_method
from ..model_utils.lora_mixin import LoraDefinition


@register_hook_method(FeatureType.LORA.value)
def enable_lora_hook(model: 'ModelMixin', lora_def: LoraDefinition, total_steps: int = 1):
    """
    NOTE: This is not exactly a hook, as all the lora state and functionality
    logic is inside the LoraMixin class. This simply acts like an entry point
    to some of its methods and a translation layer between user input and lora 
    application.
    """
    model.add_lora(lora_def)
    model.setup_lora_schedule(total_steps)