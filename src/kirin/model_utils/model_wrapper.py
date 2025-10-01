from ..utils.enum import ExtendedEnum


class ModelOption(ExtendedEnum):
    POST_CFG_FUNC = "sampler_post_cfg_function"

class ModelWrapper:
    def __init__(self, *args, **kwargs):
        pass
    
    # TODO: probably needs to be broken into smaller funcs
    def update_options(self, key, value):
        assert key in ModelOption.value_list(), "invalid model option"
        
        self.model_options[key] = self.model_options.get(key, []) + [value]
        return self.model_options