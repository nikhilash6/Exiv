from .hook import TaylorSeerModelHook, TaylorSeerModuleHook
from ....components.enum import Model

def wan_module_filter(model):
    return model.blocks

def get_taylor_seer_data(model):
    module_list = []
    if model.model_type in [Model.WANT2V.value, Model.WANTI2V.value]:
        module_list = wan_module_filter(model)
    else:
        raise Exception(f"{model.model_type} is not supported by Taylor Seer caching atm")
    
    return module_list, [TaylorSeerModuleHook() for _ in range(len(module_list))], TaylorSeerModelHook()