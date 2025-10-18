from torch import Tensor

# TODO: complete this when working with smaller dtypes
def cast_like_reference(t: Tensor, reference: Tensor):
    # casts the dtype and device of 't' to the same as the reference
    return t