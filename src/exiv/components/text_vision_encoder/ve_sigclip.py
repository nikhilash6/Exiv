from .encoder_base import VisionEncoder


# TODO: implement this
class SigLIP(VisionEncoder):
    """
    - Identified by: Unique hidden size (1152) and embedding shapes (e.g., 729, 1024).
    - Trained by Google Research.
    - Similar to CLIP (image-text pairs), but uses a simpler and more efficient Sigmoid (siglip) 
        loss function instead of a contrastive one.
    - Good At: Same tasks as CLIP (text-image similarity, zero-shot classification),
       often achieving better performance than similarly-sized CLIP models. A more modern evolution of CLIP.
    """
    pass