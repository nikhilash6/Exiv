from .encoder_base import VisionEncoder

# TODO: implement this
class DINOv2(VisionEncoder):
    """
    -   Identified by: Unique key names like 'encoder.layer.39.layer_scale2.lambda1'.
    -   Trained by Meta AI (Facebook AI Research).
    -   Self-supervised learning on *images only* (no text).
        Learns by masking parts of an image and predicting the hidden content.
    -   Good At: A pure, powerful vision feature extractor. It has *no text understanding*.
        It excels at tasks like depth estimation, segmentation, and dense feature matching.
    """
    pass