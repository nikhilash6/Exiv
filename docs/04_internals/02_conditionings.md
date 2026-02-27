# Conditionings

Conditioning in Exiv dictates how text, images, or auxiliary inputs guide the generation process. In simpler terms, everything that goes inside a model to steer its output can be considered a conditioning.

## Conditioning Mixin
The `conditioning_mixin.py` manages how various constraints (like ControlNet, Text Encoder, or Vision Encoder outputs) are combined into a standardized latent format.
We have a **main conditioning** (like the primary text prompt or visual embedding), to which we attach **auxiliary conditionings** to provide additional guidance.

### Conditioning Class
The `Conditioning` class is the common structure that supports all conditioning inputs:

```python
@dataclass
class Conditioning:
    group_name: str = "positive"
    data: Optional[Tensor] = None
    input_metadata: Optional[Union[dict, str]] = None
    type: ConditioningType = ConditioningType.EMBEDDING
    # --- Timings & Ranges ---
    timestep_range: Tuple[float, float] = (0, -1)          # (start, end), -1 means it spans the complete range
    frame_range: Optional[Tuple[int, int]] = (0, -1)       # (start_idx, end_idx)
    mask: Tensor | None = None
    strength: Union[float, Tensor, List[float]] = 1.0
    aux: Optional[List[AuxConditioning]] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    model_input: Optional[ModelForwardInput] = None
```

### Auxiliary Conditionings
Auxiliary conditionings provide extra or support signals for the generation process. The types of auxiliary conditionings include:
- `TIME_HINT`: Time-related cues (duration, camera pose, etc.)
- `REF_LATENT`: Reference latents for style transfer, audio timbre, etc.
- `VISUAL_EMBEDDING`: Image Prompt Adapters (IP-Adapter) or vision embeddings.
- `CONTROLNET`: ControlNet signals.
- `VACE_CTX`: Wan VACE context.
- `KEYFRAMES`: Keyframe inputs, where the respective mask is generated through model-specific logic.
- `POSE_LATENTS`: Pose latents (e.g., Wan animate pose).
- `FACE_PIXEL_VALUES`: Face pixel values (e.g., Wan animate face).

Conditionings are passed from an `App` directly into the `Samplers` to steer generation toward the desired output.

> **Note:** The structure and types of auxiliary conditionings are subject to change. In the future, they might be consolidated into dict-like attributes for easier management and extensibility.