import torch
from torch import nn
from torch import Tensor
from einops import rearrange

# taken from base flux impl.
def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """
    Calculates the Rotary Positional Embedding (RoPE) rotation matrices.

    Args:
        pos: A tensor of positions (e.g., token indices).
        dim: The dimensionality of the feature vector to be rotated.
        theta: A base value for the frequency calculation, typically a large integer like 10000.

    Returns:
        A tensor containing the 2x2 rotation matrices for each position and dimension pair.
    """
    # Ensure the dimension is even, as we are rotating pairs of features.
    assert dim % 2 == 0

    # Create a scale for the frequencies. This corresponds to the 'i' in the RoPE formula's
    # denominator, ensuring each pair of dimensions gets a different frequency.
    # Example: if dim=4, scale will be [0/4, 2/4] = [0.0, 0.5]
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim

    # Calculate the rotational frequencies (omega) for each pair of dimensions.
    # This creates a geometric progression of frequencies.
    # omega = 1 / (theta^(2i/d))
    omega = 1.0 / (theta**scale)

    # Calculate the angle for each position and frequency pair by multiplying them.
    # This is equivalent to 'm * θ_i' from the formula.
    # pos - [b, n] , omega - [d/2], out - [b, n, d/2] (batch, sequence_len, d/2)
    out = torch.einsum("...n,d->...nd", pos, omega)

    # Create the 2x2 rotation matrix elements: [cos(angle), -sin(angle), sin(angle), cos(angle)]
    # This prepares the values for the final matrix.
    # out - [batch_size, sequence_length, d/2, 4]
    # This final tensor holds the [cos, -sin, sin, cos] values for every position and every feature pair.
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)

    # Reshape the tensor to form explicit 2x2 matrices for each rotation.
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)

    # Return the final rotation matrices.
    # NOTE: fp16 also works here perfectly fine, but since this doesn't affect perf that much
    # and can be used for longer videos as well, we are keeping it in fp32
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """
    Applies the pre-computed RoPE rotation matrices to the input tensors (query and key).

    Args:
        x: The query tensor.
        freqs_cis: The pre-computed rotation matrices from the `rope` function.

    Returns:
        A tuple containing the rotated query and key tensors.
    """
    # Reshape the query tensor to group its dimensions into pairs.
    # If x has shape [..., d], it becomes [..., d/2, 2].
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)

    # Apply the rotation. This is a vectorized implementation of the 2x2 matrix multiplication:
    # x' = x * cos(θ) - y * sin(θ)
    # y' = x * sin(θ) + y * cos(θ)
    # Here, freqs_cis[..., 0] contains the [cos, sin] part and x_[..., 0] is the [x, y] part.
    # The multiplication and summation effectively perform the rotation.
    x_out = freqs_cis[..., 0] * x_[..., 0] + freqs_cis[..., 1] * x_[..., 1]

    # Reshape the rotated tensors back to their original d-dimensional shape.
    return x_out.reshape(*x.shape).type_as(x)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)