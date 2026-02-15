
import pytest
import torch
from exiv.utils.tensor import common_upscale

@pytest.mark.parametrize("upscale_method", ["nearest", "bilinear", "bicubic", "area", "lanczos", "bislerp"])
def test_common_upscale_methods(upscale_method):
    """Test that common_upscale works with all supported upscale methods."""
    input_tensor = torch.rand((1, 3, 64, 64))
    target_w, target_h = 128, 128
    
    output = common_upscale(input_tensor, target_w, target_h, upscale_method=upscale_method)[0]
    
    assert output.shape == (1, 3, 128, 128)
    assert not torch.isnan(output).any()

def test_common_upscale_crop_center():
    """Test standard center cropping behavior."""
    # Input: 100x50 (Wide)
    # Target: 50x50 (Square)
    # Should crop the center 50x50 from the input
    input_tensor = torch.arange(100 * 50).reshape(1, 1, 50, 100).float()
    target_w, target_h = 50, 50
    
    output = common_upscale(input_tensor, target_w, target_h, upscale_method="nearest", crop="center")[0]
    
    assert output.shape == (1, 1, 50, 50)
    
    # The center 50x50 of the 100x50 image should be preserved
    # Input coords x: 25 to 75, y: 0 to 50
    expected_center = input_tensor[..., :, 25:75]
    assert torch.allclose(output, expected_center)

def test_common_upscale_crop_pad_scalar_color():
    """Test padding behavior with a scalar color."""
    # Input: 50x50 (Square)
    # Target: 100x50 (Wide)
    # Should scale to 50x50 (to fit height) and pad width to 100
    input_tensor = torch.ones((1, 3, 50, 50))
    target_w, target_h = 100, 50
    pad_color = 0.5
    
    output = common_upscale(input_tensor, target_w, target_h, upscale_method="nearest", crop="pad", padding_color=pad_color)[0]
    
    assert output.shape == (1, 3, 50, 100)
    
    # Check padding (25 pixels on left and right)
    left_pad = output[..., :, :25]
    right_pad = output[..., :, 75:]
    center = output[..., :, 25:75]
    
    assert torch.allclose(left_pad, torch.tensor(pad_color))
    assert torch.allclose(right_pad, torch.tensor(pad_color))
    assert torch.allclose(center, torch.ones_like(center))

def test_common_upscale_crop_pad_tuple_color():
    """Test padding behavior with a per-channel tuple color."""
    # Input: 50x50
    # Target: 50x100 (Tall)
    # Should scale to 50x50 (to fit width) and pad height to 100
    input_tensor = torch.ones((1, 3, 50, 50))
    target_w, target_h = 50, 100
    pad_color = (0.1, 0.2, 0.3)
    
    output = common_upscale(input_tensor, target_w, target_h, upscale_method="nearest", crop="pad", padding_color=pad_color)[0]
    
    assert output.shape == (1, 3, 100, 50)
    
    # Check padding (25 pixels top and bottom)
    top_pad = output[..., :25, :]
    bottom_pad = output[..., 75:, :]
    center = output[..., 25:75, :]
    
    # Check channels for top pad
    assert torch.allclose(top_pad[:, 0, ...], torch.tensor(0.1))
    assert torch.allclose(top_pad[:, 1, ...], torch.tensor(0.2))
    assert torch.allclose(top_pad[:, 2, ...], torch.tensor(0.3))
    
    assert torch.allclose(center, torch.ones_like(center))

def test_common_upscale_5d_input():
    """Test that 5D inputs (e.g. video batches) are handled correctly."""
    # Input: B=1, F=2, C=3, H=10, W=10
    input_tensor = torch.zeros((1, 2, 3, 10, 10))
    target_w, target_h = 20, 20
    
    output = common_upscale(input_tensor, target_w, target_h, upscale_method="nearest", crop="pad", padding_color=1.0)[0]
    
    # Should maintain 5D structure
    assert output.shape == (1, 2, 3, 20, 20)
    
    # With nearest neighbor upscale of zeros, content should be zeros
    # But if we pad? Let's try a case with padding
    
    target_w_pad, target_h_pad = 40, 20 # Wide target
    output_pad = common_upscale(input_tensor, target_w_pad, target_h_pad, upscale_method="nearest", crop="pad", padding_color=1.0)[0]
    
    assert output_pad.shape == (1, 2, 3, 20, 40)
    
    # Check padding (should be 1.0)
    left_pad = output_pad[..., :, :10]
    assert torch.allclose(left_pad, torch.tensor(1.0))
    
    # Check content (should be 0.0)
    center = output_pad[..., :, 10:30]
    assert torch.allclose(center, torch.tensor(0.0))

def test_common_upscale_list_input():
    """Test that a list of tensors is handled correctly."""
    t1 = torch.rand((1, 3, 64, 64))
    t2 = torch.rand((1, 3, 64, 64))
    samples = [t1, t2]
    target_w, target_h = 128, 128
    
    outputs = common_upscale(samples, target_w, target_h)
    
    assert isinstance(outputs, list)
    assert len(outputs) == 2
    assert outputs[0].shape == (1, 3, 128, 128)
    assert outputs[1].shape == (1, 3, 128, 128)

def test_common_upscale_no_crop_resizes_directly():
    """Test that crop != 'center' and != 'pad' (default/fallback) just stretches the image."""
    # Input: 10x10
    # Target: 20x40
    input_tensor = torch.ones((1, 1, 10, 10))
    target_w, target_h = 20, 40
    
    # Using 'stretch' or any unknown string acts as direct resize/stretch
    output = common_upscale(input_tensor, target_w, target_h, upscale_method="nearest", crop="stretch")[0]
    
    assert output.shape == (1, 1, 40, 20)
    assert torch.allclose(output, torch.ones_like(output))
