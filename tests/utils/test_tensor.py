import unittest
import torch
from exiv.utils.tensor import common_upscale

class TestTensorUtilities(unittest.TestCase):
    def test_common_upscale_methods(self):
        """Test that common_upscale works with all supported upscale methods."""
        input_tensor = torch.rand((1, 3, 64, 64))
        target_w, target_h = 128, 128
        
        for upscale_method in ["nearest", "bilinear", "bicubic", "area", "lanczos", "bislerp"]:
            with self.subTest(upscale_method=upscale_method):
                output = common_upscale(input_tensor, target_w, target_h, upscale_method=upscale_method)[0]
                self.assertEqual(output.shape, (1, 3, target_h, target_w))
                self.assertFalse(torch.isnan(output).any())

    def test_common_upscale_crop_center(self):
        """Test standard center cropping behavior."""
        # Input: 100x50 (Wide) -> shape [1, 1, 50, 100] (H=50, W=100)
        # Target: 50x50 (Square)
        # Should scale height to 50 (scale factor 1) and crop width to 50 from the center
        input_tensor = torch.arange(100 * 50).reshape(1, 1, 50, 100).float()
        target_w, target_h = 50, 50
        
        output = common_upscale(input_tensor, target_w, target_h, upscale_method="nearest", crop="center")[0]
        
        self.assertEqual(output.shape, (1, 1, target_h, target_w))
        
        # The center 50x50 of the 100x50 image should be preserved
        # Input coords x: 25 to 75, y: 0 to 50
        expected_center = input_tensor[..., :, 25:75]
        self.assertTrue(torch.allclose(output, expected_center))

    def test_common_upscale_crop_pad_scalar_color(self):
        """Test padding behavior with a scalar color."""
        # Input: 50x50 (Square) -> H=50, W=50
        # Target: 100x50 (Wide) -> H=50, W=100
        # Should scale to 50x50 (to fit height) and pad width to 100
        input_tensor = torch.ones((1, 3, 50, 50))
        target_w, target_h = 100, 50
        pad_color = 0.5
        
        output = common_upscale(input_tensor, target_w, target_h, upscale_method="nearest", crop="pad", padding_color=pad_color)[0]
        
        self.assertEqual(output.shape, (1, 3, target_h, target_w))
        
        # Check padding (25 pixels on left and right)
        left_pad = output[..., :, :25]
        right_pad = output[..., :, 75:]
        center = output[..., :, 25:75]
        
        self.assertTrue(torch.allclose(left_pad, torch.tensor(pad_color)))
        self.assertTrue(torch.allclose(right_pad, torch.tensor(pad_color)))
        self.assertTrue(torch.allclose(center, torch.ones_like(center)))

    def test_common_upscale_crop_pad_tuple_color(self):
        """Test padding behavior with a per-channel tuple color."""
        # Input: 50x50 -> H=50, W=50
        # Target: 50x100 (Tall) -> H=100, W=50
        # Should scale to 50x50 (to fit width) and pad height to 100
        input_tensor = torch.ones((1, 3, 50, 50))
        target_w, target_h = 50, 100
        pad_color = (0.1, 0.2, 0.3)
        
        output = common_upscale(input_tensor, target_w, target_h, upscale_method="nearest", crop="pad", padding_color=pad_color)[0]
        
        self.assertEqual(output.shape, (1, 3, target_h, target_w))
        
        # Check padding (25 pixels top and bottom)
        top_pad = output[..., :25, :]
        bottom_pad = output[..., 75:, :]
        center = output[..., 25:75, :]
        
        # Check channels for top pad
        self.assertTrue(torch.allclose(top_pad[:, 0, ...], torch.tensor(0.1)))
        self.assertTrue(torch.allclose(top_pad[:, 1, ...], torch.tensor(0.2)))
        self.assertTrue(torch.allclose(top_pad[:, 2, ...], torch.tensor(0.3)))
        self.assertTrue(torch.allclose(center, torch.ones_like(center)))

    def test_common_upscale_5d_input(self):
        """Test that 5D inputs (e.g. video batches) are handled correctly."""
        # Input: B=1, F=2, C=3, H=10, W=10
        input_tensor = torch.zeros((1, 2, 3, 10, 10))
        target_w, target_h = 20, 20
        
        output = common_upscale(input_tensor, target_w, target_h, upscale_method="nearest", crop="pad", padding_color=1.0)[0]
        
        # Should maintain 5D structure
        self.assertEqual(output.shape, (1, 2, 3, target_h, target_w))
        
        target_w_pad, target_h_pad = 40, 20 # Wide target
        output_pad = common_upscale(input_tensor, target_w_pad, target_h_pad, upscale_method="nearest", crop="pad", padding_color=1.0)[0]
        
        self.assertEqual(output_pad.shape, (1, 2, 3, target_h_pad, target_w_pad))
        
        # Check padding (should be 1.0)
        left_pad = output_pad[..., :, :10]
        self.assertTrue(torch.allclose(left_pad, torch.tensor(1.0)))
        
        # Check content (should be 0.0)
        center = output_pad[..., :, 10:30]
        self.assertTrue(torch.allclose(center, torch.tensor(0.0)))

    def test_common_upscale_list_input(self):
        """Test that a list of tensors is handled correctly."""
        t1 = torch.rand((1, 3, 64, 64))
        t2 = torch.rand((1, 3, 64, 64))
        samples = [t1, t2]
        target_w, target_h = 128, 128
        
        outputs = common_upscale(samples, target_w, target_h)
        
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (1, 3, target_h, target_w))
        self.assertEqual(outputs[1].shape, (1, 3, target_h, target_w))

    def test_common_upscale_no_crop_resizes_directly(self):
        """Test that crop != 'center' and != 'pad' (default/fallback) just stretches the image."""
        # Input: 10x10
        # Target: 20x40
        input_tensor = torch.ones((1, 1, 10, 10))
        target_w, target_h = 20, 40
        
        # Using 'stretch' or any unknown string acts as direct resize/stretch
        output = common_upscale(input_tensor, target_w, target_h, upscale_method="nearest", crop="stretch")[0]
        
        self.assertEqual(output.shape, (1, 1, target_h, target_w))
