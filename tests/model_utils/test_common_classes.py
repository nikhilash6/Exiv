import unittest
from dataclasses import asdict

from exiv.model_utils.common_classes import Conditioning

class TestConditioning(unittest.TestCase):
    
    def test_standard_json(self):
        """Test from_json with valid standardized keys"""
        standard_json = {
            "group_name": "positive",
            "input_metadata": "A beautiful sunset",
            "timestep_range": [0.1, 0.9],
            "frame_range": [0, 40],
            "strength": 0.75,
            "aux": [
                { "type": "ref_img", "input_metadata": "img.jpg", "timestep_range": [0.0, 1.0] }
            ]
        }
        
        cond = Conditioning.from_json(standard_json)
        self.assertIsNotNone(cond)
        self.assertEqual(cond.input_metadata, "A beautiful sunset")
        self.assertEqual(cond.timestep_range, (0.1, 0.9))
        self.assertEqual(cond.frame_range, (0, 40))
        self.assertEqual(cond.strength, 0.75)
        self.assertEqual(len(cond.aux), 1)
        self.assertEqual(cond.aux[0].type, "ref_img")
        self.assertEqual(cond.aux[0].input_metadata, "img.jpg")

    def test_legacy_keys_rejection(self):
        """Test to ensure legacy keys like 'text' are rejected"""
        legacy_json = {
            "text": "Legacy text key",
            "time_range": [0.0, 1.0],
        }
        # Should fail/return None because 'data' key is missing
        cond = Conditioning.from_json(legacy_json)
        self.assertIsNone(cond, "Should reject legacy keys without 'data' field")

    def test_partial_standard_keys(self):
        """Test with minimal standard keys"""
        partial_json = {"group_name": "positive", "input_metadata": "Just content"}
        cond = Conditioning.from_json(partial_json)
        self.assertIsNotNone(cond)
        self.assertEqual(cond.input_metadata, "Just content")
        self.assertEqual(cond.timestep_range, (0, -1))
        self.assertEqual(cond.frame_range, (0, -1))

    def test_aux_list_structure(self):
        """Test auxiliary input list validation"""
        aux_json = {
            "group_name": "positive",
            "input_metadata": "Aux test",
            "aux": [
                { "type": "t1", "input_metadata": "d1" },
                "invalid_item" # Should be ignored safely
            ]
        }
        cond = Conditioning.from_json(aux_json)
        self.assertIsNotNone(cond)
        self.assertEqual(len(cond.aux), 1)
        self.assertEqual(cond.aux[0].type, "t1")
        self.assertEqual(cond.aux[0].input_metadata, "d1")

    def test_complex_robust_json(self):
        """Test with a complex JSON structure including extra and multiple aux types"""
        complex_json = {
            "group_name": "positive",
            "input_metadata": "Complex content",
            "timestep_range": [0.0, 1.0],
            "strength": 1.2,
            "extra": {
                "param1": "value1",
                "nested": { "a": 1 }
            },
            "aux": [
                {
                    "type": "visual_embedding", 
                    "input_metadata": "path/to/embed.pt",
                    "timestep_range": [0.0, 0.5]
                },
                {
                    "type": "ref_latent", 
                    "input_metadata": "path/to/latent.pt",
                    # Defaults check
                },
                {
                    "type": "controlnet",
                    "input_metadata": "pose_image.png",
                    "frame_range": [10, 20]
                }
            ]
        }
        
        cond = Conditioning.from_json(complex_json)
        self.assertIsNotNone(cond)
        self.assertEqual(cond.input_metadata, "Complex content")
        self.assertEqual(cond.strength, 1.2)
        
        # Check extra
        self.assertEqual(cond.extra, {"param1": "value1", "nested": { "a": 1 }})
        
        # Check aux
        self.assertEqual(len(cond.aux), 3)
        
        # Item 1
        self.assertEqual(cond.aux[0].type, "visual_embedding")
        self.assertEqual(cond.aux[0].input_metadata, "path/to/embed.pt")
        self.assertEqual(cond.aux[0].timestep_range, (0.0, 0.5))
        
        # Item 2
        self.assertEqual(cond.aux[1].type, "ref_latent")
        self.assertEqual(cond.aux[1].input_metadata, "path/to/latent.pt")
        self.assertEqual(cond.aux[1].timestep_range, (0.0, -1)) # Default

        # Item 3
        self.assertEqual(cond.aux[2].type, "controlnet")
        self.assertEqual(cond.aux[2].input_metadata, "pose_image.png")
        self.assertEqual(cond.aux[2].frame_range, (10, 20))

if __name__ == "__main__":
    unittest.main()
