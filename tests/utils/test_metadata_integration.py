import unittest
import torch
import os
import shutil
import tempfile
import subprocess
from exiv.utils.file import MediaProcessor, FilePaths

class TestMediaProcessorIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_output_dir = FilePaths.OUTPUT_DIRECTORY
        FilePaths.OUTPUT_DIRECTORY = self.test_dir

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        FilePaths.OUTPUT_DIRECTORY = self.original_output_dir

    def test_save_and_verify_metadata(self):
        video_tensor = torch.randn(1, 3, 5, 64, 64)
        
        metadata_in = {
            "title": "Integration Test Title",
            "artist": "Automated Tester",
            "comment": "This is a real file test"
        }
        
        output_paths = MediaProcessor.save_latents_to_media(video_tensor, metadata=metadata_in)
        
        self.assertEqual(len(output_paths), 1)
        saved_path = output_paths[0]
        
        self.assertTrue(os.path.exists(saved_path), f"File was not created at {saved_path}")
        self.assertTrue(saved_path.endswith(".mp4"))
        
        ffprobe_metadata = MediaProcessor.get_metadata(saved_path)

        self.assertIsInstance(ffprobe_metadata, dict)
        self.assertEqual(ffprobe_metadata.get("title"), metadata_in["title"])
        self.assertEqual(ffprobe_metadata.get("artist"), metadata_in["artist"])
        self.assertEqual(ffprobe_metadata.get("comment"), metadata_in["comment"])
