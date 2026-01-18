import torch

import unittest
import os
import shutil
import tempfile
import av
from PIL import Image
from fractions import Fraction

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

    def _create_dummy_video(self, path):
        with av.open(path, mode='w') as container:
            try:
                stream = container.add_stream('libx264', rate=24)
            except:
                stream = container.add_stream('mpeg4', rate=24)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = 'yuv420p'
            stream.time_base = Fraction(1, 24)
            for i in range(5):
                frame = av.VideoFrame(64, 64, 'rgb24')
                frame.pts = i
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def _create_dummy_audio(self, path):
        with av.open(path, mode='w') as container:
            try:
                stream = container.add_stream('libmp3lame', rate=44100)
            except:
                stream = container.add_stream('aac', rate=44100)
            stream.time_base = Fraction(1, 44100)
            frame = av.AudioFrame(format='s16p', layout='stereo', samples=1024)
            frame.sample_rate = 44100
            frame.pts = 0
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def _create_dummy_image(self, path):
        Image.new('RGB', (64, 64), color='red').save(path)

    def _verify_metadata(self, path, expected_metadata):
        retrieved = MediaProcessor.get_metadata(path)
        self.assertIsInstance(retrieved, dict)
        # Handle case-insensitivity which occurs in some formats like MKV
        retrieved_lower = {k.lower(): v for k, v in retrieved.items()}
        for k, v in expected_metadata.items():
            k_low = k.lower()
            self.assertEqual(str(retrieved_lower.get(k_low)), str(v), f"Metadata mismatch for '{k}' in {path}. Retrieved: {retrieved}")

    def test_save_latents_metadata(self):
        video_tensor = torch.randn(1, 3, 5, 64, 64)
        metadata_in = {
            "title": "Latent Test Title",
            "name": "latent_abc",
            "age": 42
        }
        output_paths = MediaProcessor.save_latents_to_media(video_tensor, metadata=metadata_in)
        self.assertEqual(len(output_paths), 1)
        full_path = os.path.join(self.test_dir, output_paths[0])
        self.assertTrue(os.path.exists(full_path))
        self._verify_metadata(full_path, metadata_in)

    def test_mp4_metadata(self):
        path = os.path.join(self.test_dir, "test.mp4")
        self._create_dummy_video(path)
        metadata = {"title": "MP4 Test", "custom_key": "custom_val"}
        MediaProcessor.save_metadata(path, metadata)
        self._verify_metadata(path, metadata)

    def test_mkv_metadata(self):
        path = os.path.join(self.test_dir, "test.mkv")
        self._create_dummy_video(path)
        metadata = {"title": "MKV Test", "author": "Tester", "custom": "value"}
        MediaProcessor.save_metadata(path, metadata)
        self._verify_metadata(path, metadata)

    def test_mp3_metadata(self):
        path = os.path.join(self.test_dir, "test.mp3")
        self._create_dummy_audio(path)
        metadata = {"title": "Audio Test", "artist": "AI"}
        MediaProcessor.save_metadata(path, metadata)
        self._verify_metadata(path, metadata)

    def test_png_metadata(self):
        path = os.path.join(self.test_dir, "test.png")
        self._create_dummy_image(path)
        metadata = {"comment": "Image Test", "software": "Exiv"}
        MediaProcessor.save_metadata(path, metadata)
        self._verify_metadata(path, metadata)
