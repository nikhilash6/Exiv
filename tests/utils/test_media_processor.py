import unittest
import os
import shutil
import tempfile
import torch
import av
from PIL import Image
from fractions import Fraction
from exiv.utils.file import MediaProcessor, FilePaths

class TestMediaProcessorFPS(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.test_dir, "test_fps.mp4")
        self._create_dummy_video(self.video_path, fps=24, duration_sec=2)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_dummy_video(self, path, fps, duration_sec):
        # Create a video with distinct frames (color changing) to verify sampling if needed
        # For now, just ensuring frame counts is enough
        total_frames = int(fps * duration_sec)
        with av.open(path, mode='w') as container:
            stream = container.add_stream('libx264', rate=fps)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = 'yuv420p'
            stream.time_base = Fraction(1, fps)
            
            for i in range(total_frames):
                frame = av.VideoFrame(64, 64, 'rgb24')
                frame.pts = i
                # frame.time_base = stream.time_base # PyAV handles this usually
                for packet in stream.encode(frame):
                    container.mux(packet)
            
            for packet in stream.encode():
                container.mux(packet)

    def test_load_video_original_fps(self):
        """Test loading video without specifying FPS (should keep original)"""
        frames, meta = MediaProcessor.load_video(self.video_path)
        self.assertAlmostEqual(meta['fps'], 24.0, delta=0.1)
        # 2 seconds @ 24fps = 48 frames
        self.assertEqual(len(frames), 48)
        self.assertEqual(meta['loaded_frames'], 48)

    def test_load_video_downsample(self):
        """Test downsampling video to half FPS"""
        target_fps = 12.0
        frames, meta = MediaProcessor.load_video(self.video_path, fps=target_fps)
        self.assertEqual(meta['fps'], target_fps)
        # 2 seconds @ 12fps = 24 frames
        # Allowing +/- 1 frame tolerance due to rounding/alignment
        self.assertTrue(23 <= len(frames) <= 25, f"Expected ~24 frames, got {len(frames)}")

    def test_load_video_upsample(self):
        """Test upsampling video to double FPS"""
        target_fps = 48.0
        frames, meta = MediaProcessor.load_video(self.video_path, fps=target_fps)
        self.assertEqual(meta['fps'], target_fps)
        # 2 seconds @ 48fps = 96 frames
        self.assertTrue(95 <= len(frames) <= 97, f"Expected ~96 frames, got {len(frames)}")

    def test_load_video_limit_frames(self):
        """Test limiting frame count with FPS resampling"""
        target_fps = 12.0
        limit = 10
        frames, meta = MediaProcessor.load_video(self.video_path, fps=target_fps, limit_frame_count=limit)
        self.assertEqual(len(frames), limit)
        self.assertEqual(meta['loaded_frames'], limit)

import unittest.mock

class TestMediaProcessorIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.patcher = unittest.mock.patch('exiv.utils.file.FilePaths.get_output_directory', return_value=self.test_dir)
        self.patcher.start()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.patcher.stop()

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
