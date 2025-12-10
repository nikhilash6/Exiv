import unittest
import os
import shutil
import tempfile
from unittest.mock import patch
import exiv.utils.file_path
from exiv.utils.file_path import FilePaths, FilePathData

class TestFilePaths(unittest.TestCase):
    def setUp(self):
        self.original_roots = FilePaths._search_roots[:]
        self.original_cache = FilePaths._file_cache.copy()
        
        FilePaths._search_roots = []
        FilePaths._file_cache = {}

        # temporary directory and mock file structure
        self.test_dir = tempfile.mkdtemp()
        
        self.folders = [
            "models/checkpoints",
            "models/loras",
            "models/vae",
            "models/loras/nested",
            "custom_folder/my_stuff"
        ]
        
        for folder in self.folders:
            os.makedirs(os.path.join(self.test_dir, folder), exist_ok=True)
            
        self.files = [
            "models/checkpoints/v1-5.ckpt",
            "models/checkpoints/sdxl.safetensors",
            "models/loras/pixel_art.pt",
            "models/loras/details.safetensors",
            "models/loras/nested/style.safetensors",
            "models/vae/vae-ft-mse.pt",
            "custom_folder/my_stuff/custom.bin"
        ]
        
        for f in self.files:
            with open(os.path.join(self.test_dir, f), 'w') as fp:
                fp.write("dummy")

    def tearDown(self):
        # restore original state
        FilePaths._search_roots = self.original_roots
        FilePaths._file_cache = self.original_cache
        
        # remove temporary directory
        shutil.rmtree(self.test_dir)

    def test_add_search_path(self):
        """Test registering a path with default and custom mappings."""
        # Default mapping
        FilePaths.add_search_path(self.test_dir)
        self.assertEqual(len(FilePaths._search_roots), 1)
        self.assertEqual(FilePaths._search_roots[0]["path"], os.path.abspath(self.test_dir))
        self.assertIn("checkpoint", FilePaths._search_roots[0]["map"])

        # Custom mapping
        custom_map = {"special": "custom_folder/my_stuff"}
        custom_path = os.path.join(self.test_dir, "custom")
        os.makedirs(custom_path, exist_ok=True)
        
        FilePaths.add_search_path(custom_path, mapping=custom_map)
        self.assertEqual(len(FilePaths._search_roots), 2)
        self.assertEqual(FilePaths._search_roots[1]["map"]["special"], ["custom_folder/my_stuff"])

    def test_get_files_standard(self):
        """Test retrieving files using the default mapping, ensuring FilePathData return format."""
        # Patch map to empty to strictly test local discovery
        with patch.dict(exiv.utils.file_path.DOWNLOAD_MAP, {}, clear=True):
            FilePaths.add_search_path(self.test_dir)
            
            # Fetch Checkpoints
            ckpts = FilePaths.get_files("checkpoint")
            self.assertEqual(len(ckpts), 2)
            
            # Check structure
            self.assertTrue(all(isinstance(f, FilePathData) for f in ckpts))
            
            # Check content
            names = [f.name for f in ckpts]
            self.assertIn("v1-5.ckpt", names)
            self.assertIn("sdxl.safetensors", names)
            
            # Fetch LoRAs (Recursive check)
            loras = FilePaths.get_files("lora")
            # Should find 3: pixel_art, details, and nested/style
            self.assertEqual(len(loras), 3)
            lora_names = [f.name for f in loras]
            self.assertIn("style.safetensors", lora_names)

    def test_get_files_downloadable(self):
        """Test that missing files in the DOWNLOAD_MAP are returned with is_present=False but valid path."""
        FilePaths.add_search_path(self.test_dir)
        
        mock_map = {
            "flux1-dev.safetensors": {
                "type": "checkpoint",
                "url": "https://dummy.url/flux"
            },
            "sdxl.safetensors": {
                "type": "checkpoint",
                "url": "https://dummy.url/sdxl"
            }
        }
        
        with patch.dict(exiv.utils.file_path.DOWNLOAD_MAP, mock_map, clear=True):
            ckpts = FilePaths.get_files("checkpoint")
            
            # Should have: v1-5.ckpt (local), sdxl.safetensors (local), flux1-dev (downloadable)
            self.assertEqual(len(ckpts), 3)
            
            # Check existing local file with URL
            sdxl = next(f for f in ckpts if f.name == "sdxl.safetensors")
            self.assertTrue(sdxl.is_present)
            self.assertIsNotNone(sdxl.path)
            self.assertTrue(os.path.exists(sdxl.path))
            self.assertEqual(sdxl.url, "https://dummy.url/sdxl")
            
            # Check missing downloadable file
            flux = next(f for f in ckpts if f.name == "flux1-dev.safetensors")
            self.assertFalse(flux.is_present)
            # Ensure path is NOT None and points to save location
            self.assertIsNotNone(flux.path)
            self.assertTrue(flux.path.endswith("flux1-dev.safetensors"))
            self.assertFalse(os.path.exists(flux.path)) # Should not exist yet
            self.assertEqual(flux.url, "https://dummy.url/flux")

    def test_get_files_extensions(self):
        """Test filtering by extension."""
        FilePaths.add_search_path(self.test_dir)
        
        # Only get .ckpt files
        ckpts = FilePaths.get_files("checkpoint", extensions=[".ckpt"])
        self.assertEqual(len(ckpts), 1)
        self.assertEqual(ckpts[0].name, "v1-5.ckpt")

    def test_get_path_exact_match(self):
        """Test resolving an exact filename."""
        FilePaths.add_search_path(self.test_dir)
        
        result = FilePaths.get_path("v1-5.ckpt", "checkpoint")
        self.assertIsInstance(result, FilePathData)
        self.assertTrue(result.is_present)
        self.assertTrue(os.path.exists(result.path))
        self.assertTrue(result.path.endswith("v1-5.ckpt"))

    def test_get_path_stem_match(self):
        """Test resolving a file without providing extension."""
        FilePaths.add_search_path(self.test_dir)
        
        # Request "pixel_art" -> find "pixel_art.pt"
        result = FilePaths.get_path("pixel_art", "lora")
        self.assertIsInstance(result, FilePathData)
        self.assertTrue(result.is_present)
        self.assertTrue(result.path.endswith("pixel_art.pt"))

    def test_get_path_downloadable_fallback(self):
        """Test get_path finding a file in DOWNLOAD_MAP that isn't on disk."""
        FilePaths.add_search_path(self.test_dir)
        
        mock_map = {
            "flux1-dev.safetensors": {
                "type": "checkpoint",
                "url": "https://dummy.url/flux"
            }
        }
        
        with patch.dict(exiv.utils.file_path.DOWNLOAD_MAP, mock_map, clear=True):
            # Exact match
            res1 = FilePaths.get_path("flux1-dev.safetensors", "checkpoint")
            self.assertFalse(res1.is_present)
            self.assertIsNotNone(res1.path)
            self.assertTrue(res1.path.endswith("flux1-dev.safetensors"))
            self.assertFalse(os.path.exists(res1.path))
            self.assertEqual(res1.url, "https://dummy.url/flux")
            
            # Stem match
            res2 = FilePaths.get_path("flux1-dev", "checkpoint")
            self.assertFalse(res2.is_present)
            self.assertEqual(res2.name, "flux1-dev.safetensors")
            self.assertIsNotNone(res2.path)

    def test_get_path_not_found(self):
        """Test not found behavior (now returns empty FilePathData instead of raising)."""
        FilePaths.add_search_path(self.test_dir)
        
        # Non-existent file
        res = FilePaths.get_path("non_existent_model", "checkpoint")
        self.assertIsInstance(res, FilePathData)
        self.assertIsNone(res.name)
        self.assertIsNone(res.path)
        self.assertFalse(res.is_present)
        
        # Exists, but wrong type (looking for ckpt in lora folder)
        res2 = FilePaths.get_path("v1-5.ckpt", "lora")
        self.assertIsInstance(res2, FilePathData)
        self.assertIsNone(res2.name)

    def test_multiple_search_roots(self):
        """Test combining files from multiple root directories."""
        root2 = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(root2, "models/checkpoints"), exist_ok=True)
            with open(os.path.join(root2, "models/checkpoints/model2.ckpt"), 'w') as f:
                f.write("dummy")
            
            with patch.dict(exiv.utils.file_path.DOWNLOAD_MAP, {}, clear=True):
                FilePaths.add_search_path(self.test_dir)
                FilePaths.add_search_path(root2)
                
                ckpts = FilePaths.get_files("checkpoint")
                self.assertEqual(len(ckpts), 3) # 2 from test_dir, 1 from root2
                
                names = [f.name for f in ckpts]
                self.assertIn("model2.ckpt", names)
        finally:
            shutil.rmtree(root2)

    def test_cache_reset_on_add(self):
        """Ensure adding a path clears the cache to allow discovery of new files."""
        FilePaths.add_search_path(self.test_dir)
        
        # Initialize cache
        _ = FilePaths.get_files("checkpoint")
        self.assertTrue(len(FilePaths._file_cache) > 0)
        
        # Add new path (even if same, it triggers logic)
        FilePaths.add_search_path(self.test_dir) 
        self.assertEqual(FilePaths._file_cache, {}) # Cache should be empty

    def test_get_save_folder(self):
        """Test retrieving the default save folder for a type."""
        FilePaths.add_search_path(self.test_dir)
        
        # Standard type
        save_path = FilePaths.get_save_folder("checkpoint")
        expected = os.path.join(os.path.abspath(self.test_dir), "models/checkpoints")
        self.assertEqual(save_path, expected)
        self.assertTrue(os.path.exists(save_path))
        
        # Type with no folder (should error)
        with self.assertRaises(ValueError):
            FilePaths.get_save_folder("non_existent_type")
            
        # Ensure it uses the first registered root
        root2 = tempfile.mkdtemp()
        try:
            FilePaths._search_roots = [] # clear existing
            FilePaths.add_search_path(root2)
            FilePaths.add_search_path(self.test_dir)
            
            save_path = FilePaths.get_save_folder("checkpoint")
            self.assertTrue(save_path.startswith(os.path.abspath(root2)))
        finally:
            shutil.rmtree(root2)