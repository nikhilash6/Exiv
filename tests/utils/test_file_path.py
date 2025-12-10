import unittest
import os
import shutil
import tempfile
from exiv.utils.file_path import FilePaths



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
        """Test retrieving files using the default mapping."""
        FilePaths.add_search_path(self.test_dir)
        
        # Fetch Checkpoints
        ckpts = FilePaths.get_files("checkpoint")
        self.assertEqual(len(ckpts), 2)
        self.assertTrue(any("v1-5.ckpt" in f for f in ckpts))
        self.assertTrue(any("sdxl.safetensors" in f for f in ckpts))
        
        # Fetch LoRAs (Recursive check)
        loras = FilePaths.get_files("lora")
        # Should find 3: pixel_art, details, and nested/style
        self.assertEqual(len(loras), 3)
        self.assertTrue(any("style.safetensors" in f for f in loras))

    def test_get_files_extensions(self):
        """Test filtering by extension."""
        FilePaths.add_search_path(self.test_dir)
        
        # Only get .ckpt files
        ckpts = FilePaths.get_files("checkpoint", extensions=[".ckpt"])
        self.assertEqual(len(ckpts), 1)
        self.assertTrue(ckpts[0].endswith("v1-5.ckpt"))

    def test_get_path_exact_match(self):
        """Test resolving an exact filename."""
        FilePaths.add_search_path(self.test_dir)
        
        path = FilePaths.get_path("v1-5.ckpt", "checkpoint")
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith("v1-5.ckpt"))

    def test_get_path_stem_match(self):
        """Test resolving a file without providing extension."""
        FilePaths.add_search_path(self.test_dir)
        
        # Request "pixel_art" -> find "pixel_art.pt"
        path = FilePaths.get_path("pixel_art", "lora")
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.endswith("pixel_art.pt"))

    def test_get_path_not_found(self):
        """Test validation errors."""
        FilePaths.add_search_path(self.test_dir)
        
        with self.assertRaises(FileNotFoundError):
            FilePaths.get_path("non_existent_model", "checkpoint")
            
        with self.assertRaises(FileNotFoundError):
            # exists, but wrong type (looking for ckpt in lora folder)
            FilePaths.get_path("v1-5.ckpt", "lora")

    def test_multiple_search_roots(self):
        """Test combining files from multiple root directories."""
        root2 = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(root2, "models/checkpoints"), exist_ok=True)
            with open(os.path.join(root2, "models/checkpoints/model2.ckpt"), 'w') as f:
                f.write("dummy")
            
            FilePaths.add_search_path(self.test_dir)
            FilePaths.add_search_path(root2)
            
            ckpts = FilePaths.get_files("checkpoint")
            self.assertEqual(len(ckpts), 3) # 2 from test_dir, 1 from root2
            self.assertTrue(any("model2.ckpt" in f for f in ckpts))
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

