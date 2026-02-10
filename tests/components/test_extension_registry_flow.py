import unittest
import os
import json
import shutil
import tempfile
import sys
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from exiv.components.extension_registry import ExtensionRegistry
from exiv.components.extensions import Extension
from exiv.main import cli

class TestExtensionRegistryFlow(unittest.TestCase):
    def setUp(self):
        # reset singleton
        ExtensionRegistry._instance = None
        
        # create a temporary directory for our "project root"
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # create a fake built-in extensions directory
        self.builtin_dir = os.path.join(self.test_dir, "src", "exiv", "extensions")
        os.makedirs(self.builtin_dir)
        
        # create a fake user extension directory
        self.user_ext_dir = os.path.join(self.test_dir, "my_extensions")
        os.makedirs(self.user_ext_dir)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
        ExtensionRegistry._instance = None

    def create_dummy_extension(self, base_dir, name, class_name):
        """Helper to create a valid extension folder structure"""
        ext_dir = os.path.join(base_dir, name)
        os.makedirs(ext_dir, exist_ok=True)
        
        init_file = os.path.join(ext_dir, "__init__.py")
        content = f"""
        from exiv.components.extensions import Extension

        class {class_name}(Extension):
            ID = "{name}_id"
            DISPLAY_NAME = "{name}"
            VERSION = "1.0"
            SLOT = "NONE"
            
            def register(self):
                return []
        """
        with open(init_file, "w") as f:
            f.write(content)
        return ext_dir

    def test_singleton_pattern(self):
        reg1 = ExtensionRegistry.get_instance()
        reg2 = ExtensionRegistry.get_instance()
        self.assertIs(reg1, reg2)

    def test_load_extensions_direct_path(self):
        """Test loading extensions directly from a given path"""
        self.create_dummy_extension(self.user_ext_dir, "ext1", "ExtOne")
        
        registry = ExtensionRegistry.get_instance()
        registry.load_extensions_from_path(self.user_ext_dir)
        
        self.assertIn("ext1_id", registry.extensions)
        self.assertEqual(registry.extensions["ext1_id"].DISPLAY_NAME, "ext1")

    @patch('subprocess.check_call')
    def test_cli_register_command(self, mock_subprocess):
        """Test the 'exiv register' CLI command"""
        runner = CliRunner()
        
        # dummy extension
        ext_path = self.create_dummy_extension(self.user_ext_dir, "cli_ext", "CliExt")
        with open(os.path.join(self.user_ext_dir, "requirements.txt"), "w") as f:
            f.write("dummy-package")

        # run command
        # we pass the absolute path, but the tool should convert it to relative if possible
        result = runner.invoke(cli, ['register', self.user_ext_dir])
        self.assertEqual(result.exit_code, 0)

        # verify .exivrc content
        config_path = os.path.join(self.test_dir, ".exivrc")
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r") as f:
            config = json.load(f)
            # tool logic prefers relative paths
            expected_rel_path = os.path.relpath(self.user_ext_dir, self.test_dir)
            self.assertIn(expected_rel_path, config["extensions"])

    def test_initialize_loads_from_config(self):
        """Test that registry.initialize() reads .exivrc and loads extensions"""
        self.create_dummy_extension(self.user_ext_dir, "config_ext", "ConfigExt")
        config_path = os.path.join(self.test_dir, ".exivrc")
        with open(config_path, "w") as f:
            json.dump({"extensions": [self.user_ext_dir]}, f)
            
        # initialize registry
        registry = ExtensionRegistry.get_instance()
        registry.initialize()
        self.assertIn("config_ext_id", registry.extensions)

    def test_multiple_registries(self):
        """Test registering multiple folders"""
        dir1 = os.path.join(self.test_dir, "dir1")
        dir2 = os.path.join(self.test_dir, "dir2")
        
        self.create_dummy_extension(dir1, "ext_d1", "ExtD1")
        self.create_dummy_extension(dir2, "ext_d2", "ExtD2")
        
        config_path = os.path.join(self.test_dir, ".exivrc")
        with open(config_path, "w") as f:
            json.dump({"extensions": [dir1, dir2]}, f)
            
        registry = ExtensionRegistry.get_instance()
        registry.initialize()
        
        self.assertIn("ext_d1_id", registry.extensions)
        self.assertIn("ext_d2_id", registry.extensions)
