import unittest
import os
import json
import shutil
import tempfile
import sys
import subprocess
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from exiv.components.extension_registry import ExtensionRegistry, EXTENSION_ENTRYPOINT
from exiv.components.extensions import Extension
from exiv.utils.file import CONFIG_FILENAME
from exiv.main import cli



class TestExtensionNamespace(unittest.TestCase):
    def setUp(self):
        # Reset singleton
        ExtensionRegistry._instance = None
        
        # Create temp dir
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Setup config
        self.config_path = os.path.join(self.test_dir, CONFIG_FILENAME)
        
    def tearDown(self):
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
        ExtensionRegistry._instance = None
        
        # Clean up sys.modules to avoid pollution
        to_remove = [k for k in sys.modules if k.startswith("exiv.ext.test_namespace_ext")]
        for k in to_remove:
            del sys.modules[k]

    def test_extension_loads_in_namespace(self):
        """
        Test that an extension is loaded into 'exiv.ext.<name>'
        and can perform relative imports.
        """
        ext_name = "test_namespace_ext"
        ext_dir = os.path.join(self.test_dir, ext_name)
        os.makedirs(ext_dir)
        
        # Create a helper module
        with open(os.path.join(ext_dir, "helper.py"), "w") as f:
            f.write("def get_msg(): return 'hello from helper'")
            
        # Create extension entry point with relative import
        with open(os.path.join(ext_dir, EXTENSION_ENTRYPOINT), "w") as f:
            f.write("from . import helper\n")
            f.write("from exiv.components.extensions import Extension\n\n")
            f.write("class NamespaceExt(Extension):\n")
            f.write("    ID = 'namespace_ext'\n")
            f.write("    DISPLAY_NAME = 'Namespace Ext'\n")
            f.write("    VERSION = '1.0'\n")
            f.write("    def register(self):\n")
            f.write("        return [helper.get_msg()]\n")
            
        # Point config to it
        with open(self.config_path, "w") as f:
            json.dump({"extensions": [ext_dir]}, f)
            
        # Initialize
        registry = ExtensionRegistry.get_instance()
        registry.initialize(run_patches=False)
        
        # 1. Check if extension is registered
        self.assertIn("namespace_ext", registry.extensions)
        
        # 2. Check if module is in sys.modules with correct name
        full_name = f"exiv.ext.{ext_name}"
        self.assertIn(full_name, sys.modules)
        
        # 3. Check if the capability (from relative import) was loaded
        # The 'register' method returns [helper.get_msg()] -> ['hello from helper']
        # This is stored in registry.patches
        self.assertIn("hello from helper", registry.patches)
        
        # 4. Check that the helper is also in the namespace (implicitly)
        # Note: 'helper' itself might not be in sys.modules as a top-level key,
        # but it should be accessible via the parent
        module = sys.modules[full_name]
        self.assertTrue(hasattr(module, 'helper'))
        self.assertEqual(module.helper.get_msg(), 'hello from helper')

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
        
        entry_file = os.path.join(ext_dir, EXTENSION_ENTRYPOINT)
        content = (
            f"from exiv.components.extensions import Extension\n"
            f"\n"
            f"class {class_name}(Extension):\n"
            f"    ID = \"{name}_id\"\n"
            f"    DISPLAY_NAME = \"{name}\"\n"
            f"    VERSION = \"1.0\"\n"
            f"\n"
            f"\n"
            f"    def register(self):\n"
            f"        return []\n"
        )
        with open(entry_file, "w") as f:
            f.write(content)
        return ext_dir

    def test_singleton_pattern(self):
        reg1 = ExtensionRegistry.get_instance()
        reg2 = ExtensionRegistry.get_instance()
        self.assertIs(reg1, reg2)
        
    def test_initialize_no_patches(self):
        """Test that run_patches=False prevents patches from running"""
        # Create extension that patches something
        ext_dir = os.path.join(self.user_ext_dir, "patch_ext")
        os.makedirs(ext_dir)
        with open(os.path.join(ext_dir, EXTENSION_ENTRYPOINT), "w") as f:
            f.write(
                "from exiv.components.extensions import Extension\n"
                "from unittest.mock import MagicMock\n"
                "mock_patch = MagicMock()\n"
                "\n"
                "class PatchExt(Extension):\n"
                "    ID = 'patch_ext'\n"
                "    DISPLAY_NAME = 'Patch Ext'\n"
                "    VERSION = '1.0'\n"
                "    def register(self):\n"
                "        return [mock_patch]\n"
            )
            
        config_path = os.path.join(self.test_dir, CONFIG_FILENAME)
        with open(config_path, "w") as f:
            json.dump({"extensions": [self.user_ext_dir]}, f)
            
        registry = ExtensionRegistry.get_instance()
        registry.initialize(run_patches=False)
        self.assertIn("patch_ext", registry.extensions)
        
        # check that patch was NOT called
        import sys
        patch_module = sys.modules["exiv.ext.patch_ext"]
        patch_module.mock_patch.assert_not_called()
        
        # run patches manually
        registry.run_patches()
        patch_module.mock_patch.assert_called_once()

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

        # run command on the PARENT directory
        # scan this parent directory and add the child 'cli_ext' to config
        result = runner.invoke(cli, ['register', self.user_ext_dir])
        self.assertEqual(result.exit_code, 0)

        # verify exiv_config.json content
        config_path = os.path.join(self.test_dir, CONFIG_FILENAME)
        self.assertTrue(os.path.exists(config_path))
        with open(config_path, "r") as f:
            config = json.load(f)
            expected_rel_path = os.path.relpath(ext_path, self.test_dir)
            
            # checking if any path in config matches our expectation
            # (Windows/Posix path separators might differ so we normalize or check existence)
            found = False
            for p in config["extensions"]:
                if os.path.normpath(p) == os.path.normpath(expected_rel_path):
                    found = True
                    break
            self.assertTrue(found, f"Expected {expected_rel_path} in {config['extensions']}")

    @patch('subprocess.check_call')
    def test_cli_register_fails_on_install_error(self, mock_subprocess):
        """Test that 'exiv register' does NOT add extension if requirements fail to install"""
        runner = CliRunner()
        # dummy extension with requirements
        ext_path = self.create_dummy_extension(self.user_ext_dir, "fail_ext", "FailExt")
        with open(os.path.join(ext_path, "requirements.txt"), "w") as f:
            f.write("non-existent-package-xyz-123")

        # mocking subprocess.check_call to raise CalledProcessError
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, ["pip", "install"])
        result = runner.invoke(cli, ['register', ext_path])
        config_path = os.path.join(self.test_dir, CONFIG_FILENAME)
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                extensions = config.get("extensions", [])
                # path should NOT be in there
                self.assertFalse(any("fail_ext" in p for p in extensions), "Extension should not be registered on failure")
        
        # verify the mock was called
        mock_subprocess.assert_called()

    def test_cli_list_extensions(self):
        """Test 'exiv list extensions' command"""
        runner = CliRunner()
        config_path = os.path.join(self.test_dir, CONFIG_FILENAME)
        with open(config_path, "w") as f:
            json.dump({"extensions": ["some/relative/path"]}, f)
            
        result = runner.invoke(cli, ['list', 'extensions'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("[Built-in Extensions Directory]:", result.output)
        self.assertIn("[Registered Extensions]", result.output)
        self.assertIn("some/relative/path", result.output)

    def test_initialize_loads_from_config(self):
        """Test that registry.initialize() reads exiv_config.json and loads extensions"""
        self.create_dummy_extension(self.user_ext_dir, "config_ext", "ConfigExt")
        config_path = os.path.join(self.test_dir, CONFIG_FILENAME)
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
        
        config_path = os.path.join(self.test_dir, "exiv_config.json")
        with open(config_path, "w") as f:
            json.dump({"extensions": [dir1, dir2]}, f)
            
        registry = ExtensionRegistry.get_instance()
        registry.initialize()
        
        self.assertIn("ext_d1_id", registry.extensions)
        self.assertIn("ext_d2_id", registry.extensions)

    def test_config_relative_path_resolution(self):
        """
        Test that relative paths in config are resolved relative to the CONFIG FILE,
        not the CWD
        Structure:
          /root
            /extensions/ext1
            /project
              exiv_config.json -> points to "../extensions"
              /subdir  <- we run from here
        """
        # create directory structure
        root = self.test_dir
        ext_base = os.path.join(root, "extensions")
        project_dir = os.path.join(root, "project")
        run_dir = os.path.join(project_dir, "subdir")
        
        os.makedirs(ext_base)
        os.makedirs(run_dir)
        
        # create the extension
        self.create_dummy_extension(ext_base, "ext1", "ExtOne")
        
        # config in /project, pointing to ../extensions
        config_path = os.path.join(project_dir, "exiv_config.json")
        encoded_rel_path = os.path.join("..", "extensions") 
        with open(config_path, "w") as f:
            json.dump({"extensions": [encoded_rel_path]}, f)
            
        # change CWD to /project/subdir
        os.chdir(run_dir)
        
        # initialize registry
        # it should find config in ../exiv_config.json
        # and resolve ../extensions relative to ../
        registry = ExtensionRegistry.get_instance()
        registry.initialize(run_patches=False)
        self.assertIn("ext1_id", registry.extensions)

if __name__ == '__main__':
    unittest.main()

