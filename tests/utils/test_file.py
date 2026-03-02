import unittest
from unittest.mock import patch, MagicMock
from exiv.utils.file import _interactive_download_check
from exiv.config import global_config

class TestFileUtilities(unittest.TestCase):
    def setUp(self):
        self.original_auto_download = global_config.auto_download
        global_config.auto_download = False

    def tearDown(self):
        global_config.auto_download = self.original_auto_download

    def _setup_mock_head(self, mock_head):
        mock_response = MagicMock()
        mock_response.headers = {'content-length': '1048576'} # 1 MB
        mock_head.return_value = mock_response

    @patch("exiv.utils.file.requests.head")
    @patch("builtins.input")
    def test_interactive_download_check_yes(self, mock_input, mock_head):
        """Test that user answering 'yes' enables download for current file but doesn't set global config."""
        self._setup_mock_head(mock_head)
        mock_input.return_value = "y"
        
        result = _interactive_download_check("dummy_model.safetensors", "http://dummy")
        
        self.assertTrue(result)
        self.assertFalse(global_config.auto_download)

    @patch("exiv.utils.file.requests.head")
    @patch("builtins.input")
    def test_interactive_download_check_no(self, mock_input, mock_head):
        """Test that user answering 'no' returns False and doesn't download."""
        self._setup_mock_head(mock_head)
        mock_input.return_value = "n"
        
        result = _interactive_download_check("dummy_model.safetensors", "http://dummy")
        
        self.assertFalse(result)
        self.assertFalse(global_config.auto_download)

    @patch("exiv.utils.file.requests.head")
    @patch("builtins.input")
    def test_interactive_download_check_always(self, mock_input, mock_head):
        """Test that user answering 'always' ('a') enables download and sets the global config."""
        self._setup_mock_head(mock_head)
        mock_input.return_value = "a"
        
        result = _interactive_download_check("dummy_model.safetensors", "http://dummy")
        
        self.assertTrue(result)
        self.assertTrue(global_config.auto_download)

    @patch("exiv.utils.file.requests.head")
    @patch("builtins.input")
    def test_interactive_download_check_already_enabled(self, mock_input, mock_head):
        """Test that if auto_download is already enabled globally, the prompt is skipped."""
        global_config.auto_download = True
        
        result = _interactive_download_check("dummy_model.safetensors", "http://dummy")
        
        self.assertTrue(result)
        # should return early without prompting or checking file size
        mock_input.assert_not_called()
        mock_head.assert_not_called()
