import logging
import colorlog
from ..config import global_config

class AppLogger(logging.Logger):
    def __init__(self, name="app_logger", log_file=None, log_level=logging.DEBUG):
        super().__init__(name, log_level)
        self.log_file = log_file

        self._configure_logging()

    def _configure_logging(self):
        log_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s:%(name)s:%(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            reset=True,
            secondary_log_colors={},
            style="%",
        )
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(log_formatter)
            self.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.addHandler(console_handler)
        
    def set_level(self, log_level: int):
        self.setLevel(log_level)
        self._handler_level = log_level
        for handler in self.handlers:
            handler.setLevel(log_level)

app_logger = AppLogger(log_level=global_config.logging_level)