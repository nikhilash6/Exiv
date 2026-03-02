__version__ = "0.1"

# core configurations and loggers
from .config import global_config
from .utils.logging import app_logger
# server stuff
from .server.task_manager import task_manager
from .server.server import start_worker, run_server
# extension management
from .components.extension_registry import ExtensionRegistry

app_logger.set_level(global_config.logging_level)

__all__ = [
    "global_config",
    "app_logger",
    "task_manager",
    "start_worker",
    "run_server",
    "ExtensionRegistry",
    "__version__",
]
