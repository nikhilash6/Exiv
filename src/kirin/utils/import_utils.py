import importlib.util
import importlib.metadata
from typing import Any, Callable, Optional, Union

from .logging import app_logger

logger = app_logger

def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[tuple[bool, str], bool]:
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists and return_version:
        try:
            distribution_names = importlib.metadata.packages_distributions().get(pkg_name)
            if distribution_names:
                # TODO: assumption of first dist is not always correct, loop through every dist ?
                distribution_name = distribution_names[0]
                package_version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            logger.debug(f"{pkg_name} found via spec but no distribution metadata available.")

    return (package_exists, package_version) if return_version else package_exists

_sentencepiece_available = _is_package_available("sentencepiece")

def is_sentencepiece_available() -> Union[tuple[bool, str], bool]:
    return _sentencepiece_available

def is_protobuf_available() -> bool:
    if importlib.util.find_spec("google") is None:
        return False
    return importlib.util.find_spec("google.protobuf") is not None