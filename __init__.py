"""Top-level package for comfytts."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Danil"""
__email__ = "maugly2019@mail.ru"
__version__ = "0.0.1"

from .src.comfytts.nodes import NODE_CLASS_MAPPINGS
from .src.comfytts.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
