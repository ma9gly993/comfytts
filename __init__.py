"""Top-level package for comfytts."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Danil"""
__email__ = "maugly2019@mail.ru"
__version__ = "0.0.1"

# from .src.comfytts.nodes import NODE_CLASS_MAPPINGS
# from .src.comfytts.nodes import NODE_DISPLAY_NAME_MAPPINGS

from .src.comfytts.nodes import Example
from .src.comfytts.LoadBarkModel import LoadBarkModel
from .src.comfytts.TextToSpeech import TextToSpeech

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"Example": Example, "LoadBarkModel": LoadBarkModel, "TextToSpeech": TextToSpeech}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"Example": "Example Node", "LoadBarkModel": "Load Bark Model", "TextToSpeech": "Text To Speech"}


WEB_DIRECTORY = "./web"
