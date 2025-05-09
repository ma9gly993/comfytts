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
from .src.comfytts.AudioCombine import AudioCombine
from .src.comfytts.AudioSpeedShift import AudioSpeedShift
from .src.comfytts.AudioNormalize import AudioNormalize
from .src.comfytts.ChatGPT_api import ChatGPT_api
from .src.comfytts.TTS_ElevenLabs_api import TTS_ElevenLabs_api
from .src.comfytts.GetTimeAudio import GetTimeAudio
from .src.comfytts.ApplyVideoEffect import ApplyVideoEffect



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {"Example": Example,
                       "LoadBarkModel": LoadBarkModel,
                       "TextToSpeech": TextToSpeech,
                       "AudioCombine": AudioCombine,
                       "AudioSpeedShift": AudioSpeedShift,
                       "AudioNormalize": AudioNormalize,
                       "ChatGPTApi": ChatGPT_api,
                       "TTS_ElevenLabs_api": TTS_ElevenLabs_api,
                       "GetTimeAudio": GetTimeAudio,
                       "ApplyVideoEffect": ApplyVideoEffect
                       }

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"Example": "Example Node",
                              "LoadBarkModel": "Load Bark Model",
                              "TextToSpeech": "Text To Speech",
                              "AudioCombine": "Audio Combine",
                              "AudioSpeedShift": "Audio Speed Shift",
                              "AudioNormalize": "Audio Normalize",
                              "ChatGPTApi": "Chat GPT (api)",
                              "TTS_ElevenLabs_api": "Text-To-Speech(TTS) ElevenLabs",
                              "GetTimeAudio": "Get Time Audio",
                              "ApplyVideoEffect": "Apply Effect to Video"
                              }

WEB_DIRECTORY = "./web"
