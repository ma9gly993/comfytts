import numpy as np
import torch
import os
import nltk

import comfy.model_management as model_management
import folder_paths

from .BARK_REPO.generation import generate_text_semantic, SAMPLE_RATE
from .BARK_REPO.api import semantic_to_waveform

# GEN_TEMP = 0.6
# silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

# set the models directory
if "comfytts_voicesamples" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "comfytts", "VoiceSamples")]
    folder_paths.supported_pt_extensions.add(".npz")
    folder_paths.folder_names_and_paths["comfytts_voicesamples"] = (current_paths, folder_paths.supported_pt_extensions)
    os.makedirs(current_paths[0], exist_ok=True)


class TextToSpeech:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": ""}),
                # "data_processor": ("DATA_PROCESSOR", {"tooltip": ""}),
                "voice_preset": (folder_paths.get_filename_list("comfytts_voicesamples"),),

            },
            "optional": {
                "text_splitting": (
                    ["every line", "auto"],
                    {
                        "default": "auto",
                        "tooltip": "The method used to split text into sentences.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.5,
                        "step": 0.01,
                        "tooltip": "temperature for generation audio",
                    },
                ),
                "silence_time": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Silence after each sentence",
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "Hello World!",
                    },
                ),

            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "inference"
    CATEGORY = "comfytts"

    def inference(self, model, voice_preset, text, text_splitting, temperature, silence_time):
        voices_path, _ = folder_paths.folder_names_and_paths["comfytts_voicesamples"]
        voice_preset_path = os.path.join(voices_path[0], voice_preset)
        try:
            return self.inferenceBarkModel(model, voice_preset_path, text, text_splitting, temperature, silence_time)
        except Exception as e:
            raise Exception(f"Error with model {e}")

    # TODO: AudioNormalize to 0.99
    # TODO: Load Models not from preload_models() -> model return
    # TODO: в Load Bark Model выбрать какую модель загружать (Large, Small, v2 или v1)
    # TODO: также проверить, что она грузится как с HF норм (автоматом), так и из папки
    # TODO: зафиксировать сид как параметр (оч сильно отличаются генерации)

    @torch.inference_mode()
    def inferenceBarkModel(self, model, voice_preset_path, text, text_splitting, temperature, silence_time):
        if text_splitting == "auto":
            text = text.replace("\n", " ").strip()
            sentences = nltk.sent_tokenize(text)
        elif text_splitting == "every line":
            sentences = text.strip().split('\n')
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        silence = np.zeros(int(silence_time * SAMPLE_RATE))
        pieces = []
        for sentence in sentences:
            print("sentence: ", sentence)
            semantic_tokens = generate_text_semantic(
                sentence,
                history_prompt_path=voice_preset_path,
                temp=temperature,
                min_eos_p=0.05,  # this controls how likely the generation is to end
            )

            audio_array = semantic_to_waveform(semantic_tokens, history_prompt=voice_preset_path, )
            pieces += [audio_array, silence.copy()]

        # ComfyUI
        full_audio = np.concatenate(pieces)
        full_audio = np.clip(full_audio, -1.0, 1.0)
        if full_audio.dtype != np.float32:
            full_audio = full_audio.astype(np.float32)
        audio_tensor = torch.from_numpy(full_audio).unsqueeze(0).unsqueeze(0)
        print(f"✅ Final audio tensor shape: {audio_tensor.shape}")

        return ({"waveform": audio_tensor.cpu(), "sample_rate": SAMPLE_RATE},)

