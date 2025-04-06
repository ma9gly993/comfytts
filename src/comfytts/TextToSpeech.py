import numpy as np
import torch
import os
import scipy
from transformers import BarkModel

import comfy.model_management as model_management
import folder_paths


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
                "data_processor": ("DATA_PROCESSOR", {"tooltip": ""}),
                "voice_preset": (folder_paths.get_filename_list("comfytts_voicesamples"),),
                "text": (
                    "STRING",
                    {
                        "multiline": True,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "Hello World!",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "inference"
    CATEGORY = "comfytts"

    def inference(self, model, data_processor, voice_preset, text):
        voices_path, _ = folder_paths.folder_names_and_paths["comfytts_voicesamples"]
        voice_preset_path = os.path.join(voices_path[0], voice_preset)
        try:
            if isinstance(model, BarkModel):
                return self.inferenceBarkModel(model, data_processor, voice_preset_path, text)
        except Exception as e:
            raise Exception(f"Error with model {e}")


    @torch.inference_mode()
    def inferenceBarkModel(self, model, data_processor, voice_preset_path, text):
        print("✅ Start inference with Bark")
        inputs = data_processor(text, voice_preset=voice_preset_path)
        torch_device = model_management.get_torch_device()
        inputs = {key: value.to(torch_device) for key, value in inputs.items()}
        audio_array = model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()

        sample_rate = model.generation_config.sample_rate

        # Ensure audio is float32
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Ensure shape is (batch, channels, samples)
        if audio_array.ndim == 1:
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).unsqueeze(0)  # Add channels dim -> (1, 1, samples)
        else:
            raise ValueError(f"Unexpected audio array shape: {audio_array.shape}")

        print("✅ End inference with Bark! shape_audio: ", audio_tensor.shape)

        torch.cuda.empty_cache()
        return ({"waveform": audio_tensor, "sample_rate": sample_rate},)


