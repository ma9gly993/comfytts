import torch
from torchaudio.transforms import Resample
from typing import Tuple

import comfy.model_management
from ._types import AUDIO


class AudioCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
            },
            "optional": {
                "method": (
                    ["add", "mean", "subtract", "multiply", "divide"],
                    {
                        "default": "add",
                        "tooltip": "The method used to combine the audio waveforms.",
                    },
                ),
                "volume_1": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Volume for the first audio track (0 to 1).",
                    },
                ),
                "volume_2": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Volume for the second audio track (0 to 1).",
                    },
                ),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "audio"
    DESCRIPTION = "Combine two audio tracks by overlaying their waveforms."

    def main(
        self,
        audio_1: AUDIO,
        audio_2: AUDIO,
        method: str = "add",
        volume_1: float = 1.0,
        volume_2: float = 1.0,
    ) -> Tuple[AUDIO]:

        waveform_1: torch.Tensor = audio_1["waveform"]
        input_sample_rate_1: int = audio_1["sample_rate"]

        waveform_2: torch.Tensor = audio_2["waveform"]
        input_sample_rate_2: int = audio_2["sample_rate"]

        device: torch.device = comfy.model_management.get_torch_device()

        # Resample the audio if the sample rates are different
        if input_sample_rate_1 != input_sample_rate_2:
            if input_sample_rate_1 < input_sample_rate_2:
                resample = Resample(input_sample_rate_1, input_sample_rate_2).to(device)
                waveform_1: torch.Tensor = resample(waveform_1.to(device))
                waveform_1.to("cpu")
                output_sample_rate = input_sample_rate_2
            else:
                resample = Resample(input_sample_rate_2, input_sample_rate_1).to(device)
                waveform_2: torch.Tensor = resample(waveform_2.to(device))
                waveform_2.to("cpu")
                output_sample_rate = input_sample_rate_1
        else:
            output_sample_rate = input_sample_rate_1

        # Ensure the audio is the same length
        min_length = min(waveform_1.shape[-1], waveform_2.shape[-1])
        if waveform_1.shape[-1] != min_length:
            waveform_1 = waveform_1[..., :min_length]
        if waveform_2.shape[-1] != min_length:
            waveform_2 = waveform_2[..., :min_length]

        # Ensure both waveforms are on the same device before performing the operation
        waveform_1 = waveform_1.to(device)
        waveform_2 = waveform_2.to(device)

        # Apply volume adjustment
        waveform_1 *= volume_1
        waveform_2 *= volume_2

        match method:
            case "add":
                waveform = waveform_1 + waveform_2
            case "subtract":
                waveform = waveform_1 - waveform_2
            case "multiply":
                waveform = waveform_1 * waveform_2
            case "divide":
                waveform = waveform_1 / waveform_2
            case "mean":
                waveform = (waveform_1 + waveform_2) / 2

        return (
            {
                "waveform": waveform.cpu(),
                "sample_rate": output_sample_rate,
            },
        )
