import torch
import math
import torchaudio.functional as F
from typing import Tuple

from ._types import AUDIO


class AudioNormalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO",)
    CATEGORY = "audio"
    DESCRIPTION = "Time-stretch or time-compress audio by a given rate. A rate of 2.0 will double the speed of the audio, while a rate of 0.5 will halve the speed."

    def main(self, audio: AUDIO,) -> Tuple[AUDIO]:
        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = audio["sample_rate"]
        # rate = min(max(rate, 0.1), 10.0)
        # shifted = self.time_shift(waveform, rate)

        peak = waveform.abs().max()

        if peak == 0:
            return (audio,)  # тишина, возвращаем без изменений

        target_peak = 0.99
        # Нормализуем
        waveform = waveform * (target_peak / peak)

        return ({"waveform": waveform.cpu(), "sample_rate": sample_rate},)


