from ._types import AUDIO

class GetTimeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "main"
    CATEGORY = "audio"
    DESCRIPTION = "Get audio duration in seconds (float)"

    def main(self, audio: AUDIO):
        waveform = audio["waveform"]  # Tensor
        sample_rate = audio["sample_rate"]

        # Убираем размерности с 1 (batch, channel и т.п.)
        waveform = waveform.squeeze()

        # Убедимся, что у нас осталась либо (samples,) либо (channels, samples)
        if waveform.dim() == 1:
            num_samples = waveform.shape[0]
        elif waveform.dim() == 2:
            num_samples = waveform.shape[1]
        else:
            raise ValueError(f"Unsupported waveform shape after squeeze: {waveform.shape}")

        duration_sec = float(num_samples) / sample_rate
        return (duration_sec,)
