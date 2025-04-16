import os
from io import BytesIO
from typing import IO

import numpy as np
import torch
import torchaudio

import httpx

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

class TTS_ElevenLabs_api:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "Hello world!",
                    },
                ),
                "ELEVENLABS_API_KEY": (
                    "STRING",
                    {
                        "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "Input your elevenlabs api key here",
                    }
                )
            },

            "optional": {
                "PROXY": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "It can be empty. Proxy format: 'host:port:username:password.'",
                    }
                )

            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "main"
    CATEGORY = "comfytts"

    def main(self, text, PROXY, ELEVENLABS_API_KEY):
        timeout = 60
        follow_redirects = True

        if PROXY:
            host, port, username, password = PROXY.split(':')
            proxy_url = f"socks5h://{username}:{password}@{host}:{port}"

            httpx_client = httpx.Client(proxy=proxy_url, timeout=timeout, follow_redirects=follow_redirects)
        else:
            httpx_client = httpx.Client(timeout=timeout, follow_redirects=follow_redirects)

        client = ElevenLabs(
            api_key=ELEVENLABS_API_KEY,
            httpx_client=httpx_client
        )

        response = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
            optimize_streaming_latency="0",
            output_format="pcm_22050",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.5,
                use_speaker_boost=True,
            ),
        )

        print("Streaming audio data...")

        # Получаем байты из ответа
        pcm_bytes = BytesIO()
        for chunk in response:
            pcm_bytes.write(chunk)

        # Переходим в начало потока
        pcm_bytes.seek(0)

        # Читаем в numpy
        pcm_array = np.frombuffer(pcm_bytes.read(), dtype=np.int16)

        # Преобразуем в тензор
        waveform = torch.from_numpy(pcm_array).float() / 32768.0  # нормализация до [-1.0, 1.0]
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, X]

        # Возвращаем в формате ComfyUI
        return ({"waveform": waveform.cpu(), "sample_rate": 22050},)



