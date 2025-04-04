import torch
import os
import scipy

import comfy.model_management as model_management
import folder_paths


# set the models directory
if "comfytts_voicesamples" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "comfytts", "VoiceSamples")]
    folder_paths.supported_pt_extensions.add(".npz")
    folder_paths.folder_names_and_paths["comfytts_voicesamples"] = (current_paths, folder_paths.supported_pt_extensions)


class TextToSpeech:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
        Return a dictionary which contains config for all input fields.
        Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
        Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
        The type can be a list for selection.

        Returns: `dict`:
            - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
            - Value input_fields (`dict`): Contains input fields config:
                * Key field_name (`string`): Name of a entry-point method's argument
                * Value field_config (`tuple`):
                    + First value is a string indicate the type of field or a list for selection.
                    + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
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
    # DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "inference"

    # OUTPUT_NODE = False
    # OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "comfytts"

    def inference(self, model, data_processor, voice_preset, text):
        voices_path, _ = folder_paths.folder_names_and_paths["comfytts_voicesamples"]
        voice_preset_path = os.path.join(voices_path[0], voice_preset)
        try:
            model_name = model.config._name_or_path
            if model_name == "suno/bark":
                self.inferenceBarkModel(model, data_processor, voice_preset_path, text)
        except Exception as e:
            raise Exception(f"Error with model {e}")

    @torch.inference_mode()
    def inferenceBarkModel(self, model, data_processor, voice_preset_path, text):
        print("model: ", model)
        # print("data_processor: ", data_processor)
        print("voice_preset_path: ", voice_preset_path)
        print("text: ", text)

        inputs = data_processor(text, voice_preset=voice_preset_path)
        torch_device = model_management.get_torch_device()
        inputs = {key: value.to(torch_device) for key, value in inputs.items()}
        audio_array = model.generate(**inputs)
        audio_array = audio_array.cpu().numpy().squeeze()
        print("Generated audio array:", audio_array.shape)

        sample_rate = model.generation_config.sample_rate
        scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)

        # torch.cuda.empty_cache()
        return ([audio_array],)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    # @classmethod
    # def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""
