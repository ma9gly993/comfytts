import os
from transformers import AutoProcessor, BarkModel

import folder_paths
import comfy.model_management as model_management


# set the models directory
if "comfytts_barkmodel" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "comfytts", "BarkModel")]
    folder_paths.folder_names_and_paths["comfytts_barkmodel"] = (current_paths, folder_paths.supported_pt_extensions)

model_path, _ = folder_paths.folder_names_and_paths["comfytts_barkmodel"]


class LoadBarkModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {}

    RETURN_TYPES = (
        "MODEL",
        "DATA_PROCESSOR",
    )
    RETURN_NAMES = (
        "model",
        "data_processor",
    )
    FUNCTION = "load_models"
    CATEGORY = "comfytts"
    # DESCRIPTION = cleandoc(__doc__)

    # TODO: ИЗМЕНИТЬ ЗАГРУЗКУ МОДЕЛИ
    # def load_models(self):
    #     os.environ["HF_HUB_ENABLE_PROGRESS_BAR"] = "1"  # Включаем прогресс-бар
    #
    #     torch_device = model_management.get_torch_device()
    #     processor = AutoProcessor.from_pretrained("suno/bark", cache_dir=model_path[0])
    #     model = BarkModel.from_pretrained("suno/bark", cache_dir=model_path[0]).to(torch_device)
    #     return (model, processor, )

    def load_models(self):
        os.environ["HF_HUB_ENABLE_PROGRESS_BAR"] = "1"  # Включаем прогресс-бар

        torch_device = model_management.get_torch_device()
        processor = AutoProcessor.from_pretrained("suno/bark")
        model = BarkModel.from_pretrained("suno/bark").to(torch_device)
        return (
            model,
            processor,
        )
