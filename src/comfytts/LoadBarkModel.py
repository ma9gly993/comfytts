# import libs
import os
# import comfy dependencies
import folder_paths
import comfy.model_management as model_management
# import custom bark
from .BARK_REPO.generation import preload_models

# set the models directory
if "comfytts_barkmodel" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "comfytts", "BarkModel")]
    folder_paths.folder_names_and_paths["comfytts_barkmodel"] = (current_paths, folder_paths.supported_pt_extensions)

model_path_list, _ = folder_paths.folder_names_and_paths["comfytts_barkmodel"]
model_path = model_path_list[0]
os.makedirs(model_path, exist_ok=True)  # чтобы HuggingFace не жаловался


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


    def load_models(self):
        preload_models()
        return ('model', )

