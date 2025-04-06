import os
from transformers import AutoProcessor, BarkModel

import folder_paths
import comfy.model_management as model_management


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
        os.environ["HF_HUB_ENABLE_PROGRESS_BAR"] = "1"
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json"
        ]
        model_files_exist = all(os.path.exists(os.path.join(model_path, f)) for f in required_files)
        if model_files_exist:
            print(f"✅ Load Bark local from {model_path}")
            return self.load_model_local()
        else:
            print(f"⏬ Can't find Bark in path {model_path}. Start Loading weights from HF... This may take a very long time.")
            return self.load_model_HF()

    def load_model_local(self):
        torch_device = model_management.get_torch_device()
        processor = AutoProcessor.from_pretrained(model_path)
        model = BarkModel.from_pretrained(model_path).to(torch_device)
        return (model, processor, )

    def load_model_HF(self):
        torch_device = model_management.get_torch_device()
        processor = AutoProcessor.from_pretrained("suno/bark", cache_dir=model_path)
        model = BarkModel.from_pretrained("suno/bark", cache_dir=model_path).to(torch_device)
        return (model, processor, )
