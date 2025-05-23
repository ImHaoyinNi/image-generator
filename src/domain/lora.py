from pathlib import Path

from src.service.utils import get_root_path

LORA_DIR = get_root_path()/"models/lora"
LORA_NAME = "MoriiMee_Gothic_Niji_Style__Pony_LoRA.safetensors"
class LoraInfo:
    def __init__(self, name: str = LORA_NAME, scale: float = 0.8, lora_dir: str = LORA_DIR):
        self.name = name
        self.scale = scale
        self.path = Path(lora_dir) / name if not Path(name).is_absolute() else Path(name)