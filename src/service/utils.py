import base64
from io import BytesIO
from pathlib import Path

from PIL import Image


_root_path: Path | None = None


def get_root_path() -> Path:
    current_path = Path(__file__).resolve()
    print(current_path)
    while current_path != current_path.parent:
        if (current_path / ".git").exists():  # Change marker if needed
            return current_path
        current_path = current_path.parent
    return current_path

def encode_image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")