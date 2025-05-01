from PIL import Image

from src.service.prompts import POS_PROMPT_PREFIX, NEG_PROMPT_PREFIX
from src.service.workflow import LoraInfo, workflow


class ImageGenerator:
    def __init__(self):
        pass

    def generate_image(
        self,
        pos_prompt: str,
        neg_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.0,
        height: int = 1024,
        width: int = 1024,
        base_model: str = "",
        loras: list[LoraInfo] | None = None,
        seed: int = None
    ) -> Image.Image | None:
        pos_prompt = POS_PROMPT_PREFIX + pos_prompt
        neg_prompt = NEG_PROMPT_PREFIX + neg_prompt
        # TODO: we should have multiple workflows
        image = workflow.generate_image(
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            loras=loras,
            seed=seed
        )
        return image

image_generator = ImageGenerator()