import datetime
from collections import deque
from dataclasses import dataclass, field, Field
from typing import Deque, Optional, List, Dict

from PIL import Image

from src.domain.image_generation_task import ImageGenerationTask
from src.service.prompts import POS_PROMPT_PREFIX, NEG_PROMPT_PREFIX
from src.service.workflow import LoraInfo, workflow


class ImageGenerator:
    def __init__(self):
        self.tasks: Deque[ImageGenerationTask] = deque()
        self.generated_images: Dict[int, Image.Image] = {}
        self.is_running = False

    def process_task(self):
        if self.is_running:
            return
        if not self.tasks:
            return
        print("Processing task...")
        task: ImageGenerationTask = self.tasks.popleft()
        self.is_running = True
        image: Image.Image = self.generate_image(
            pos_prompt=task.pos_prompt,
            neg_prompt=task.neg_prompt,
            num_inference_steps=task.num_inference_steps,
            cfg=task.cfg,
            height=task.height,
            width=task.width,
            base_model=task.base_model,
            loras=task.loras,
            seed=task.seed,
        )
        self.generated_images[task.task_id] = image
        self.is_running = False

    def generate_image(
        self,
        pos_prompt: str,
        neg_prompt: str = "",
        num_inference_steps: int = 20,
        cfg: float = 7.0,
        height: int = 1024,
        width: int = 1024,
        base_model: str = "",
        loras: list[LoraInfo] | None = None,
        seed: int = None
    ) -> Image.Image | None:
        pos_prompt = POS_PROMPT_PREFIX + pos_prompt
        neg_prompt = NEG_PROMPT_PREFIX + neg_prompt
        # TODO: we should have multiple workflows
        image: Optional[Image.Image]  = workflow.generate_image(
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            num_inference_steps=num_inference_steps,
            cfg=cfg,
            height=height,
            width=width,
            loras=loras,
            seed=seed
        )
        return image

image_generator = ImageGenerator()