import time

from PIL import Image
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from src.domain.image_generation_request import GenerateResponse, GenerateRequest
from src.domain.image_generation_task import ImageGenerationTask
from src.service.image_generator import image_generator
from src.service.workflow import LoraInfo
from src.service.utils import encode_image_to_base64

router = APIRouter()
task_id = 0

@router.post("/generate", response_model=GenerateResponse)
def generate_image(request: GenerateRequest):
    try:
        lora_objs = [LoraInfo(name=l.name, scale=l.scale) for l in request.loras or []]
        task: ImageGenerationTask = ImageGenerationTask(
            pos_prompt=request.pos_prompt,
            neg_prompt=request.neg_prompt,
            num_inference_steps=request.num_inference_steps,
            cfg=request.cfg,
            height=request.height,
            width=request.width,
            base_model=request.base_model,
            loras=lora_objs,
            seed=request.seed,
        )
        image_generator.tasks.append(task)
        max_wait_seconds = 120
        while True:
            waited = time.time() - task.start_time
            if waited > max_wait_seconds:
                raise HTTPException(status_code=500, detail="Image generation timed out")
            if task.task_id in image_generator.generated_images:
                image: Image.Image = image_generator.generated_images[task.task_id]
                if image is None:
                    raise HTTPException(status_code=500, detail="Image generation failed.")
                else:
                    return GenerateResponse(image_base64=encode_image_to_base64(image))
            else:
                time.sleep(5)
        # image = image_generator.generate_image(pos_prompt=request.pos_prompt,neg_prompt=request.neg_prompt,
        #                                num_inference_steps=request.num_inference_steps,
        #                                guidance_scale=request.guidance_scale,
        #                                height=request.height,
        #                                width=request.width,
        #                                base_model=request.base_model,
        #                                loras=lora_objs,
        #                                seed=request.seed)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
