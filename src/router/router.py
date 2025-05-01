from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from src.service.image_generator import image_generator
from src.service.workflow import LoraInfo
from src.service.utils import encode_image_to_base64

router = APIRouter()

class LoraInput(BaseModel):
    name: str
    scale: float = 1.0

class GenerateRequest(BaseModel):
    pos_prompt: str
    neg_prompt: Optional[str] = Field(default="")
    num_inference_steps: int = Field(default=30, ge=1, le=100)
    guidance_scale: float = Field(default=7.0, ge=1.0, le=20.0)
    height: int = Field(default=1024)
    width: int = Field(default=1024)
    base_model: Optional[str] = None
    loras: Optional[List[LoraInput]] = None
    seed: Optional[int] = None

class GenerateResponse(BaseModel):
    image_base64: str

@router.post("/generate", response_model=GenerateResponse)
def generate_image(request: GenerateRequest):
    try:
        lora_objs = [LoraInfo(name=l.name, scale=l.scale) for l in request.loras or []]
        image = image_generator.generate_image(pos_prompt=request.pos_prompt,neg_prompt=request.neg_prompt,
                                       num_inference_steps=request.num_inference_steps,
                                       guidance_scale=request.guidance_scale,
                                       height=request.height,
                                       width=request.width,
                                       base_model=request.base_model,
                                       loras=lora_objs,
                                       seed=request.seed)
        if image is None:
            raise HTTPException(status_code=500, detail="Image generation failed.")
        return GenerateResponse(image_base64=encode_image_to_base64(image))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
