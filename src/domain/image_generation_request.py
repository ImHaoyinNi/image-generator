from typing import Optional, List

from pydantic import BaseModel, Field


class LoraInput(BaseModel):
    name: str
    scale: float = 1.0

class GenerateRequest(BaseModel):
    pos_prompt: str
    neg_prompt: Optional[str] = Field(default="")
    num_inference_steps: int = Field(default=20, ge=1, le=50)
    cfg: float = Field(default=7.0, ge=1.0, le=20.0)
    height: int = Field(default=1024)
    width: int = Field(default=1024)
    base_model: Optional[str] = None
    loras: Optional[List[LoraInput]] = None
    seed: Optional[int] = None

class GenerateResponse(BaseModel):
    image_base64: str
