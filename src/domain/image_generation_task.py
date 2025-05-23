import datetime
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

from src.domain.image_generation_request import LoraInput
from src.domain.lora import LoraInfo


class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    FINISHED = "FINISHED"
    TIMED_OUT = "TIMED_OUT"

@dataclass
class ImageGenerationTask:
    _id_counter: int = field(init=False, repr=False, default=0)
    task_id: int = field(init=False)
    pos_prompt: str
    neg_prompt: str
    num_inference_steps: int
    cfg: float
    height: int
    width: int
    base_model: Optional[str]
    loras: Optional[List[LoraInfo]] = None
    seed: Optional[int] = None
    create_time: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)
    end_time: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING

    def __post_init__(self):
        # Auto-increment task_id
        type(self)._id_counter += 1
        self.task_id = type(self)._id_counter


if __name__ == '__main__':
    task1 = ImageGenerationTask(pos_prompt="pos",
                                neg_prompt="neg",
                                num_inference_steps=20,
                                cfg=7,
                                height=1024,
                                width=1024,
                                base_model="")
    task2 = ImageGenerationTask(pos_prompt="pos",
                                neg_prompt="neg",
                                num_inference_steps=20,
                                cfg=7,
                                height=1024,
                                width=1024,
                                base_model="")
    print(task1)
    print(task2)