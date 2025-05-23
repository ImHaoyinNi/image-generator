from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler
from PIL import Image
import os
import gc

from src.domain.lora import LORA_DIR, LoraInfo, LORA_NAME
from src.service.utils import get_root_path

BASE_MODEL_PATH = get_root_path()/"models/checkpoints/perfect_pony/prefectPonyXL_v50.safetensors"

VAE_PATH = "madebyollin/sdxl-vae-fp16-fix"


class Workflow:
    def __init__(
        self,
        base_model_path: Path,
        vae_path: str = VAE_PATH,
        lora_dir: Path = LORA_DIR,
        device: str = "cuda",
        torch_dtype=torch.float16
    ):
        self.base_model_path: Path = base_model_path
        self.vae_path: str = vae_path # We use a default online vae here
        self.device: str = device
        self.dtype = torch_dtype
        self.lora_dir: Path = lora_dir
        self.pipe = None

    def _init_pipeline(self):
        try:
            print("Loading VAE...")
            vae = AutoencoderKL.from_pretrained(
                self.vae_path,
                torch_dtype=self.dtype
            ).to(self.device)

            print("Setting up scheduler...")
            scheduler = EulerAncestralDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear"
            )

            print("Loading SDXL Base Pipeline...")
            print(self.base_model_path)
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                self.base_model_path,
                vae=vae,
                scheduler=scheduler,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16",
            ).to(self.device)

            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("xformers enabled.")
            except Exception as e:
                print(f"Could not enable xformers: {e}. Using attention slicing instead.")
                self.pipe.enable_attention_slicing()

            self.pipe.watermark = None  # Disable watermarking
            print("Pipeline loaded successfully.")

        except Exception as e:
            print(f"CRITICAL: Failed to initialize pipeline: {e}")
            self.pipe = None

    def generate_image(
        self,
        pos_prompt: str,
        neg_prompt: str = "low quality, blurry, worst quality, bad anatomy, disfigured, malformed limbs",
        num_inference_steps: int = 20,
        cfg: float = 7.0,
        height: int = 1024,
        width: int = 1024,
        loras: List[LoraInfo] | None = None,
        seed: int = None
    ) -> Image.Image | None:
        if self.pipe is None:
            self._init_pipeline()

        print(f"Generating image for prompt: '{pos_prompt}'")
        torch.cuda.empty_cache()
        gc.collect()
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        if seed is not None:
            print(f"Using seed: {seed}")

        loaded_loras = []
        try:
            if loras:
                print(f"Applying LoRAs: {[(l.name, l.scale) for l in loras]}")
                for lora in loras:
                    adapter_name = os.path.basename(lora.name).replace('.safetensors', '')
                    if not os.path.exists(lora.path):
                        print(f"Warning: LoRA not found at {lora.path}, skipping.")
                        continue
                    self.pipe.load_lora_weights(
                        str(lora.path.parent),  # LoRA directory (converted to string)
                        weight_name=lora.path.name,  # LoRA file name (converted to string)
                        adapter_name=adapter_name
                    )
                    loaded_loras.append((adapter_name, lora.scale))

                if loaded_loras:
                    adapter_names = [n for n, _ in loaded_loras]
                    self.pipe.set_adapters(adapter_names)
                    print(f"Activated adapters: {adapter_names}")

            cross_attention_kwargs = {"scale": loaded_loras[0][1]} if loaded_loras else None
            with torch.inference_mode():
                image = self.pipe(
                    prompt=pos_prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=cfg,
                    height=height,
                    width=width,
                    generator=generator,
                    cross_attention_kwargs=cross_attention_kwargs
                ).images[0]

            print("Image generated successfully.")
            return image

        except Exception as e:
            print(f"Error during image generation: {e}")
            return None

        finally:
            if loaded_loras:
                print(f"Unloading LoRAs: {[n for n, _ in loaded_loras]}")
                self.pipe.unload_lora_weights()
            torch.cuda.empty_cache()
            gc.collect()

workflow = Workflow(BASE_MODEL_PATH, vae_path=VAE_PATH)


if __name__ == "__main__":
    test_prompt = "1girl, black eyes, black hair, bob haircut, sexy pose, white shirt, red tie, black suit, detailed background, outdoors, sunset lighting, bust shot"
    test_lora = LoraInfo(name=LORA_NAME, scale=1.0)

    img = workflow.generate_image(
        pos_prompt=test_prompt,
        loras=[test_lora],
        seed=42
    )

    if img:
        img.save("test_output_workflow.png")
        print("Image saved to test_output_workflow.png")
