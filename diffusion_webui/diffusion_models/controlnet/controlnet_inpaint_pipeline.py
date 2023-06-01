import numpy as np
import torch
from diffusers import ControlNetModel
from PIL import Image

from diffusers import StableDiffusionControlNetInpaintPipeline
from diffusion_models.controlnet.base_controlnet_pipeline import ControlnetPipeline
from diffusion_webui.utils.scheduler_list import get_scheduler_list
from diffusion_webui.utils.preprocces_utils import PREPROCCES_DICT


class StableDiffusionControlNetInpaintGenerator(ControlnetPipeline):
    def __init__(self):
        super().__init__()

    def load_model(self, stable_model_path, controlnet_model_path, scheduler):
        if self.pipe is None:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_path, torch_dtype=torch.float16
            )
            self.pipe = (
                StableDiffusionControlNetInpaintPipeline.from_pretrained(
                    pretrained_model_name_or_path=stable_model_path,
                    controlnet=controlnet,
                    safety_checker=None,
                    torch_dtype=torch.float16,
                )
            )

        self.pipe = get_scheduler_list(pipe=self.pipe, scheduler=scheduler)
        self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        return self.pipe

    def load_image(self, image):
        image = np.array(image)
        image = Image.fromarray(image)
        return image

    def controlnet_preprocces(
        self,
        read_image: str,
        preprocces_type: str,
    ):
        processed_image = PREPROCCES_DICT[preprocces_type](read_image)
        return processed_image

    def generate_image(
        self,
        image_path: str,
        stable_model_path: str,
        controlnet_model_path: str,
        prompt: str,
        negative_prompt: str,
        num_images_per_prompt: int,
        height:int,
        width:int,
        strength:int,
        guess_mode: bool,
        guidance_scale: int,
        num_inference_step: int,
        controlnet_conditioning_scale: int,
        scheduler: str,
        seed_generator: int,
        preprocces_type: str,
    ):

        normal_image = image_path["image"].convert("RGB").resize((512, 512))
        mask_image = image_path["mask"].convert("RGB").resize((512, 512))

        normal_image = self.load_image(image=normal_image)
        mask_image = self.load_image(image=mask_image)

        control_image = self.controlnet_preprocces(image=normal_image, preprocces_type=preprocces_type)
        pipe = self.load_model(
            stable_model_path=stable_model_path,
            controlnet_model_path=controlnet_model_path,
            scheduler=scheduler,
        )

        if seed_generator == 0:
            random_seed = torch.randint(0, 1000000, (1,))
            generator = torch.manual_seed(random_seed)
        else:
            generator = torch.manual_seed(seed_generator)

        output = pipe(
            prompt=prompt,
            image=normal_image,
            height=height,
            width=width,
            mask_image=mask_image,
            strength=strength,
            guess_mode=guess_mode,
            control_image=control_image,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_step,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        ).images

        return output
