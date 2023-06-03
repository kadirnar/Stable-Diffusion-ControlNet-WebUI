import gradio as gr
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from PIL import Image

from diffusion_webui.diffusion_models.base_controlnet_pipeline import (
    ControlnetPipeline,
)
from diffusion_webui.utils.model_list import (
    controlnet_model_list,
    stable_model_list,
)
from diffusion_webui.utils.preprocces_utils import PREPROCCES_DICT
from diffusion_webui.utils.scheduler_list import (
    SCHEDULER_MAPPING,
    get_scheduler,
)


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

        self.pipe = get_scheduler(pipe=self.pipe, scheduler=scheduler)
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
        height: int,
        width: int,
        strength: int,
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

        control_image = self.controlnet_preprocces(
            read_image=normal_image, preprocces_type=preprocces_type
        )
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
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            generator=generator,
        ).images

        return output

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    controlnet_inpaint_image_path = gr.Image(
                        source="upload",
                        tool="sketch",
                        elem_id="image_upload",
                        type="pil",
                        label="Upload",
                    ).style(height=260)

                    controlnet_inpaint_prompt = gr.Textbox(
                        lines=1, placeholder="Prompt", show_label=False
                    )
                    controlnet_inpaint_negative_prompt = gr.Textbox(
                        lines=1, placeholder="Negative Prompt", show_label=False
                    )

                    with gr.Row():
                        with gr.Column():
                            controlnet_inpaint_stable_model_path = gr.Dropdown(
                                choices=stable_model_list,
                                value=stable_model_list[0],
                                label="Stable Model Path",
                            )
                            controlnet_inpaint_preprocces_type = gr.Dropdown(
                                choices=list(PREPROCCES_DICT.keys()),
                                value=list(PREPROCCES_DICT.keys())[0],
                                label="Preprocess Type",
                            )
                            controlnet_inpaint_conditioning_scale = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                value=1.0,
                                label="ControlNet Conditioning Scale",
                            )
                            controlnet_inpaint_guidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=7.5,
                                label="Guidance Scale",
                            )
                            controlnet_inpaint_height = gr.Slider(
                                minimum=128,
                                maximum=1280,
                                step=32,
                                value=512,
                                label="Height",
                            )
                            controlnet_inpaint_width = gr.Slider(
                                minimum=128,
                                maximum=1280,
                                step=32,
                                value=512,
                                label="Width",
                            )
                            controlnet_inpaint_guess_mode = gr.Checkbox(
                                label="Guess Mode"
                            )

                        with gr.Column():
                            controlnet_inpaint_model_path = gr.Dropdown(
                                choices=controlnet_model_list,
                                value=controlnet_model_list[0],
                                label="ControlNet Model Path",
                            )
                            controlnet_inpaint_scheduler = gr.Dropdown(
                                choices=list(SCHEDULER_MAPPING.keys()),
                                value=list(SCHEDULER_MAPPING.keys())[0],
                                label="Scheduler",
                            )
                            controlnet_inpaint_strength = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=7.5,
                                label="Strength",
                            )
                            controlnet_inpaint_num_inference_step = gr.Slider(
                                minimum=1,
                                maximum=150,
                                step=1,
                                value=30,
                                label="Num Inference Step",
                            )
                            controlnet_inpaint_num_images_per_prompt = (
                                gr.Slider(
                                    minimum=1,
                                    maximum=4,
                                    step=1,
                                    value=1,
                                    label="Number Of Images",
                                )
                            )
                            controlnet_inpaint_seed_generator = gr.Slider(
                                minimum=0,
                                maximum=1000000,
                                step=1,
                                value=0,
                                label="Seed(0 for random)",
                            )

                    # Button to generate the image
                    controlnet_inpaint_predict_button = gr.Button(
                        value="Generate Image"
                    )

                with gr.Column():
                    # Gallery to display the generated images
                    controlnet_inpaint_output_image = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=(1, 2))

        controlnet_inpaint_predict_button.click(
            fn=StableDiffusionControlNetInpaintGenerator().generate_image,
            inputs=[
                controlnet_inpaint_image_path,
                controlnet_inpaint_stable_model_path,
                controlnet_inpaint_model_path,
                controlnet_inpaint_prompt,
                controlnet_inpaint_negative_prompt,
                controlnet_inpaint_num_images_per_prompt,
                controlnet_inpaint_height,
                controlnet_inpaint_width,
                controlnet_inpaint_strength,
                controlnet_inpaint_guess_mode,
                controlnet_inpaint_guidance_scale,
                controlnet_inpaint_num_inference_step,
                controlnet_inpaint_conditioning_scale,
                controlnet_inpaint_scheduler,
                controlnet_inpaint_seed_generator,
                controlnet_inpaint_preprocces_type,
            ],
            outputs=[controlnet_inpaint_output_image],
        )
