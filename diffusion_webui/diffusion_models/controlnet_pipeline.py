import gradio as gr
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
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


class StableDiffusionControlNetGenerator(ControlnetPipeline):
    def __init__(self):
        self.pipe = None

    def load_model(self, stable_model_path, controlnet_model_path, scheduler):
        if self.pipe is None:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_path, torch_dtype=torch.float16
            )
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                pretrained_model_name_or_path=stable_model_path,
                controlnet=controlnet,
                safety_checker=None,
                torch_dtype=torch.float16,
            )

        self.pipe = get_scheduler(pipe=self.pipe, scheduler=scheduler)
        self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        return self.pipe

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
        height: int,
        width: int,
        guess_mode: bool,
        controlnet_conditioning_scale: int,
        prompt: str,
        negative_prompt: str,
        num_images_per_prompt: int,
        guidance_scale: int,
        num_inference_step: int,
        scheduler: str,
        seed_generator: int,
        preprocces_type: str,
    ):
        pipe = self.load_model(
            stable_model_path=stable_model_path,
            controlnet_model_path=controlnet_model_path,
            scheduler=scheduler,
        )

        read_image = Image.open(image_path)
        controlnet_image = self.controlnet_preprocces(
            read_image=read_image, preprocces_type=preprocces_type
        )

        if seed_generator == 0:
            random_seed = torch.randint(0, 1000000, (1,))
            generator = torch.manual_seed(random_seed)
        else:
            generator = torch.manual_seed(seed_generator)

        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            guess_mode=guess_mode,
            image=controlnet_image,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_step,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images

        return output

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    controlnet_image_path = gr.Image(
                        type="filepath", label="Image"
                    ).style(height=260)
                    controlnet_prompt = gr.Textbox(
                        lines=1, placeholder="Prompt", show_label=False
                    )
                    controlnet_negative_prompt = gr.Textbox(
                        lines=1, placeholder="Negative Prompt", show_label=False
                    )

                    with gr.Row():
                        with gr.Column():
                            controlnet_stable_model_path = gr.Dropdown(
                                choices=stable_model_list,
                                value=stable_model_list[0],
                                label="Stable Model Path",
                            )
                            controlnet_preprocces_type = gr.Dropdown(
                                choices=list(PREPROCCES_DICT.keys()),
                                value=list(PREPROCCES_DICT.keys())[0],
                                label="Preprocess Type",
                            )
                            controlnet_conditioning_scale = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                value=1.0,
                                label="ControlNet Conditioning Scale",
                            )
                            controlnet_guidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=7.5,
                                label="Guidance Scale",
                            )
                            controlnet_height = gr.Slider(
                                minimum=128,
                                maximum=1280,
                                step=32,
                                value=512,
                                label="Height",
                            )
                            controlnet_width = gr.Slider(
                                minimum=128,
                                maximum=1280,
                                step=32,
                                value=512,
                                label="Width",
                            )

                        with gr.Row():
                            with gr.Column():
                                controlnet_model_path = gr.Dropdown(
                                    choices=controlnet_model_list,
                                    value=controlnet_model_list[0],
                                    label="ControlNet Model Path",
                                )
                                controlnet_scheduler = gr.Dropdown(
                                    choices=list(SCHEDULER_MAPPING.keys()),
                                    value=list(SCHEDULER_MAPPING.keys())[0],
                                    label="Scheduler",
                                )
                                controlnet_num_inference_step = gr.Slider(
                                    minimum=1,
                                    maximum=150,
                                    step=1,
                                    value=30,
                                    label="Num Inference Step",
                                )

                                controlnet_num_images_per_prompt = gr.Slider(
                                    minimum=1,
                                    maximum=4,
                                    step=1,
                                    value=1,
                                    label="Number Of Images",
                                )
                                controlnet_seed_generator = gr.Slider(
                                    minimum=0,
                                    maximum=1000000,
                                    step=1,
                                    value=0,
                                    label="Seed(0 for random)",
                                )
                                controlnet_guess_mode = gr.Checkbox(
                                    label="Guess Mode"
                                )

                    # Button to generate the image
                    predict_button = gr.Button(value="Generate Image")

                with gr.Column():
                    # Gallery to display the generated images
                    output_image = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=(1, 2))

        predict_button.click(
            fn=StableDiffusionControlNetGenerator().generate_image,
            inputs=[
                controlnet_image_path,
                controlnet_stable_model_path,
                controlnet_model_path,
                controlnet_height,
                controlnet_width,
                controlnet_guess_mode,
                controlnet_conditioning_scale,
                controlnet_prompt,
                controlnet_negative_prompt,
                controlnet_num_images_per_prompt,
                controlnet_guidance_scale,
                controlnet_num_inference_step,
                controlnet_scheduler,
                controlnet_seed_generator,
                controlnet_preprocces_type,
            ],
            outputs=[output_image],
        )
