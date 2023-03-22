import gradio as gr
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

from diffusion_webui.utils.model_list import stable_model_list
from diffusion_webui.utils.scheduler_list import (
    SCHEDULER_LIST,
    get_scheduler_list,
)


class StableDiffusionImage2ImageGenerator:
    def __init__(self):
        self.pipe = None

    def load_model(self, model_path, scheduler):
        if self.pipe is None:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_path, safety_checker=None, torch_dtype=torch.float16
            )

        self.pipe = get_scheduler_list(pipe=self.pipe, scheduler=scheduler)
        self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        return self.pipe

    def generate_image(
        self,
        image_path: str,
        model_path: str,
        prompt: str,
        negative_prompt: str,
        num_images_per_prompt: int,
        scheduler: str,
        guidance_scale: int,
        num_inference_step: int,
        seed_generator=0,
    ):
        pipe = self.load_model(
            model_path=model_path,
            scheduler=scheduler,
        )

        if seed_generator == 0:
            random_seed = torch.randint(0, 1000000, (1,))
            generator = torch.manual_seed(random_seed)
        else:
            generator = torch.manual_seed(seed_generator)

        image = Image.open(image_path)
        images = pipe(
            prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_step,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images

        return images

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    image2image_image_file = gr.Image(
                        type="filepath", label="Image"
                    ).style(height=260)

                    image2image_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Prompt",
                        show_label=False,
                    )

                    image2image_negative_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Negative Prompt",
                        show_label=False,
                    )

                    with gr.Row():
                        with gr.Column():
                            image2image_model_path = gr.Dropdown(
                                choices=stable_model_list,
                                value=stable_model_list[0],
                                label="Stable Model Id",
                            )

                            image2image_guidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=7.5,
                                label="Guidance Scale",
                            )
                            image2image_num_inference_step = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=50,
                                label="Num Inference Step",
                            )
                        with gr.Row():
                            with gr.Column():
                                image2image_scheduler = gr.Dropdown(
                                    choices=SCHEDULER_LIST,
                                    value=SCHEDULER_LIST[0],
                                    label="Scheduler",
                                )
                                image2image_num_images_per_prompt = gr.Slider(
                                    minimum=1,
                                    maximum=30,
                                    step=1,
                                    value=1,
                                    label="Number Of Images",
                                )

                                image2image_seed_generator = gr.Slider(
                                    minimum=0,
                                    maximum=1000000,
                                    step=1,
                                    value=0,
                                    label="Seed(0 for random)",
                                )

                    image2image_predict_button = gr.Button(value="Generator")

                with gr.Column():
                    output_image = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=(1, 2))

        image2image_predict_button.click(
            fn=StableDiffusionImage2ImageGenerator().generate_image,
            inputs=[
                image2image_image_file,
                image2image_model_path,
                image2image_prompt,
                image2image_negative_prompt,
                image2image_num_images_per_prompt,
                image2image_scheduler,
                image2image_guidance_scale,
                image2image_num_inference_step,
                image2image_seed_generator,
            ],
            outputs=[output_image],
        )
