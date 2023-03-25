import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

from diffusion_webui.utils.model_list import stable_model_list
from diffusion_webui.utils.scheduler_list import (
    SCHEDULER_LIST,
    get_scheduler_list,
)


class StableDiffusionText2ImageGenerator:
    def __init__(self):
        self.pipe = None

    def load_model(
        self,
        model_path,
        scheduler,
    ):
        if self.pipe is None:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_path, safety_checker=None, torch_dtype=torch.float16
            )

        self.pipe = get_scheduler_list(pipe=self.pipe, scheduler=scheduler)
        self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        return self.pipe

    def generate_image(
        self,
        model_path: str,
        prompt: str,
        negative_prompt: str,
        num_images_per_prompt: int,
        scheduler: str,
        guidance_scale: int,
        num_inference_step: int,
        height: int,
        width: int,
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

        images = pipe(
            prompt=prompt,
            height=height,
            width=width,
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
                    text2image_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Prompt",
                        show_label=False,
                    )

                    text2image_negative_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Negative Prompt",
                        show_label=False,
                    )
                    with gr.Row():
                        with gr.Column():
                            text2image_model_path = gr.Dropdown(
                                choices=stable_model_list,
                                value=stable_model_list[0],
                                label="Text-Image Model Id",
                            )

                            text2image_guidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=7.5,
                                label="Guidance Scale",
                            )

                            text2image_num_inference_step = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=50,
                                label="Num Inference Step",
                            )
                            text2image_num_images_per_prompt = gr.Slider(
                                minimum=1,
                                maximum=30,
                                step=1,
                                value=1,
                                label="Number Of Images",
                            )
                        with gr.Row():
                            with gr.Column():

                                text2image_scheduler = gr.Dropdown(
                                    choices=SCHEDULER_LIST,
                                    value=SCHEDULER_LIST[0],
                                    label="Scheduler",
                                )

                                text2image_height = gr.Slider(
                                    minimum=128,
                                    maximum=1280,
                                    step=32,
                                    value=512,
                                    label="Image Height",
                                )

                                text2image_width = gr.Slider(
                                    minimum=128,
                                    maximum=1280,
                                    step=32,
                                    value=512,
                                    label="Image Width",
                                )
                                text2image_seed_generator = gr.Slider(
                                    label="Seed(0 for random)",
                                    minimum=0,
                                    maximum=1000000,
                                    value=0,
                                )
                    text2image_predict = gr.Button(value="Generator")

                with gr.Column():
                    output_image = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=(1, 2), height=200)

            text2image_predict.click(
                fn=StableDiffusionText2ImageGenerator().generate_image,
                inputs=[
                    text2image_model_path,
                    text2image_prompt,
                    text2image_negative_prompt,
                    text2image_num_images_per_prompt,
                    text2image_scheduler,
                    text2image_guidance_scale,
                    text2image_num_inference_step,
                    text2image_height,
                    text2image_width,
                    text2image_seed_generator,
                ],
                outputs=output_image,
            )
