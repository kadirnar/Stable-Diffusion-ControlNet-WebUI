import gradio as gr
import torch
from controlnet_aux import MLSDdetector
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image

from diffusion_webui.utils.model_list import stable_model_list
from diffusion_webui.utils.scheduler_list import (
    SCHEDULER_LIST,
    get_scheduler_list,
)


class StableDiffusionControlNetMLSDGenerator:
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

        self.pipe = get_scheduler_list(pipe=self.pipe, scheduler=scheduler)
        self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        return self.pipe

    def controlnet_mlsd(self, image_path: str):
        mlsd = MLSDdetector.from_pretrained("lllyasviel/ControlNet")

        image = Image.open(image_path)
        image = mlsd(image)

        return image

    def generate_image(
        self,
        image_path: str,
        model_path: str,
        prompt: str,
        negative_prompt: str,
        num_images_per_prompt: int,
        guidance_scale: int,
        num_inference_step: int,
        scheduler: str,
        seed_generator: int,
    ):
        image = self.controlnet_mlsd(image_path=image_path)

        pipe = self.load_model(
            stable_model_path=model_path,
            controlnet_model_path="lllyasviel/sd-controlnet-mlsd",
            scheduler=scheduler,
        )

        if seed_generator == 0:
            random_seed = torch.randint(0, 1000000, (1,))
            generator = torch.manual_seed(random_seed)
        else:
            generator = torch.manual_seed(seed_generator)

        output = pipe(
            prompt=prompt,
            image=image,
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
                    controlnet_mlsd_image_file = gr.Image(
                        type="filepath", label="Image"
                    )

                    controlnet_mlsd_prompt = gr.Textbox(
                        lines=1,
                        show_label=False,
                        placeholder="Prompt",
                    )

                    controlnet_mlsd_negative_prompt = gr.Textbox(
                        lines=1,
                        show_label=False,
                        placeholder="Negative Prompt",
                    )

                    with gr.Row():
                        with gr.Column():
                            controlnet_mlsd_model_id = gr.Dropdown(
                                choices=stable_model_list,
                                value=stable_model_list[0],
                                label="Stable Model Id",
                            )
                            controlnet_mlsd_guidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=7.5,
                                label="Guidance Scale",
                            )
                            controlnet_mlsd_num_inference_step = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=50,
                                label="Num Inference Step",
                            )

                        with gr.Row():
                            with gr.Column():
                                controlnet_mlsd_scheduler = gr.Dropdown(
                                    choices=SCHEDULER_LIST,
                                    value=SCHEDULER_LIST[0],
                                    label="Scheduler",
                                )

                                controlnet_mlsd_seed_generator = gr.Slider(
                                    minimum=0,
                                    maximum=1000000,
                                    step=1,
                                    value=0,
                                    label="Seed Generator",
                                )
                                controlnet_mlsd_num_images_per_prompt = (
                                    gr.Slider(
                                        minimum=1,
                                        maximum=10,
                                        step=1,
                                        value=1,
                                        label="Number Of Images",
                                    )
                                )

                    controlnet_mlsd_predict = gr.Button(value="Generator")

                with gr.Column():
                    output_image = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=(1, 2))

            controlnet_mlsd_predict.click(
                fn=StableDiffusionControlNetMLSDGenerator().generate_image,
                inputs=[
                    controlnet_mlsd_image_file,
                    controlnet_mlsd_model_id,
                    controlnet_mlsd_prompt,
                    controlnet_mlsd_negative_prompt,
                    controlnet_mlsd_num_images_per_prompt,
                    controlnet_mlsd_guidance_scale,
                    controlnet_mlsd_num_inference_step,
                    controlnet_mlsd_scheduler,
                    controlnet_mlsd_seed_generator,
                ],
                outputs=output_image,
            )
