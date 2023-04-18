from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from transformers import pipeline
from PIL import Image
import gradio as gr
import numpy as np
import torch
import cv2


from diffusion_webui.utils.model_list import (
    controlnet_normal_model_list,
    stable_model_list,
)
from diffusion_webui.utils.scheduler_list import (
    SCHEDULER_LIST,
    get_scheduler_list,
)


class StableDiffusionControlNetNormalGenerator:
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

    def controlnet_normal(
        self,
        image_path: str,
    ):
        image = load_image(image_path).convert("RGB")
        depth_estimator = pipeline("depth-estimation", model ="Intel/dpt-hybrid-midas" )
        image = depth_estimator(image)['predicted_depth'][0]
        image = image.numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)
        bg_threhold = 0.4
        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < bg_threhold] = 0
        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < bg_threhold] = 0
        z = np.ones_like(x) * np.pi * 2.0
        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        return image

    def generate_image(
        self,
        image_path: str,
        stable_model_path: str,
        controlnet_model_path: str,
        prompt: str,
        negative_prompt: str,
        num_images_per_prompt: int,
        guidance_scale: int,
        num_inference_step: int,
        scheduler: str,
        seed_generator: int,
    ):
        pipe = self.load_model(stable_model_path, controlnet_model_path, scheduler)
        image = self.controlnet_normal(image_path)

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
                    controlnet_normal_image_file = gr.Image(
                        type="filepath", label="Image"
                    )

                    controlnet_normal_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Prompt",
                        show_label=False,
                    )

                    controlnet_normal_negative_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Negative Prompt",
                        show_label=False,
                    )
                    with gr.Row():
                        with gr.Column():
                            controlnet_normal_stable_model_id = gr.Dropdown(
                                choices=stable_model_list,
                                value=stable_model_list[0],
                                label="Stable Model Id",
                            )

                            controlnet_normal_guidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=7.5,
                                label="Guidance Scale",
                            )
                            controlnet_normal_num_inference_step = gr.Slider(
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=50,
                                label="Num Inference Step",
                            )
                            controlnet_normal_num_images_per_prompt = gr.Slider(
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=1,
                                label="Number Of Images",
                            )
                        with gr.Row():
                            with gr.Column():
                                controlnet_normal_model_id = gr.Dropdown(
                                    choices=controlnet_normal_model_list,
                                    value=controlnet_normal_model_list[0],
                                    label="ControlNet Model Id",
                                )

                                controlnet_normal_scheduler = gr.Dropdown(
                                    choices=SCHEDULER_LIST,
                                    value=SCHEDULER_LIST[0],
                                    label="Scheduler",
                                )

                                controlnet_normal_seed_generator = gr.Number(
                                    value=0,
                                    label="Seed Generator",
                                )
                    controlnet_normal_predict = gr.Button(value="Generator")

                with gr.Column():
                    output_image = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=(1, 2))

            controlnet_normal_predict.click(
                fn=StableDiffusionControlNetCannyGenerator().generate_image,
                inputs=[
                    controlnet_normal_image_file,
                    controlnet_normal_stable_model_id,
                    controlnet_normal_model_id,
                    controlnet_normal_prompt,
                    controlnet_normal_negative_prompt,
                    controlnet_normal_num_images_per_prompt,
                    controlnet_normal_guidance_scale,
                    controlnet_normal_num_inference_step,
                    controlnet_normal_scheduler,
                    controlnet_normal_seed_generator,
                ],
                outputs=[output_image],
            )
