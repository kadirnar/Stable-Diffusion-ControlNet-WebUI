import gradio as gr
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from diffusion_webui.utils.model_list import (
    controlnet_seg_model_list,
    stable_inpiant_model_list,
)
from diffusion_webui.utils.scheduler_list import (
    SCHEDULER_LIST,
    get_scheduler_list,
)

# https://github.com/mikonvergence/ControlNetInpaint


def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]


class StableDiffusionControlNetInpaintSegGenerator:
    def __init__(self):
        self.pipe = None

    def load_model(
        self,
        stable_model_path,
        controlnet_model_path,
        scheduler,
    ):

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

    def load_image(self, image_path):
        image = np.array(image_path)
        image = Image.fromarray(image)
        return image

    def controlnet_seg_inpaint(self, image_path: str):
        image_processor = AutoImageProcessor.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )
        image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )

        image = image_path["image"].convert("RGB").resize((512, 512))
        image = np.array(image)
        pixel_values = image_processor(image, return_tensors="pt").pixel_values

        with torch.no_grad():
            outputs = image_segmentor(pixel_values)

        seg = image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(ade_palette())

        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color

        color_seg = color_seg.astype(np.uint8)
        image = Image.fromarray(color_seg)

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
        controlnet_conditioning_scale: int,
        scheduler: str,
        seed_generator: int,
    ):

        normal_image = image_path["image"].convert("RGB").resize((512, 512))
        mask_image = image_path["mask"].convert("RGB").resize((512, 512))

        normal_image = self.load_image(image_path=normal_image)
        mask_image = self.load_image(image_path=mask_image)

        controlnet_image = self.controlnet_seg_inpaint(image_path=image_path)

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
            mask_image=mask_image,
            control_image=controlnet_image,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_step,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        ).images

        return output

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    controlnet_seg_inpaint_image_file = gr.Image(
                        source="upload",
                        tool="sketch",
                        elem_id="image_upload",
                        type="pil",
                        label="Upload",
                    )

                    controlnet_seg_inpaint_prompt = gr.Textbox(
                        lines=1, placeholder="Prompt", show_label=False
                    )

                    controlnet_seg_inpaint_negative_prompt = gr.Textbox(
                        lines=1,
                        show_label=False,
                        placeholder="Negative Prompt",
                    )
                    with gr.Row():
                        with gr.Column():
                            controlnet_seg_inpaint_stable_model_id = (
                                gr.Dropdown(
                                    choices=stable_inpiant_model_list,
                                    value=stable_inpiant_model_list[0],
                                    label="Stable Model Id",
                                )
                            )

                            controlnet_seg_inpaint_guidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=7.5,
                                label="Guidance Scale",
                            )

                            controlnet_seg_inpaint_num_inference_step = (
                                gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=50,
                                    label="Num Inference Step",
                                )
                            )
                            controlnet_seg_inpaint_num_images_per_prompt = (
                                gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    step=1,
                                    value=1,
                                    label="Number Of Images",
                                )
                            )
                        with gr.Row():
                            with gr.Column():
                                controlnet_seg_inpaint_model_id = gr.Dropdown(
                                    choices=controlnet_seg_model_list,
                                    value=controlnet_seg_model_list[0],
                                    label="Controlnet Model Id",
                                )
                                controlnet_seg_inpaint_scheduler = gr.Dropdown(
                                    choices=SCHEDULER_LIST,
                                    value=SCHEDULER_LIST[0],
                                    label="Scheduler",
                                )
                                controlnet_seg_inpaint_controlnet_conditioning_scale = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    step=0.1,
                                    value=0.5,
                                    label="Controlnet Conditioning Scale",
                                )

                                controlnet_seg_inpaint_seed_generator = (
                                    gr.Slider(
                                        minimum=0,
                                        maximum=1000000,
                                        step=1,
                                        value=0,
                                        label="Seed Generator",
                                    )
                                )

                    controlnet_seg_inpaint_predict = gr.Button(
                        value="Generator"
                    )

                with gr.Column():
                    output_image = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=(1, 2))

            controlnet_seg_inpaint_predict.click(
                fn=StableDiffusionControlNetInpaintSegGenerator().generate_image,
                inputs=[
                    controlnet_seg_inpaint_image_file,
                    controlnet_seg_inpaint_stable_model_id,
                    controlnet_seg_inpaint_model_id,
                    controlnet_seg_inpaint_prompt,
                    controlnet_seg_inpaint_negative_prompt,
                    controlnet_seg_inpaint_num_images_per_prompt,
                    controlnet_seg_inpaint_guidance_scale,
                    controlnet_seg_inpaint_num_inference_step,
                    controlnet_seg_inpaint_controlnet_conditioning_scale,
                    controlnet_seg_inpaint_scheduler,
                    controlnet_seg_inpaint_seed_generator,
                ],
                outputs=[output_image],
            )
