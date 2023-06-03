import gradio as gr
import torch
from diffusers import DiffusionPipeline

from diffusion_webui.utils.model_list import stable_inpiant_model_list


class StableDiffusionInpaintGenerator:
    def __init__(self):
        self.pipe = None

    def load_model(self, model_path):
        if self.pipe is None:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_path, revision="fp16", torch_dtype=torch.float16
            )

        self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        return self.pipe

    def generate_image(
        self,
        pil_image: str,
        model_path: str,
        prompt: str,
        negative_prompt: str,
        num_images_per_prompt: int,
        guidance_scale: int,
        num_inference_step: int,
        seed_generator=0,
    ):
        image = pil_image["image"].convert("RGB").resize((512, 512))
        mask_image = pil_image["mask"].convert("RGB").resize((512, 512))
        pipe = self.load_model(model_path)

        if seed_generator == 0:
            random_seed = torch.randint(0, 1000000, (1,))
            generator = torch.manual_seed(random_seed)
        else:
            generator = torch.manual_seed(seed_generator)

        output = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
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
                    stable_diffusion_inpaint_image_file = gr.Image(
                        source="upload",
                        tool="sketch",
                        elem_id="image_upload",
                        type="pil",
                        label="Upload",
                    ).style(height=260)

                    stable_diffusion_inpaint_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Prompt",
                        show_label=False,
                    )

                    stable_diffusion_inpaint_negative_prompt = gr.Textbox(
                        lines=1,
                        placeholder="Negative Prompt",
                        show_label=False,
                    )
                    stable_diffusion_inpaint_model_id = gr.Dropdown(
                        choices=stable_inpiant_model_list,
                        value=stable_inpiant_model_list[0],
                        label="Inpaint Model Id",
                    )
                    with gr.Row():
                        with gr.Column():
                            stable_diffusion_inpaint_guidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=7.5,
                                label="Guidance Scale",
                            )

                            stable_diffusion_inpaint_num_inference_step = (
                                gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    value=50,
                                    label="Num Inference Step",
                                )
                            )

                        with gr.Row():
                            with gr.Column():
                                stable_diffusion_inpiant_num_images_per_prompt = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    step=1,
                                    value=1,
                                    label="Number Of Images",
                                )
                                stable_diffusion_inpaint_seed_generator = (
                                    gr.Slider(
                                        minimum=0,
                                        maximum=1000000,
                                        step=1,
                                        value=0,
                                        label="Seed(0 for random)",
                                    )
                                )

                    stable_diffusion_inpaint_predict = gr.Button(
                        value="Generator"
                    )

                with gr.Column():
                    output_image = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=(1, 2))

            stable_diffusion_inpaint_predict.click(
                fn=StableDiffusionInpaintGenerator().generate_image,
                inputs=[
                    stable_diffusion_inpaint_image_file,
                    stable_diffusion_inpaint_model_id,
                    stable_diffusion_inpaint_prompt,
                    stable_diffusion_inpaint_negative_prompt,
                    stable_diffusion_inpiant_num_images_per_prompt,
                    stable_diffusion_inpaint_guidance_scale,
                    stable_diffusion_inpaint_num_inference_step,
                    stable_diffusion_inpaint_seed_generator,
                ],
                outputs=[output_image],
            )
