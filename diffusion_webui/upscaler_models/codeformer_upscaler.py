import gradio as gr
from codeformer.app import inference_app


class CodeformerUpscalerGenerator:
    def generate_image(
        self,
        image_path: str,
        background_enhance: bool,
        face_upsample: bool,
        upscale: int,
        codeformer_fidelity: int,
    ):

        pipe = inference_app(
            image=image_path,
            background_enhance=background_enhance,
            face_upsample=face_upsample,
            upscale=upscale,
            codeformer_fidelity=codeformer_fidelity,
        )

        return [pipe]

    def app():
        with gr.Blocks():
            with gr.Row():
                with gr.Column():
                    codeformer_upscale_image_file = gr.Image(
                        type="filepath", label="Image"
                    ).style(height=260)

                    with gr.Row():
                        with gr.Column():
                            codeformer_face_upsample = gr.Checkbox(
                                label="Face Upsample",
                                value=True,
                            )
                            codeformer_upscale = gr.Slider(
                                label="Upscale",
                                minimum=1,
                                maximum=4,
                                step=1,
                                value=2,
                            )
                        with gr.Row():
                            with gr.Column():
                                codeformer_background_enhance = gr.Checkbox(
                                    label="Background Enhance",
                                    value=True,
                                )
                                codeformer_upscale_fidelity = gr.Slider(
                                    label="Codeformer Fidelity",
                                    minimum=0.1,
                                    maximum=1.0,
                                    step=0.1,
                                    value=0.5,
                                )

                    codeformer_upscale_predict_button = gr.Button(
                        value="Generator"
                    )

                with gr.Column():
                    output_image = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        elem_id="gallery",
                    ).style(grid=(1, 2))

        codeformer_upscale_predict_button.click(
            fn=CodeformerUpscalerGenerator().generate_image,
            inputs=[
                codeformer_upscale_image_file,
                codeformer_background_enhance,
                codeformer_face_upsample,
                codeformer_upscale,
                codeformer_upscale_fidelity,
            ],
            outputs=[output_image],
        )
