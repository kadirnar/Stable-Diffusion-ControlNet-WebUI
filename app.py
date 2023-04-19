import gradio as gr

from diffusion_webui import (
    CodeformerUpscalerGenerator,
    StableDiffusionControlInpaintNetDepthGenerator,
    StableDiffusionControlNetCannyGenerator,
    StableDiffusionControlNetDepthGenerator,
    StableDiffusionControlNetHEDGenerator,
    StableDiffusionControlNetInpaintCannyGenerator,
    StableDiffusionControlNetInpaintHedGenerator,
    StableDiffusionControlNetInpaintMlsdGenerator,
    StableDiffusionControlNetInpaintPoseGenerator,
    StableDiffusionControlNetInpaintScribbleGenerator,
    StableDiffusionControlNetInpaintSegGenerator,
    StableDiffusionControlNetLineArtAnimeGenerator,
    StableDiffusionControlNetLineArtGenerator,
    StableDiffusionControlNetMLSDGenerator,
    StableDiffusionControlNetNormalGenerator,
    StableDiffusionControlNetPix2PixGenerator,
    StableDiffusionControlNetPoseGenerator,
    StableDiffusionControlNetScribbleGenerator,
    StableDiffusionControlNetSegGenerator,
    StableDiffusionControlNetShuffleGenerator,
    StableDiffusionControlNetSoftEdgeGenerator,
    StableDiffusionImage2ImageGenerator,
    StableDiffusionInpaintGenerator,
    StableDiffusionText2ImageGenerator,
)


def diffusion_app():
    app = gr.Blocks()
    with app:
        with gr.Row():
            with gr.Column():
                with gr.Tab("Text2Img"):
                    StableDiffusionText2ImageGenerator.app()
                with gr.Tab("Img2Img"):
                    StableDiffusionImage2ImageGenerator.app()
                with gr.Tab("Inpaint"):
                    StableDiffusionInpaintGenerator.app()
                with gr.Tab("ControlNet"):
                    with gr.Tab("Canny"):
                        StableDiffusionControlNetCannyGenerator.app()
                    with gr.Tab("Depth"):
                        StableDiffusionControlNetDepthGenerator.app()
                    with gr.Tab("HED"):
                        StableDiffusionControlNetHEDGenerator.app()
                    with gr.Tab("MLSD"):
                        StableDiffusionControlNetMLSDGenerator.app()
                    with gr.Tab("Pose"):
                        StableDiffusionControlNetPoseGenerator.app()
                    with gr.Tab("Scribble"):
                        StableDiffusionControlNetScribbleGenerator.app()
                    with gr.Tab("Normal"):
                        StableDiffusionControlNetNormalGenerator.app()
                    with gr.Tab("Seg"):
                        StableDiffusionControlNetSegGenerator.app()
                    with gr.Tab("Shuffle"):
                        StableDiffusionControlNetShuffleGenerator.app()
                    with gr.Tab("Pix2Pix"):
                        StableDiffusionControlNetPix2PixGenerator.app()
                    with gr.Tab("LineArt"):
                        StableDiffusionControlNetLineArtGenerator.app()
                    with gr.Tab("LineArtAnime"):
                        StableDiffusionControlNetLineArtAnimeGenerator.app()
                    with gr.Tab("SoftEdge"):
                        StableDiffusionControlNetSoftEdgeGenerator.app()
                with gr.Tab("ControlNet Inpaint"):
                    with gr.Tab("Canny"):
                        StableDiffusionControlNetInpaintCannyGenerator.app()
                    with gr.Tab("Depth"):
                        StableDiffusionControlInpaintNetDepthGenerator.app()
                    with gr.Tab("HED"):
                        StableDiffusionControlNetInpaintHedGenerator.app()
                    with gr.Tab("MLSD"):
                        StableDiffusionControlNetInpaintMlsdGenerator.app()
                    with gr.Tab("Pose"):
                        StableDiffusionControlNetInpaintPoseGenerator.app()
                    with gr.Tab("Scribble"):
                        StableDiffusionControlNetInpaintScribbleGenerator.app()
                    with gr.Tab("Seg"):
                        StableDiffusionControlNetInpaintSegGenerator.app()
                with gr.Tab("Upscaler"):
                    CodeformerUpscalerGenerator.app()

    app.queue(concurrency_count=2)
    app.launch(debug=True, enable_queue=True)


if __name__ == "__main__":
    diffusion_app()
