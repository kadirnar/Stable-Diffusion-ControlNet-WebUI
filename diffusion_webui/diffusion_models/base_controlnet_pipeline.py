class ControlnetPipeline:
    def __init__(self):
        self.pipe = None

    def load_model(self, stable_model_path: str, controlnet_model_path: str):
        raise NotImplementedError()

    def load_image(self, image_path: str):
        raise NotImplementedError()

    def controlnet_preprocces(self, read_image: str):
        raise NotImplementedError()

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
        raise NotImplementedError()

    def web_interface():
        raise NotImplementedError()
