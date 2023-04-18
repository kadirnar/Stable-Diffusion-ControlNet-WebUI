from diffusion_webui.diffusion_models.controlnet import (
    StableDiffusionControlNetCannyGenerator,
    StableDiffusionControlNetDepthGenerator,
    StableDiffusionControlNetHEDGenerator,
    StableDiffusionControlNetMLSDGenerator,
    StableDiffusionControlNetNormalGenerator,
    StableDiffusionControlNetPoseGenerator,
    StableDiffusionControlNetScribbleGenerator,
    StableDiffusionControlNetSegGenerator,
)
from diffusion_webui.diffusion_models.controlnet.controlnet_inpaint import (
    StableDiffusionControlInpaintNetDepthGenerator,
    StableDiffusionControlNetInpaintCannyGenerator,
    StableDiffusionControlNetInpaintHedGenerator,
    StableDiffusionControlNetInpaintMlsdGenerator,
    StableDiffusionControlNetInpaintPoseGenerator,
    StableDiffusionControlNetInpaintScribbleGenerator,
    StableDiffusionControlNetInpaintSegGenerator,
)
from diffusion_webui.diffusion_models.stable_diffusion import (
    StableDiffusionImage2ImageGenerator,
    StableDiffusionInpaintGenerator,
    StableDiffusionText2ImageGenerator,
)
from diffusion_webui.upscaler_models import CodeformerUpscalerGenerator

__version__ = "2.3.0"
