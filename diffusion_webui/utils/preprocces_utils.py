from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector


PREPROCCES_DICT = {
    "Hed": HEDdetector.from_pretrained("lllyasviel/Annotators"),
    "Midas": MidasDetector.from_pretrained("lllyasviel/Annotators"),
    "MLSD": MLSDdetector.from_pretrained("lllyasviel/Annotators"),
    "Openpose": OpenposeDetector.from_pretrained("lllyasviel/Annotators"),
    "PidiNet": PidiNetDetector.from_pretrained("lllyasviel/Annotators"),
    "NormalBae": NormalBaeDetector.from_pretrained("lllyasviel/Annotators"),
    "Lineart": LineartDetector.from_pretrained("lllyasviel/Annotators"),
    "LineartAnime": LineartAnimeDetector.from_pretrained("lllyasviel/Annotators"),
    "Zoe": ZoeDetector.from_pretrained("lllyasviel/Annotators"),
    "Sam": SamDetector.from_pretrained("./weight_path"),
    "Canny": CannyDetector(),
    "ContentShuffle": ContentShuffleDetector(),
    "MediapipeFace": MediapipeFaceDetector(),
}
