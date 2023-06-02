from controlnet_aux import (
    CannyDetector,
    ContentShuffleDetector,
    HEDdetector,
    LineartAnimeDetector,
    LineartDetector,
    MediapipeFaceDetector,
    MidasDetector,
    MLSDdetector,
    NormalBaeDetector,
    OpenposeDetector,
    PidiNetDetector,
    SamDetector,
    ZoeDetector,
)

PREPROCCES_DICT = {
    "Hed": HEDdetector.from_pretrained("lllyasviel/Annotators"),
    "Midas": MidasDetector.from_pretrained("lllyasviel/Annotators"),
    "MLSD": MLSDdetector.from_pretrained("lllyasviel/Annotators"),
    "Openpose": OpenposeDetector.from_pretrained("lllyasviel/Annotators"),
    "PidiNet": PidiNetDetector.from_pretrained("lllyasviel/Annotators"),
    "NormalBae": NormalBaeDetector.from_pretrained("lllyasviel/Annotators"),
    "Lineart": LineartDetector.from_pretrained("lllyasviel/Annotators"),
    "LineartAnime": LineartAnimeDetector.from_pretrained(
        "lllyasviel/Annotators"
    ),
    "Zoe": ZoeDetector.from_pretrained("lllyasviel/Annotators"),
    "Canny": CannyDetector(),
    "ContentShuffle": ContentShuffleDetector(),
    "MediapipeFace": MediapipeFaceDetector(),
}
