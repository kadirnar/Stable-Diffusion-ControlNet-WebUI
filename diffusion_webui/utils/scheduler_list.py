from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepInverseScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KarrasVeScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    RePaintScheduler,
    SchedulerMixin,
    ScoreSdeVeScheduler,
    UnCLIPScheduler,
    UniPCMultistepScheduler,
    VQDiffusionScheduler,
)

SCHEDULER_MAPPING = {
    "DDIM": DDIMScheduler,
    "DDIMInverse": DDIMInverseScheduler,
    "DDPMScheduler": DDPMScheduler,
    "DEISMultistep": DEISMultistepScheduler,
    "DPMSolverMultistepInverse": DPMSolverMultistepInverseScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "DPMSolverSinglestep": DPMSolverSinglestepScheduler,
    "EulerAncestralDiscrete": EulerAncestralDiscreteScheduler,
    "EulerDiscrete": EulerDiscreteScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "IPNDMScheduler": IPNDMScheduler,
    "KarrasVe": KarrasVeScheduler,
    "KDPM2AncestralDiscrete": KDPM2AncestralDiscreteScheduler,
    "KDPM2Discrete": KDPM2DiscreteScheduler,
    "PNDMScheduler": PNDMScheduler,
    "RePaint": RePaintScheduler,
    "ScoreSdeVe": ScoreSdeVeScheduler,
    "UnCLIP": UnCLIPScheduler,
    "UniPCMultistep": UniPCMultistepScheduler,
    "VQDiffusion": VQDiffusionScheduler,
}


def get_scheduler(pipe, scheduler_name):
    if scheduler_name in SCHEDULER_MAPPING:
        SchedulerClass = SCHEDULER_MAPPING[scheduler_name]
        pipe.scheduler = SchedulerClass.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Invalid scheduler name {scheduler_name}")

    return pipe
