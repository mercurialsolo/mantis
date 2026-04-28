from truss.base.truss_config import AcceleratorSpec
from truss_train import (
    CacheConfig,
    CheckpointingConfig,
    Compute,
    Image,
    Runtime,
    TrainingJob,
    TrainingProject,
)


BASE_IMAGE = "pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"

training_runtime = Runtime(
    start_commands=[
        "chmod +x ./run.sh && ./run.sh",
    ],
    environment_variables={
        "EPOCHS": "1",
        "BATCH_SIZE": "1",
        "LORA_RANK": "16",
        "LR": "1e-4",
        "MAX_SAMPLES": "73",
        "DISABLE_UNSLOTH": "true",
        "EXPORT_GGUF": "false",
        "MAX_MEMORY_PER_GPU": "70GiB",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    cache_config=CacheConfig(enabled=True),
    checkpointing_config=CheckpointingConfig(enabled=True),
)

training_compute = Compute(
    accelerator=AcceleratorSpec(accelerator="H100", count=2),
)

training_job = TrainingJob(
    image=Image(base_image=BASE_IMAGE),
    compute=training_compute,
    runtime=training_runtime,
)

training_project = TrainingProject(
    name="mantis-holo3-boattrader-distill",
    job=training_job,
)
