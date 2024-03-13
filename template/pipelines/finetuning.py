# {% include 'template/license_header' %}

from typing import Optional

from steps import finetune
from zenml import get_pipeline_context, pipeline
from zenml.config import DockerSettings


@pipeline(
    settings={
        "docker": DockerSettings(
            apt_packages=["git"], requirements="requirements.txt"
        )
    }
)
def {{product_name}}_finetuning(
    dataset_artifact_name: Optional[str] = None,
    dataset_artifact_version: Optional[str] = None,
) -> None:
    """Pipeline to finetune LLMs using LoRA."""
    dataset_directory = None
    if dataset_artifact_name:
        dataset_directory = get_pipeline_context().model.get_artifact(
            name=dataset_artifact_name, version=dataset_artifact_version
        )

    finetune(dataset_directory=dataset_directory)
