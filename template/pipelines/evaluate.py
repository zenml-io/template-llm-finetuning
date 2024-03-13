# {% include 'template/license_header' %}

from steps import evaluate
from zenml import pipeline
from zenml.config import DockerSettings


@pipeline(
    settings={
        "docker": DockerSettings(
            apt_packages=["git"], requirements="requirements.txt"
        )
    }
)
def {{product_name}}_evaluation() -> None:
    """Pipeline to evaluate a LoRA fine-tuned LLM."""
    evaluate()
