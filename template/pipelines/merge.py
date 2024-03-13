# {% include 'template/license_header' %}

from steps import merge
from zenml import pipeline
from zenml.config import DockerSettings


@pipeline(
    settings={
        "docker": DockerSettings(
            apt_packages=["git"], requirements="requirements.txt"
        )
    }
)
def {{product_name}}_merging() -> None:
    """Pipeline to merge LLMs with adapters."""
    merge()
