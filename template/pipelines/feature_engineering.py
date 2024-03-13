# {% include 'template/license_header' %}

from steps import feature_engineering
from zenml import pipeline
from zenml.config import DockerSettings


@pipeline(
    settings={
        "docker": DockerSettings(
            apt_packages=["git"], requirements="requirements.txt"
        )
    }
)
def {{product_name}}_feature_engineering() -> None:
    """Feature engineering pipeline."""
    feature_engineering()
