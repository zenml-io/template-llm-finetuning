# {% include 'template/license_header' %}

import os
from pathlib import Path
from typing import Optional

from zenml.client import Client

from scripts.convert_hf_checkpoint import convert_hf_checkpoint


def get_huggingface_access_token() -> Optional[str]:
    """Get access token for huggingface.

    Returns:
        The access token if one was found.
    """
    try:
        return (
            Client()
            .get_secret("huggingface_credentials")
            .secret_values["token"]
        )
    except KeyError:
        return os.getenv("HF_TOKEN")


def convert_to_lit_checkpoint_if_necessary(checkpoint_dir: Path) -> None:
    """Convert an HF checkpoint to a lit checkpoint if necessary.

    Args:
        checkpoint_dir: The directory of the HF checkpoint.
    """
    lit_model_path = checkpoint_dir / "lit_model.pth"

    if lit_model_path.is_file():
        return

    convert_hf_checkpoint(checkpoint_dir=checkpoint_dir)
