# {% include 'template/license_header' %}

import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, upload_folder
from pydantic import BaseModel
from zenml import log_model_metadata, step
from zenml.logger import get_logger

from scripts.convert_lit_checkpoint import convert_lit_checkpoint
from scripts.download import download_from_hub
from scripts.merge_lora import merge_lora
from steps.params import LoraParameters
from steps.utils import (
    convert_to_lit_checkpoint_if_necessary,
    get_huggingface_access_token,
)

logger = get_logger(__file__)


class MergeParameters(BaseModel):
    """Parameters for the merging step."""

    base_model_repo: str
    from_safetensors: bool = False

    adapter_repo: str
    output_repo: str
    convert_to_hf_checkpoint: bool = False

    precision: Optional[str] = None
    lora: LoraParameters = LoraParameters()


@step
def merge(config: MergeParameters) -> None:
    """Merge base model and LoRA adapter.

    Args:
        config: Configuration for this step.
    """
    access_token = get_huggingface_access_token()

    checkpoint_root_dir = Path("checkpoints")
    base_model_dir = checkpoint_root_dir / config.base_model_repo
    adapter_dir = Path("adapters") / config.adapter_repo

    if base_model_dir.exists():
        logger.info(
            "Checkpoint directory already exists, skipping download..."
        )
    else:
        download_from_hub(
            repo_id=config.base_model_repo,
            from_safetensors=config.from_safetensors,
            checkpoint_dir=checkpoint_root_dir,
            access_token=access_token,
        )

    convert_to_lit_checkpoint_if_necessary(checkpoint_dir=base_model_dir)

    snapshot_download(
        config.adapter_repo,
        local_dir=adapter_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=access_token,
    )

    lora_path = adapter_dir / "lit_model_lora_finetuned.pth"
    merged_dir = Path("output/merged")

    merge_lora(
        lora_path=lora_path,
        checkpoint_dir=base_model_dir,
        out_dir=merged_dir,
        precision=config.precision,
        **config.lora.dict()
    )

    for path in Path(base_model_dir).glob("*.json"):
        destination = Path(merged_dir) / path.name
        shutil.copy(src=path, dst=destination)

    if config.convert_to_hf_checkpoint:
        model_name = base_model_dir.name

        output_dir = Path("output/lora_merged_hf") / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        convert_lit_checkpoint(
            checkpoint_path=merged_dir / "lit_model.pth",
            config_path=merged_dir / "lit_config.json",
            output_path=output_dir / "pytorch_model",
        )
    else:
        output_dir = merged_dir

    commit = upload_folder(
        repo_id=config.output_repo, folder_path=output_dir, token=access_token
    )
    log_model_metadata(
        metadata={
            "merged_model_huggingface_commit_hash": commit.oid,
            "merged_model_huggingface_commit_url": commit.commit_url,
        }
    )
