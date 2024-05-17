# {% include 'template/license_header' %}

import gc

import torch
from zenml.logger import get_logger

logger = get_logger(__name__)


def cleanup_memory() -> None:
    """Clean up GPU memory."""
    logger.info("Cleaning up GPU memory on the machine...")
    while gc.collect():
        torch.cuda.empty_cache()
