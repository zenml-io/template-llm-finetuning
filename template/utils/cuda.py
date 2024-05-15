# {% include 'template/license_header' %}

import gc

import torch


def cleanup_memory() -> None:
    """Clean up GPU memory."""
    while gc.collect():
        torch.cuda.empty_cache()
