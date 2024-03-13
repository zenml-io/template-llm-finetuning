# {% include 'template/license_header' %}

from pydantic import BaseModel


class LoraParameters(BaseModel):
    """Lora specific parameters."""

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_query: bool = True
    lora_key: bool = False
    lora_value: bool = True
    lora_projection: bool = False
    lora_mlp: bool = False
    lora_head: bool = False
