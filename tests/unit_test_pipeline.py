from steps import prepare_data
from zenml import pipeline


@pipeline
def {{PLACEHOLDER}}_full_finetune(
    system_prompt: str,
    base_model_id: str,
    use_fast: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
):
    """Very simplified pipeline for unit testing only."""
    datasets_dir = prepare_data(
        base_model_id=base_model_id,
        system_prompt=system_prompt,
        use_fast=use_fast,
    )
