# {% include 'template/license_header' %}

import os
from typing import Optional

import click


@click.command(
    help="""
ZenML LLM Finetuning project CLI v0.2.0.

Run the ZenML LLM Finetuning project LLM PEFT finetuning pipelines.

Examples:

  \b
  # Run the pipeline
    python run.py
  
  \b
  # Run the pipeline with custom config
    python run.py --config custom_finetune.yaml
"""
)
@click.option(
    "--config",
    type=str,
    default="default_finetune.yaml",
    help="Path to the YAML config file.",
)
@click.option(
    "--accelerate",
    is_flag=True,
    default=False,
    help="Run the pipeline with Accelerate.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
def main(
    config: Optional[str] = None,
    accelerate: bool = False,
    no_cache: bool = False,
):
    """Main entry point for the pipeline execution.

    Args:
        config: Path to the YAML config file.
        accelerate: If `True` Accelerate will be used.
        no_cache: If `True` cache will be disabled.
    """
    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )
    pipeline_args = {"enable_cache": not no_cache}
    if not config:
        raise RuntimeError("Config file is required to run a pipeline.")

    pipeline_args["config_path"] = os.path.join(config_folder, config)

    if accelerate:
        from pipelines.train_accelerated import {{ product_name.replace("-","_") }}_full_finetune

        {{ product_name.replace("-","_") }}_full_finetune.with_options(**pipeline_args)()
    else:
        from pipelines.train import {{ product_name.replace("-","_") }}_full_finetune

        {{ product_name.replace("-","_") }}_full_finetune.with_options(**pipeline_args)()


if __name__ == "__main__":
    main()
