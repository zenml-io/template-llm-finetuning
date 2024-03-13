# {% include 'template/license_header' %}

import os
from typing import Optional

import click
from zenml.logger import get_logger
from pipelines import {{product_name}}_evaluation, {{product_name}}_feature_engineering, {{product_name}}_finetuning, {{product_name}}_merging

logger = get_logger(__name__)


@click.command(
    help="""
{{ project_name }} CLI v{{ version }}.

Run the {{ project_name }} LLM LoRA finetuning pipelines.

Examples:

  \b
  # Run the feature feature engineering pipeline
    python run.py --feature-pipeline
  
  \b
  # Run the finetuning pipeline
    python run.py --finetuning-pipeline

  \b 
  # Run the merging pipeline
    python run.py --merging-pipeline

  \b
  # Run the evaluation pipeline
    python run.py --eval-pipeline
"""
)
@click.option(
    "--config",
    type=str,
    default=None,
    help="Path to the YAML config file.",
)
@click.option(
    "--feature-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that creates the dataset.",
)
@click.option(
    "--finetuning-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that finetunes the model.",
)
@click.option(
    "--merging-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that merges the model and adapter.",
)
@click.option(
    "--eval-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that evaluates the model.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
def main(
    config: Optional[str] = None,
    feature_pipeline: bool = False,
    finetuning_pipeline: bool = False,
    merging_pipeline: bool = False,
    eval_pipeline: bool = False,
    no_cache: bool = False,
):
    """Main entry point for the pipeline execution.

    Args:
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

    if feature_pipeline:
        {{product_name}}_feature_engineering.with_options(**pipeline_args)()

    if finetuning_pipeline:
        {{product_name}}_finetuning.with_options(**pipeline_args)()

    if merging_pipeline:
        {{product_name}}_merging.with_options(**pipeline_args)()

    if eval_pipeline:
        {{product_name}}_evaluation.with_options(**pipeline_args)()


if __name__ == "__main__":
    main()
