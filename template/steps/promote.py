# {% include 'template/license_header' %}

from zenml import get_step_context, step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.utils.cuda_utils import cleanup_gpu_memory

logger = get_logger(__name__)


@step(enable_cache=False)
def promote(
    metric: str = "rouge1",
    target_stage: str = "staging",
) -> None:
    """Promote the model to the target stage.

    If the model does not outperform the base model, it will be skipped.
    If the model does not outperform the model in the target stage, it will be skipped.

    Args:
        metric: The metric to use for promotion.
        target_stage: The target stage to promote to.
    """
    cleanup_gpu_memory(force=True)
    context_model = get_step_context().model
    base_metrics = context_model.load_artifact("base_model_rouge_metrics")
    ft_metrics = context_model.load_artifact("finetuned_model_rouge_metrics")
    staging_metrics = None
    try:
        staging_model = Client().get_model_version(context_model.name, target_stage)
        staging_metrics = staging_model.get_artifact(
            "finetuned_model_rouge_metrics"
        ).load()
    except KeyError:
        pass

    msg = (
        f"`{metric}` values to compare:\n"
        f"base={base_metrics[metric]*100:.2f}%\n"
        f"finetuned={ft_metrics[metric]*100:.2f}%"
    )
    if staging_metrics:
        msg += f"\nstaging={staging_metrics[metric]*100:.2f}%"
    logger.info(msg)

    if base_metrics[metric] <= ft_metrics[metric]:
        if staging_metrics is not None and (
            staging_metrics[metric] > ft_metrics[metric]
        ):
            logger.info(
                "Skipping promotion: model does not "
                f"outperform the current model in `{target_stage}`."
            )
        else:
            logger.info(f"Promoting model to `{target_stage}`")
            get_step_context().model.set_stage(target_stage, True)
    else:
        logger.info("Skipping promotion: model does not outperform the base model.")