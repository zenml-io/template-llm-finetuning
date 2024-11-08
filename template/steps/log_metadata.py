# {% include 'template/license_header' %}

from typing import Any, Dict

from zenml import log_model_metadata, step, get_step_context


@step(enable_cache=False)
def log_metadata_from_step_artifact(
    step_name: str,
    artifact_name: str,
) -> None:
    """Log metadata to the model from saved artifact.

    Args:
        step_name: The name of the step.
        artifact_name: The name of the artifact.
    """

    context = get_step_context()
    metadata_dict: Dict[str, Any] = (
        context.pipeline_run.steps[step_name].outputs[artifact_name][0].load()
    )

    metadata = {artifact_name: metadata_dict}

    log_model_metadata(metadata)