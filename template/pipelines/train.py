# {% include 'template/license_header' %}


from steps import evaluate_model, finetune, prepare_data, promote
from zenml import logging as zenml_logging
from zenml import pipeline

zenml_logging.STEP_LOGS_STORAGE_MAX_MESSAGES = (
    10000  # workaround for https://github.com/zenml-io/zenml/issues/2252
)


@pipeline
def llm_peft_full_finetune(
    system_prompt:str, 
    base_model_id:str,
    use_fast: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    cpu_mode: bool = False,
):
    """Pipeline for finetuning an LLM with peft.
    
    It will run the following steps:

    - prepare_data: prepare the datasets and tokenize them
    - finetune: finetune the model
    - evaluate_model: evaluate the base and finetuned model
    - promote: promote the model to the target stage, if evaluation was successful
    """ 
    datasets_dir = prepare_data(
        base_model_id=base_model_id, 
        system_prompt=system_prompt,
        use_fast=use_fast,
    )
    ft_model_dir = finetune(
        base_model_id,
        datasets_dir,
        use_fast=use_fast,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        cpu_mode=cpu_mode,
    )
    evaluate_model(
        base_model_id,
        system_prompt,
        datasets_dir,
        ft_model_dir,
        use_fast=use_fast,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        cpu_mode=cpu_mode,
        id="evaluate_finetuned",
    )
    evaluate_model(
        base_model_id,
        system_prompt,
        datasets_dir,
        None,
        use_fast=use_fast,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        cpu_mode=cpu_mode,
        id="evaluate_base",
    )
    promote(after=["evaluate_finetuned", "evaluate_base"])
