#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import os
import pathlib
import shutil
import subprocess
import sys

import pytest
from copier import Worker
from zenml.client import Client
from zenml.enums import ExecutionStatus

TEMPLATE_DIRECTORY = str(pathlib.Path.joinpath(pathlib.Path(__file__).parent.parent))


def generate_and_run_project(
    tmp_path_factory: pytest.TempPathFactory,
    product_name: str = "llm-peft-pytest",
    model_repository: str = "microsoft/phi-2",
):
    """Generate and run the starter project with different options."""

    answers = {
        "project_name": "Pytest Templated Project",
        "version": "0.0.1",
        "open_source_license": "apache",
        "email": "pytest@zenml.io",
        "full_name": "Pytest",
        "product_name": product_name,
        "model_repository": model_repository,
        "steps_of_finetuning": 1,
        "cuda_version": "cuda11.8",
        "system_prompt": """
Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']""",
        "dataset_name": "gem/viggo",
        "step_operator": "gcp_a100",
        "bf16": False,
        "zenml_server_url": "",
        "use_fast_tokenizer": False,
    }

    # generate the template in a temp path
    current_dir = os.getcwd()
    dst_path = tmp_path_factory.mktemp("pytest-template")
    config_path = os.path.join(TEMPLATE_DIRECTORY, "tests", "test_config.yaml")
    print("TEMPLATE_DIR:", TEMPLATE_DIRECTORY)
    print("dst_path:", dst_path)
    print("current_dir:", current_dir)
    os.chdir(str(dst_path))
    with Worker(
        src_path=TEMPLATE_DIRECTORY,
        dst_path=str(dst_path),
        data=answers,
        unsafe=True,
        vcs_ref="HEAD",
    ) as worker:
        worker.run_copy()

    # use only data prep in the testing - finetuning is too heavy for runners
    product_name_underscored = product_name.replace("-", "_")
    os.remove(os.path.join(dst_path, "pipelines", "train.py"))
    with open(os.path.join(TEMPLATE_DIRECTORY, "tests", "unit_test_pipeline.py"),"r") as src:
        with open(os.path.join(dst_path, "pipelines", "train.py"), "w") as dst:
            dst.write(src.read().replace("{{PLACEHOLDER}}", product_name_underscored))

    call = [
        sys.executable,
        "run.py",
        "--config",
        config_path,
        "--no-cache",
    ]

    try:
        process = subprocess.Popen(
            call,
            cwd=str(dst_path),
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for line in iter(process.stdout.readline, b""):
            print(line.decode(),end="")
        process.wait()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to run project generated with parameters: {answers}\n"
            f"{e.output.decode()}"
        ) from e

    pipeline_name = f"{product_name_underscored}_full_finetune"
    pipeline = Client().get_pipeline(pipeline_name)
    assert pipeline
    runs = pipeline.runs
    assert len(runs) == 1
    assert runs[0].status == ExecutionStatus.COMPLETED

    # clean up
    Client().delete_pipeline(pipeline_name)
    Client().delete_model(f"pytest-{model_repository.replace('/','-')}")

    os.chdir(current_dir)
    shutil.rmtree(dst_path)


def test_custom_product_name(
    clean_zenml_client,
    tmp_path_factory: pytest.TempPathFactory,
):
    """Test using custom pipeline name."""

    generate_and_run_project(
        tmp_path_factory=tmp_path_factory,
        product_name="custom-product-name",
    )
