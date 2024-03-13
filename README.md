# 📜 ZenML LLM Finetuning Template

This repository contains an LLM Finetuning template from which a simple ZenML project
can be generated. It contains a collection of steps, pipelines, configurations and
other artifacts and useful resources that can get you started with finetuning open-source LLMs
with ZenML.

🔥 **Do you have a personal project powered by ZenML that you would like to see here?** 

At ZenML, we are looking for design partnerships and collaboration to help us
better understand the real-world scenarios in which MLOps is being used and to
build the best possible experience for our users. If you are interested in
sharing all or parts of your project with us in the form of a ZenML project
template, please [join our Slack](https://zenml.io/slack/) and leave us a
message!

## 📃 Template Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Name | The name of the person/entity holding the copyright | ZenML GmbH |
| Email | The email of the person/entity holding the copyright | info@zenml.io |
| Project Name | Short name for your project | ZenML LLM Finetuning project |
| Project Version | The version of your project | 0.1.0 |
| Project License | The license under which your project will be released (one of `Apache Software License 2.0`, `MIT license`, `BSD license`, `ISC license`, `GNU General Public License v3` and `Not open source`) | Apache Software License 2.0 |
| Product name | The technical name to prefix all tech assets (pipelines, models, etc.) | llm_lora |
| Model repository | The Huggingface repository of the model to finetune | mistralai/Mistral-7B-Instruct-v0.1 |
| From safetensors | Whether the Huggingface model repository stores the model weights as safetensors | False |
| CUDA version | The available cuda version installed in the remote orchestrator environment (one of `CUDA 11.8` and `CUDA 12.1`) | CUDA 11.8 |
| Merged model repository | The huggingface repository to which the finetuned model should be pushed | - |
| Adapter repository | The huggingface repository to which the finetuned adapter should be pushed | - |
| Remote ZenML Server URL | Optional URL of a remote ZenML server for support scripts | - |

## 🚀 Generate a ZenML Project

First, to use the templates, you need to have Zenml and its `templates` extras installed: 

```bash
pip install "zenml[templates]"
```

Now you can generate a project from one of the existing templates by using the `--template` flag with the `zenml init` command:

```bash
zenml init --template <short_name_of_template>
# example: zenml init --template llm_finetuning
```

Running the command above will result in input prompts being shown to you. If you would like to rely on default values for the ZenML project template - you can add `--template-with-defaults` to the same command, like this:

```bash
zenml init --template <short_name_of_template> --template-with-defaults
# example: zenml init --template llm_finetuning --template-with-defaults
```
