# Fine-tuning Phi-3 with Microsoft Olive: A Step-by-Step Guide

## Overview

Microsoft Olive is an open-source, hardware-aware model optimization tool that automates techniques for model compression, optimization, and compilation. It simplifies the process of preparing and deploying machine learning models across various hardware targets, from cloud GPUs to edge devices.

This guide walks you through using Olive to fine-tune a Phi-3 model for a text classification task, merge the adapter weights, and optimize the model through quantization.

## Prerequisites

Ensure you have a system running **Ubuntu 20.04 or 22.04**. You will also need:
*   A Hugging Face account and access token.
*   (Optional) An Azure subscription with Azure Machine Learning configured if using cloud resources.

## Step 1: Install Microsoft Olive

Install Olive using `pip`. Choose the variant that matches your target hardware.

**For basic installation:**
```bash
pip install olive-ai
```

**For CPU-optimized ONNX runtime:**
```bash
pip install olive-ai[cpu]
```

**For GPU-optimized ONNX runtime:**
```bash
pip install olive-ai[gpu]
```

**To use Azure Machine Learning:**
```bash
pip install git+https://github.com/microsoft/Olive#egg=olive-ai[azureml]
```

## Step 2: Configure Your Fine-Tuning Job

Olive uses a `config.json` file to define the entire optimization pipeline. Below is a breakdown of its core sections.

### 2.1 Define the Input Model

Specify the base model you want to fine-tune. You can load it from Hugging Face or from the Azure AI Studio model catalog.

**Loading from Hugging Face (Local Cache):**
```json
{
    "input_model": {
        "type": "PyTorchModel",
        "config": {
            "hf_config": {
                "model_name": "model-cache/microsoft/phi-3-mini",
                "task": "text-generation",
                "model_loading_args": {
                    "trust_remote_code": true
                }
            }
        }
    }
}
```

**Loading from Azure AI Studio:**
```json
{
    "input_model": {
        "type": "PyTorchModel",
        "config": {
            "model_path": {
                "type": "azureml_registry_model",
                "config": {
                    "name": "microsoft/Phi-3-mini-4k-instruct",
                    "registry_name": "azureml-msr",
                    "version": "11"
                }
            },
            "model_file_format": "PyTorch.MLflow",
            "hf_config": {
                "model_name": "microsoft/Phi-3-mini-4k-instruct",
                "task": "text-generation",
                "from_pretrained_args": {
                    "trust_remote_code": true
                }
            }
        }
    }
}
```
> **Note:** When using Azure AI Studio models, set `model_file_format` to `"PyTorch.MLflow"`. Ensure your Hugging Face token is linked to your Azure ML workspace as a key vault secret.

### 2.2 Configure Your Training Data

Define the dataset for fine-tuning. Olive supports local files and cloud data stored in Azure ML datastores.

**Using a Local JSON File:**
This example configures a dataset for a tone classification task (e.g., Sad, Joy, Fear, Surprise).
```json
{
    "data_configs": [
        {
            "name": "dataset_default_train",
            "type": "HuggingfaceContainer",
            "load_dataset_config": {
                "params": {
                    "data_name": "json",
                    "data_files": "dataset/dataset-classification.json",
                    "split": "train"
                }
            },
            "pre_process_data_config": {
                "params": {
                    "dataset_type": "corpus",
                    "text_cols": ["phrase", "tone"],
                    "text_template": "### Text: {phrase}\n### The tone is:\n{tone}",
                    "corpus_strategy": "join",
                    "source_max_len": 2048,
                    "pad_to_max_len": false,
                    "use_attention_mask": false
                }
            }
        }
    ]
}
```

**Using an Azure ML Datastore:**
```json
{
    "data_configs": [
        {
            "name": "dataset_default_train",
            "type": "HuggingfaceContainer",
            "load_dataset_config": {
                "params": {
                    "data_name": "json",
                    "data_files": {
                        "type": "azureml_datastore",
                        "config": {
                            "azureml_client": {
                                "subscription_id": "YOUR_SUBSCRIPTION_ID",
                                "resource_group": "YOUR_RESOURCE_GROUP",
                                "workspace_name": "YOUR_WORKSPACE_NAME"
                            },
                            "datastore_name": "workspaceblobstore",
                            "relative_path": "path/to/your/train_data.json"
                        }
                    },
                    "split": "train"
                }
            },
            "pre_process_data_config": {
                "params": {
                    "dataset_type": "corpus",
                    "text_cols": ["Question", "Best Answer"],
                    "text_template": "<|user|>\n{Question}<|end|>\n<|assistant|>\n{Best Answer}\n<|end|>",
                    "corpus_strategy": "join",
                    "source_max_len": 2048,
                    "pad_to_max_len": false,
                    "use_attention_mask": false
                }
            }
        }
    ]
}
```

### 2.3 Set Up the Compute System

Configure where the fine-tuning job will run. For Azure ML, you must specify the compute target and a Docker environment.

```json
{
    "systems": {
        "aml": {
            "type": "AzureML",
            "config": {
                "accelerators": ["gpu"],
                "hf_token": true,
                "aml_compute": "YOUR_COMPUTE_CLUSTER_NAME",
                "aml_docker_config": {
                    "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-ubuntu22.04",
                    "conda_file_path": "conda.yaml"
                }
            }
        }
    }
}
```

The `conda.yaml` file defines the Python environment inside the Docker container:

```yaml
name: project_environment
channels:
  - defaults
dependencies:
  - python=3.8.13
  - pip=22.3.1
  - pip:
      - einops
      - accelerate
      - azure-keyvault-secrets
      - azure-identity
      - bitsandbytes
      - datasets
      - huggingface_hub
      - peft
      - scipy
      - sentencepiece
      - torch>=2.2.0
      - transformers
      - git+https://github.com/microsoft/Olive@jiapli/mlflow_loading_fix#egg=olive-ai[gpu]
      - --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/
      - ort-nightly-gpu==1.18.0.dev20240307004
      - --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
      - onnxruntime-genai-cuda
```

### 2.4 Define the Optimization Passes

This is the core of your pipeline. You define a sequence of "passes" that Olive will execute. Here, we configure QLoRA fine-tuning, weight merging, and INT4 quantization.

```json
{
    "passes": {
        "lora": {
            "type": "LoRA",
            "config": {
                "target_modules": ["o_proj", "qkv_proj"],
                "double_quant": true,
                "lora_r": 64,
                "lora_alpha": 64,
                "lora_dropout": 0.1,
                "train_data_config": "dataset_default_train",
                "eval_dataset_size": 0.3,
                "training_args": {
                    "seed": 0,
                    "data_seed": 42,
                    "per_device_train_batch_size": 1,
                    "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "gradient_checkpointing": false,
                    "learning_rate": 0.0001,
                    "num_train_epochs": 3,
                    "max_steps": 10,
                    "logging_steps": 10,
                    "evaluation_strategy": "steps",
                    "eval_steps": 187,
                    "group_by_length": true,
                    "adam_beta2": 0.999,
                    "max_grad_norm": 0.3
                }
            }
        },
        "merge_adapter_weights": {
            "type": "MergeAdapterWeights"
        },
        "builder": {
            "type": "ModelBuilder",
            "config": {
                "precision": "int4"
            }
        }
    }
}
```
> **Important Notes:**
> *   The `target_modules` must be adjusted based on your specific model architecture.
> *   **QLoRA Limitation:** If you use QLoRA (enabled by `"double_quant": true`), the subsequent `ModelBuilder` pass for ONNX Runtime GenAI quantization is not currently supported. For quantization, use standard LoRA (`"double_quant": false`).
> *   You can customize this sequence. For example, you could skip fine-tuning entirely and only run the `ModelBuilder` pass for quantization.

### 2.5 Configure the Execution Engine

Finally, tell Olive where to run the pipeline and where to save the outputs.

```json
{
    "engine": {
        "log_severity_level": 0,
        "host": "aml",
        "target": "aml",
        "search_strategy": false,
        "execution_providers": ["CUDAExecutionProvider"],
        "cache_dir": "../model-cache/models/phi3-finetuned/cache",
        "output_dir": "../model-cache/models/phi3-finetuned"
    }
}
```
*   `host` and `target`: Set to `"aml"` to use the Azure ML system defined earlier. For local runs, you would configure a local system and reference it here.
*   `execution_providers`: Defines the ONNX Runtime backend (e.g., CUDA for NVIDIA GPUs).
*   `cache_dir` & `output_dir`: Specify local paths for intermediate files and final optimized models.

## Step 3: Run the Fine-Tuning Pipeline

Once your `olive-config.json` file is complete, navigate to its directory in your terminal and execute the Olive run command:

```bash
olive run --config olive-config.json
```

Olive will execute the pipeline defined in your config:
1.  Download the base Phi-3 model.
2.  Fine-tune it using QLoRA on your specified dataset.
3.  Merge the LoRA adapter weights back into the base model.
4.  Convert and quantize the merged model to INT4 precision (if using standard LoRA).
5.  Save the final optimized model to the `output_dir`.

## Summary

You have now configured an end-to-end fine-tuning and optimization pipeline for the Phi-3 model using Microsoft Olive. By modifying the `config.json` file, you can easily adapt this process for different models, datasets, and hardware targets, significantly reducing the engineering effort required for model deployment.