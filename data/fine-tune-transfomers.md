# Fine-tuning a Multilingual Reasoner with Hugging Face

Authored by: [Edward Beeching](https://huggingface.co/edbeeching), [Quentin Gallouédec](https://huggingface.co/qgallouedec), and [Lewis Tunstall](https://huggingface.co/lewtun)

Large reasoning models like [OpenAI o3](https://openai.com/index/introducing-o3-and-o4-mini/) generate a chain-of-thought to improve the accuracy and quality of their responses. However, most of these models reason in English, even when a question is asked in another language.

In this guide, we show how OpenAI's open-weight reasoning model [OpenAI gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) can be fine-tuned to reason effectively in multiple languages. We'll do this by adding a new _"reasoning language"_ option to the model's system prompt and applying [supervised fine-tuning](https://huggingface.co/learn/llm-course/chapter11/1) with Hugging Face's [TRL library](https://github.com/huggingface/trl) on a multilingual reasoning dataset.

We'll cover the following steps:

1.  **Setup:** Install the required libraries.
2.  **Prepare the Dataset:** Download and format the dataset for fine-tuning.
3.  **Prepare the Model:** Load the base model and configure it for memory-efficient fine-tuning with [LoRA](https://huggingface.co/learn/llm-course/chapter11/4).
4.  **Fine-tuning:** Train the model with our multilingual reasoning data.
5.  **Inference:** Generate reasoning responses in different languages using the fine-tuned model.

The end result is a multilingual reasoning model that can generate a chain-of-thought in English, Spanish, French, Italian, or German. You can even *mix languages*—for example, ask a question in Spanish, request reasoning in German, and receive the final response in Spanish.

We hope this tutorial will enable AI developers working with under-represented languages to improve the interpretability of [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) in their native languages.

> **Note:** This guide is designed to be run on a single H100 GPU with 80GB of memory. If you have access to a smaller GPU, you can reduce the batch size and sequence length in the hyperparameters below.

## 1. Setup

To get started, let's install all the necessary libraries.

First, install PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Next, install the remaining dependencies:

```bash
pip install "trl>=0.20.0" "peft>=0.17.0" "transformers>=4.55.0" trackio
```

Finally, log into your Hugging Face account:

```python
from huggingface_hub import notebook_login

notebook_login()
```

## 2. Prepare the Dataset

We will be using the [Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) dataset, where the chain-of-thought has been translated into several languages such as French, Spanish, and German. By fine-tuning `openai/gpt-oss-20b` on this dataset, it will learn to generate reasoning steps in these languages.

Let's download this dataset from the Hugging Face Hub:

```python
from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
dataset
```

This is a small dataset of 1,000 examples, which is usually sufficient for models like `openai/gpt-oss-20b` that have undergone extensive post-training. Let's examine one of the training examples:

```python
dataset[0]
```

The `gpt-oss` models were trained on the Harmony response format. The table below summarizes the different message types used in the dataset:

| Role | Description |
| :--- | :--- |
| `developer` | Provides custom instructions for the model (similar to the `system` role). |
| `user` | The input from the user. |
| `assistant` | Output by the model, which can be a tool call or a message. It may be associated with a specific "channel" identifying the intent. |
| `analysis` | Used by the model for its chain-of-thought reasoning. |
| `final` | Messages tagged in the final channel are intended to be shown to the end-user as the model's response. |
| `messages` | The list of messages that combine the content above to produce a full conversation. This is the input to the model. |

If you're familiar with [OpenAI's messages format](https://platform.openai.com/docs/api-reference/messages/object), you will recognize this as being quite similar, but with an important difference:

> The `assistant` turn contains two special fields: a `thinking` one which contains the model's reasoning process, and a `content` one which contains the final response to the user.

To fine-tune the model, we need to convert these messages into a format the model can understand. This is done by formatting each message with the model's [_chat template_](https://huggingface.co/docs/transformers/chat_templating) and then tokenizing the resulting text. The TRL library does this automatically, but let's walk through it step by step.

First, load the tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
```

Then, use the tokenizer's `apply_chat_template()` method to format the messages:

```python
messages = dataset[0]["messages"]
conversation = tokenizer.apply_chat_template(messages, tokenize=False)
print(conversation)
```

This chat template is sophisticated. Let's break it down:
*   **Special Tokens:** `<|start|>` and `<|end|>` indicate the start and end of each message. `<|return|>` marks the end of the conversation.
*   **System Messages:** There are two types:
    *   A default `system` message (e.g., "You are ChatGPT, a large language model trained by OpenAI...").
    *   A special `developer` message containing custom instructions (defined by the `system` role in our `messages` object).
*   **Assistant Channels:** The assistant response is contained in channels:
    *   The `analysis` channel is for the model's reasoning process.
    *   The `final` channel is for the model's final response to the user.

Now that we understand the dataset format, let's prepare the model for training.

## 3. Prepare the Model

To prepare the model for training, let's first download the weights from the Hugging Face Hub. We will use the `AutoModelForCausalLM` class from 🤗 Transformers to load the model:

```python
import torch
from transformers import AutoModelForCausalLM, Mxfp4Config

quantization_config = Mxfp4Config(dequantize=True)
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)

model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs)
```

This loads the model with configurations for training. The `attn_implementation` is set to `eager` for better performance, and `use_cache` is set to `False` since we will fine-tune the model with gradient checkpointing.

We are using the `Mxfp4Config` for quantization. This is a specific configuration for OpenAI models that allows mixed precision training with a special 4-bit floating point format called [MXFP4](https://en.wikipedia.org/wiki/Block_floating_point) optimized for AI workloads.

Before training, let's generate a sample response to see the model's default behavior:

```python
messages = [
    {"role": "user", "content": "¿Cuál es el capital de Australia?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

output_ids = model.generate(input_ids, max_new_tokens=512)
response = tokenizer.batch_decode(output_ids)[0]
print(response)
```

The model first reasons about the question in English, then provides a final response in Spanish. Let's see if we can change this with fine-tuning.

We will use [LoRA](https://huggingface.co/learn/llm-course/chapter11/4) (Low-Rank Adaptation) to fine-tune the model efficiently. First, wrap the model as a `PeftModel` and define the LoRA configuration:

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    target_parameters=[
        "7.mlp.experts.gate_up_proj",
        "7.mlp.experts.down_proj",
        "15.mlp.experts.gate_up_proj",
        "15.mlp.experts.down_proj",
        "23.mlp.experts.gate_up_proj",
        "23.mlp.experts.down_proj",
    ],
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
```

Here we've used basic hyperparameters for LoRA. You can experiment with different values (e.g., increasing `r`) to see how they affect performance.

> **Note:** The `openai/gpt-oss-20b` model is a [Mixture-of-Experts (MoE)](https://huggingface.co/blog/moe) architecture. In addition to targeting the attention layers (`target_modules="all-linear"`), it's also important to include the projection layers within the expert modules. PEFT facilitates this via the `target_parameters` argument, which allows you to specify expert-specific layers such as `mlp.experts.down_proj` and `mlp.experts.gate_up_proj`. In this example, we target a subset of these projection layers, but you are encouraged to experiment with different configurations.

Now that we have the model and dataset ready, we can define the hyperparameters for training.

## 4. Fine-tuning

TRL provides a convenient way to define hyperparameters using the `SFTConfig` class:

```python
from trl import SFTConfig

training_args = SFTConfig(
    learning_rate=2e-4,
    gradient_checkpointing=True,
    num_train_epochs=1,
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_length=2048,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    output_dir="gpt-oss-20b-multilingual-reasoner",
    report_to="trackio",
    push_to_hub=True,
)
```

Note that `per_device_train_batch_size` is set to 4, and `gradient_accumulation_steps` is set to 4. This means we will effectively have a batch size of 16 across 1 GPU. Adjust these values based on your hardware setup.

We now have all the pieces needed to train the model. We will use the `SFTTrainer` class from TRL:

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

On an H100 GPU, this takes about 18 minutes to train, but may take longer depending on your hardware.

## 5. Save the Model and Push to the Hugging Face Hub

Finally, you can push the fine-tuned model to your Hub repository to share with the community:

```python
trainer.save_model(training_args.output_dir)
trainer.push_to_hub(dataset_name="HuggingFaceH4/Multilingual-Thinking")
```

> **Note:** To avoid out-of-memory (OOM) errors, we recommend restarting the kernel at this point. The trained model is still occupying GPU memory, but it's no longer needed.

## 6. Inference

Once the model is uploaded to the Hub, we can use it for inference. First, initialize the original base model and its tokenizer, then merge the fine-tuned weights with the base model for fast inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

# Load the original model first
model_kwargs = dict(attn_implementation="eager", torch_dtype="auto", use_cache=True, device_map="auto")
base_model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs).cuda()

# Merge fine-tuned weights with the base model
peft_model_id = "gpt-oss-20b-multilingual-reasoner"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model = model.merge_and_unload()
```

Now that the model is loaded, we can generate tokens. Let's define a prompt that requests reasoning in German for a Spanish question:

```python
REASONING_LANGUAGE = "German"
SYSTEM_PROMPT = f"reasoning language: {REASONING_LANGUAGE}"
USER_PROMPT = "¿Cuál es el capital de Australia?"  # Spanish for "What is the capital of Australia?"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

gen_kwargs = {"max_new_tokens": 512, "do_sample": True, "temperature": 0.6, "top_p": None, "top_k": None}

output_ids = model.generate(input_ids, **gen_kwargs)
response = tokenizer.batch_decode(output_ids)[0]
print(response)
```

You should now see the model generating its reasoning steps in German, while the final answer remains in Spanish. Congratulations! You have successfully fine-tuned a multilingual reasoning model.