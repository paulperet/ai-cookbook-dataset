# Fine-tune ChatGPT-3.5 and GPT-4 with Weights & Biases

<!--- @wandbcode{openai-finetune-gpt35} -->

<a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/openai/Fine_tune_GPT_3_with_Weights_&_Biases.ipynb" target="_parent">Open In Colab</a>

**Note:** you will need an [OpenAI API key](https://platform.openai.com/account/api-keys) to run this colab.

If you use OpenAI's API to [fine-tune ChatGPT-3.5](https://platform.openai.com/docs/guides/fine-tuning), you can now use the W&B integration to track experiments, models, and datasets in your central dashboard.

All it takes is one line: `openai wandb sync`

See the [OpenAI section](https://wandb.me/openai-docs) in the Weights & Biases documentation for full details of the integration

```
!pip install -Uq openai tiktoken datasets tenacity wandb
```

```
# Remove once this PR is merged: https://github.com/openai/openai-python/pull/590 and openai release is made
!pip uninstall -y openai -qq \
&& pip install git+https://github.com/morganmcg1/openai-python.git@update_wandb_logger -qqq
```

## Optional: Fine-tune ChatGPT-3.5

It's always more fun to experiment with your own projects so if you have already used the openai API to fine-tune an OpenAI model, just skip this section.

Otherwise let's fine-tune ChatGPT-3.5 on a legal dataset!

### Imports and initial set-up

```
import openai
import wandb

import os
import json
import random
import tiktoken
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_fixed
```

Start your Weigths & Biases run. If you don't have an account you can sign up for one for free at www.wandb.ai

```
WANDB_PROJECT = "OpenAI-Fine-Tune"
```

### Set up your API key

```
# # Enter credentials
openai_key = "YOUR_API_KEY"

openai.api_key = openai_key
```

### Dataset Preparation

We download a dataset from [LegalBench](https://hazyresearch.stanford.edu/legalbench/), a project to curate tasks for evaluating legal reasoning, specifically the [Contract NLI Explicit Identification task](https://github.com/HazyResearch/legalbench/tree/main/tasks/contract_nli_explicit_identification).

This comprises of a total of 117 examples, from which we will create our own train and test datasets

```
from datasets import load_dataset

# Download the data, merge into a single dataset and shuffle
dataset = load_dataset("nguha/legalbench", "contract_nli_explicit_identification")

data = []
for d in dataset["train"]:
  data.append(d)

for d in dataset["test"]:
  data.append(d)

random.shuffle(data)

for idx, d in enumerate(data):
  d["new_index"] = idx
```

Let's look at a few samples.

```
len(data), data[0:2]
```

### Format our Data for Chat Completion Models
We modify the `base_prompt` from the LegalBench task to make it a zero-shot prompt, as we are training the model instead of using few-shot prompting

```
base_prompt_zero_shot = "Identify if the clause provides that all Confidential Information shall be expressly identified by the Disclosing Party. Answer with only `Yes` or `No`"
```

We now split it into training/validation dataset, lets train on 30 samples and test on the remainder

```
n_train = 30
n_test = len(data) - n_train
```

```
train_messages = []
test_messages = []

for d in data:
  prompts = []
  prompts.append({"role": "system", "content": base_prompt_zero_shot})
  prompts.append({"role": "user", "content": d["text"]})
  prompts.append({"role": "assistant", "content": d["answer"]})

  if int(d["new_index"]) < n_train:
    train_messages.append({'messages': prompts})
  else:
    test_messages.append({'messages': prompts})

len(train_messages), len(test_messages), n_test, train_messages[5]
```

### Save the data to Weigths & Biases

Save the data in a train and test file first

```
train_file_path = 'encoded_train_data.jsonl'
with open(train_file_path, 'w') as file:
    for item in train_messages:
        line = json.dumps(item)
        file.write(line + '\n')

test_file_path = 'encoded_test_data.jsonl'
with open(test_file_path, 'w') as file:
    for item in test_messages:
        line = json.dumps(item)
        file.write(line + '\n')
```

Next, we validate that our training data is in the correct format using a script from the [OpenAI fine-tuning documentation](https://platform.openai.com/docs/guides/fine-tuning/)

```
# Next, we specify the data path and open the JSONL file

def openai_validate_data(dataset_path):
  data_path = dataset_path

  # Load dataset
  with open(data_path) as f:
      dataset = [json.loads(line) for line in f]

  # We can inspect the data quickly by checking the number of examples and the first item

  # Initial dataset stats
  print("Num examples:", len(dataset))
  print("First example:")
  for message in dataset[0]["messages"]:
      print(message)

  # Now that we have a sense of the data, we need to go through all the different examples and check to make sure the formatting is correct and matches the Chat completions message structure

  # Format error checks
  format_errors = defaultdict(int)

  for ex in dataset:
      if not isinstance(ex, dict):
          format_errors["data_type"] += 1
          continue

      messages = ex.get("messages", None)
      if not messages:
          format_errors["missing_messages_list"] += 1
          continue

      for message in messages:
          if "role" not in message or "content" not in message:
              format_errors["message_missing_key"] += 1

          if any(k not in ("role", "content", "name") for k in message):
              format_errors["message_unrecognized_key"] += 1

          if message.get("role", None) not in ("system", "user", "assistant"):
              format_errors["unrecognized_role"] += 1

          content = message.get("content", None)
          if not content or not isinstance(content, str):
              format_errors["missing_content"] += 1

      if not any(message.get("role", None) == "assistant" for message in messages):
          format_errors["example_missing_assistant_message"] += 1

  if format_errors:
      print("Found errors:")
      for k, v in format_errors.items():
          print(f"{k}: {v}")
  else:
      print("No errors found")

  # Beyond the structure of the message, we also need to ensure that the length does not exceed the 4096 token limit.

  # Token counting functions
  encoding = tiktoken.get_encoding("cl100k_base")

  # not exact!
  # simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
  def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
      num_tokens = 0
      for message in messages:
          num_tokens += tokens_per_message
          for key, value in message.items():
              num_tokens += len(encoding.encode(value))
              if key == "name":
                  num_tokens += tokens_per_name
      num_tokens += 3
      return num_tokens

  def num_assistant_tokens_from_messages(messages):
      num_tokens = 0
      for message in messages:
          if message["role"] == "assistant":
              num_tokens += len(encoding.encode(message["content"]))
      return num_tokens

  def print_distribution(values, name):
      print(f"\n#### Distribution of {name}:")
      print(f"min / max: {min(values)}, {max(values)}")
      print(f"mean / median: {np.mean(values)}, {np.median(values)}")
      print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

  # Last, we can look at the results of the different formatting operations before proceeding with creating a fine-tuning job:

  # Warnings and tokens counts
  n_missing_system = 0
  n_missing_user = 0
  n_messages = []
  convo_lens = []
  assistant_message_lens = []

  for ex in dataset:
      messages = ex["messages"]
      if not any(message["role"] == "system" for message in messages):
          n_missing_system += 1
      if not any(message["role"] == "user" for message in messages):
          n_missing_user += 1
      n_messages.append(len(messages))
      convo_lens.append(num_tokens_from_messages(messages))
      assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

  print("Num examples missing system message:", n_missing_system)
  print("Num examples missing user message:", n_missing_user)
  print_distribution(n_messages, "num_messages_per_example")
  print_distribution(convo_lens, "num_total_tokens_per_example")
  print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
  n_too_long = sum(l > 4096 for l in convo_lens)
  print(f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

  # Pricing and default n_epochs estimate
  MAX_TOKENS_PER_EXAMPLE = 4096

  MIN_TARGET_EXAMPLES = 100
  MAX_TARGET_EXAMPLES = 25000
  TARGET_EPOCHS = 3
  MIN_EPOCHS = 1
  MAX_EPOCHS = 25

  n_epochs = TARGET_EPOCHS
  n_train_examples = len(dataset)
  if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
      n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
  elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
      n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

  n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
  print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
  print(f"By default, you'll train for {n_epochs} epochs on this dataset")
  print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
  print("See pricing page to estimate total costs")
```

Validate train data

```
openai_validate_data(train_file_path)
```

Log our data to Weigths & Biases Artifacts for storage and versioning

```
wandb.init(
    project=WANDB_PROJECT,
    # entity="prompt-eng",
    job_type="log-data",
    config = {'n_train': n_train,
              'n_valid': n_test})

wandb.log_artifact(train_file_path,
                   "legalbench-contract_nli_explicit_identification-train",
                   type="train-data")

wandb.log_artifact(test_file_path,
                   "legalbench-contract_nli_explicit_identification-test",
                   type="test-data")

# keep entity (typically your wandb username) for reference of artifact later in this demo
entity = wandb.run.entity

wandb.finish()
```

### Create a fine-tuned model

We'll now use OpenAI API to fine-tune ChatGPT-3.5

Let's first download our training & validation files and save them to a folder called `my_data`. We will retrieve the `latest` version of the artifact, but it could also be `v0`, `v1` or any alias we associated with it

```
wandb.init(project=WANDB_PROJECT,
          #  entity="prompt-eng",
           job_type="finetune")

artifact_train = wandb.use_artifact(
    f'{entity}/{WANDB_PROJECT}/legalbench-contract_nli_explicit_identification-train:latest',
    type='train-data')
train_file = artifact_train.get_path(train_file_path).download("my_data")

train_file
```

Then we upload the training data to OpenAI. OpenAi has to process the data, so this will take a few minutes depending on the size of your dataset.

```
openai_train_file_info = openai.File.create(
  file=open(train_file, "rb"),
  purpose='fine-tune'
)

# you may need to wait a couple of minutes for OpenAI to process the file
openai_train_file_info
```

### Time to train the model!

Let's define our ChatGPT-3.5 fine-tuning hyper-parameters.

```
model = 'gpt-3.5-turbo'
n_epochs = 3
```

```
openai_ft_job_info = openai.FineTuningJob.create(
    training_file=openai_train_file_info["id"],
    model=model,
    hyperparameters={"n_epochs": n_epochs}
)

ft_job_id = openai_ft_job_info["id"]

openai_ft_job_info
```

> this takes around 5 minutes to train, and you get an email from OpenAI when finished.

**Thats it!**

Now your model is training on OpenAI's machines. To get the current state of your fine-tuning job, run:

```
state = openai.FineTuningJob.retrieve(ft_job_id)
state["status"], state["trained_tokens"], state["finished_at"], state["fine_tuned_model"]
```

Show recent events for our fine-tuning job

```
openai.FineTuningJob.list_events(id=ft_job_id, limit=5)
```

We can run a few different fine-tunes with different parameters or even with different datasets.

## Log OpenAI fine-tune jobs to Weights & Biases

We can log our fine-tunes with a simple command.

```
!openai wandb sync --help
```

Calling `openai wandb sync` will log all un-synced fine-tuned jobs to W&B

Below we are just logging 1 job, passing:
- our OpenAI key as an environment variable
- the id of the fine-tune job we'd like to log
- the W&B project of where to log it to

See the [OpenAI section](https://wandb.me/openai-docs) in the Weights & Biases documentation for full details of the integration

```
!OPENAI_API_KEY={openai_key} openai wandb sync --id {ft_job_id} --project {WANDB_PROJECT}
```

```
wandb.finish()
```

Our fine-tunes are now successfully synced to Weights & Biases.

Anytime we have new fine-tunes, we can just call `openai wandb sync` to add them to our dashboard.

## Run evalution and log the results

The best way to evaluate a generative model is to explore sample predictions from your evaluation set.

Let's generate a few inference samples and log them to W&B and see how the performance compares to a baseline ChatGPT-3.5 model

```
wandb.init(project=WANDB_PROJECT,
           job_type='eval')

artifact_valid = wandb.use_artifact(
    f'{entity}/{WANDB_PROJECT}/legalbench-contract_nli_explicit_identification-test:latest',
    type='test-data')
test_file = artifact_valid.get_path(test_file_path).download("my_data")

with open(test_file) as f:
    test_dataset = [json.loads(line) for line in f]

print(f"There are {len(test_dataset)} test examples")
wandb.config.update({"num_test_samples":len(test_dataset)})
```

### Run evaluation on the Fine-Tuned Model
Set up OpenAI call with retries

```
@retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
def call_openai(messages="", model="gpt-3.5-turbo"):
  return openai.ChatCompletion.create(model=model, messages=messages, max_tokens=10)
```

Let's get our trained model id

```
state = openai.FineTuningJob.retrieve(ft_job_id)
ft_model_id = state["fine_tuned_model"]
ft_model_id
```

Run evaluation and log results to W&B

```
prediction_table = wandb.Table(columns=['messages', 'completion', 'target'])

eval_data = []

for row in tqdm(test_dataset):
    messages = row['messages'][:2]
    target = row["messages"][2]

    # res = call_openai(model=ft_model_id, messages=messages)
    res = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=10)
    completion = res.choices[0].message.content

    eval_data.append([messages, completion, target])
    prediction_table.add_data(messages[1]['content'], completion, target["content"])

wandb.log({'predictions': prediction_table})
```

Calculate the accuracy of the fine-tuned model and log to W&B

```
correct = 0
for e in eval_data:
  if e[1].lower() == e[2]["content"].lower():
    correct+=1

accuracy = correct / len(eval_data)

print(f"Accuracy is {accuracy}")
wandb.log({"eval/accuracy": accuracy})
wandb.summary["eval/accuracy"] = accuracy
```

### Run evaluation on a Baseline model for comparison
Lets compare our model to the baseline model, `gpt-3.5-turbo`

```
baseline_prediction_table = wandb.Table(columns=['messages', 'completion', 'target'])
baseline_eval_data