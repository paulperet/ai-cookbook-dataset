### 🔖 GitHub Tag Generator with T5 + PEFT (LoRA)

**_Authored by: [Zamal Babar](https://huggingface.co/zamal)_**

In this notebook, we walk through a complete **end-to-end implementation** of a lightweight, fast, and open-source **GitHub tag generator** using **T5-small** fine-tuned on a custom dataset with **PEFT (LoRA)**. This tool can automatically generate relevant tags from a GitHub repository description or summary — useful for improving discoverability and organizing repos more intelligently.

---

#### 💡 Use Case

Imagine you're building a tool that helps users explore GitHub repositories more effectively. Instead of relying on manually written or sometimes missing tags, we train a model that **automatically generates descriptive tags** for any GitHub project. This could help:

- Improve search functionality  
- Automatically tag new repos  
- Build better filters for discovery  

---

#### 📦 Dataset

We use a dataset of GitHub project descriptions and their associated tags. Each training example contains:

- `"input"`: A natural language description of a GitHub repository  
- `"target"`: A comma-separated list of relevant tags  

The dataset was initially loaded from a local `.jsonl` file, but is now also available on the Hugging Face Hub here:  
➡️ [`zamal/github-meta-data`](https://huggingface.co/datasets/zamal/github-meta-data)

---

#### 🧠 Model Architecture

We fine-tuned the [`T5-small`](https://huggingface.co/t5-small) model for this task — a lightweight encoder-decoder transformer that's well-suited for text-to-text generation tasks.  
To make fine-tuning faster and more efficient, we used the 🤗 `peft` library with **LoRA (Low-Rank Adaptation)** to update only a subset of model parameters.

---

#### ✅ What This Notebook Covers

This notebook includes:

- ✅ Loading and preprocessing a custom dataset  
- ✅ Setting up a T5-small model with LoRA  
- ✅ Training the model using the Hugging Face `Trainer`  
- ✅ Monitoring progress with **Weights & Biases**  
- ✅ Saving and pushing the model to the Hugging Face Hub  
- ✅ Performing inference and postprocessing for clean, deduplicated tags  

---

#### 🔍 Final Outcome

By the end of this notebook, you’ll have:

- 🚀 A fully trained and hosted GitHub tag generator  
- 🔁 A deployable and shareable model on Hugging Face Hub  
- 🧠 An inference function to use your model anywhere with just a few lines of code  

Let’s dive in! 🎯


We begin by:

- Importing essential libraries for model training (`transformers`, `datasets`, `peft`)
- Loading the T5 tokenizer
- Setting the Hugging Face token (stored securely in Colab’s `userdata`)

Make sure you've stored your `HUGGINGFACE_TOKEN` in your Colab's secrets before running this cell.



```python
from google.colab import userdata
import os
os.environ['HUGGINGFACE_TOKEN'] = userdata.get('HUGGINGFACE_TOKEN')
```


```python
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig
```


```python
tokenizer = T5Tokenizer.from_pretrained("t5-small")
```

    You are using the default legacy behaviour of the <class 'transformers.models.t5.tokenization_t5.T5Tokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565


#### 📦 Load and Prepare the Dataset

We now load our training data from a local JSONL file that contains repository descriptions and their corresponding tags.

Each line in the file is a JSON object with two fields:
- `input`: a short repository description
- `target`: the tags (comma-separated)

We split this dataset into training and validation sets using a 90/10 ratio.

🔁 _Note_: When this notebook was initially run, the dataset was loaded locally from a file. However, the same dataset is now also available on the Hugging Face Hub here: [zamal/github-meta-data](https://huggingface.co/datasets/zamal/github-meta-data). Feel free to load it directly using `load_dataset("zamal/github-meta-data")` in your workflow as shown below.



```python
from datasets import load_dataset, DatasetDict

# Load existing dataset with only a "train" split
dataset = load_dataset("zamal/github-meta-data")  # returns DatasetDict

# Split the train set into train and validation
split = dataset["train"].train_test_split(test_size=0.1, seed=42)

# Wrap into a new DatasetDict
dataset_dict = DatasetDict({
    "train": split["train"],
    "validation": split["test"]
})

```


```python
print(len(dataset_dict["train"]))
print(len(dataset_dict["validation"]))
```

    552
    62


#### 🔤 Load the Tokenizer

We load the tokenizer associated with the `t5-small` model. T5 expects input and output text to be tokenized in a specific way, and this tokenizer ensures compatibility during training and inference.



```python
from transformers import AutoTokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

#### 🧹 Preprocessing the Dataset

Next, we define a preprocessing function to tokenize both the inputs and the targets using the T5 tokenizer.
- The inputs are padded and truncated to a maximum length of 128 tokens.
- The target labels (i.e., tags) are also tokenized with a shorter maximum length of 64 tokens.

We then map this preprocessing function across our training and validation datasets and format the output for PyTorch compatibility. This prepares the dataset for training.



```python
def preprocess(batch):
    inputs = batch["input"]
    targets = batch["target"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

```


```python
tokenized = dataset_dict.map(preprocess, batched=True, remove_columns=dataset_dict["train"].column_names)
tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
```

#### Loading the Base T5 Model

We load the base T5 model (`t5-small`) for conditional generation. This model serves as the backbone for our tag generation task, where the goal is to generate relevant tags given a description of a GitHub repository.



```python
model = T5ForConditionalGeneration.from_pretrained(model_name)
```

    Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
    WARNING:huggingface_hub.file_download:Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`


#### 🔧 Preparing the LoRA Configuration

We configure LoRA (Low-Rank Adaptation) to fine-tune the T5 model efficiently. LoRA injects trainable low-rank matrices into attention layers, significantly reducing the number of trainable parameters while maintaining performance.

In this setup:
- `r=16` defines the rank of the update matrices.
- `lora_alpha=32` scales the updates.
- We apply LoRA to the `"q"` and `"v"` attention projection modules.
- The task type is set to `"SEQ_2_SEQ_LM"` since we're working on a sequence-to-sequence task.



```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],  # Adjust based on model architecture
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

```

#### 🔌 Injecting LoRA into the Base T5 Model

Now that we've defined our LoRA configuration, we apply it to the base T5 model using `get_peft_model()`. This wraps the original model with the LoRA adapters, allowing us to fine-tune only a small number of parameters instead of the entire model—making training faster and more memory-efficient.



```python
model = get_peft_model(model, lora_config)
```

#### 🛠️ TrainingArguments Configuration

We use the `TrainingArguments` class to define the hyperparameters and training behavior for our model. Here's a breakdown of each parameter:

- **`output_dir="./t5_tag_generator"`**  
  Directory to save model checkpoints and training logs.

- **`per_device_train_batch_size=8`**  
  Number of training samples per GPU/TPU core (or CPU) in each training step.

- **`per_device_eval_batch_size=8`**  
  Number of evaluation samples per GPU/TPU core (or CPU) in each evaluation step.

- **`learning_rate=1e-4`**  
  Initial learning rate. A good starting point for T5 models with LoRA.

- **`num_train_epochs=25`**  
  Total number of training epochs. This is relatively high to ensure convergence for our use case.

- **`logging_steps=10`**  
  How often (in steps) to log training metrics to the console and W&B.

- **`eval_strategy="steps"`**  
  Run evaluation every `eval_steps` instead of after every epoch.

- **`eval_steps=50`**  
  Evaluate the model every 50 steps to monitor progress during training.

- **`save_steps=50`**  
  Save model checkpoints every 50 steps for redundancy and safe restoration.

- **`save_total_limit=2`**  
  Keep only the 2 most recent model checkpoints to save disk space.

- **`fp16=True`**  
  Enable mixed precision training (faster and memory-efficient on supported GPUs).

- **`push_to_hub=True`**  
  Automatically push the trained model to the Hugging Face Hub.

- **`hub_model_id="zamal/github-tag-generatorr"`**  
  The model repo name on Hugging Face under your username. This is where checkpoints and final model weights will be pushed.

- **`hub_token=os.environ['HUGGINGFACE_TOKEN']`**  
  Token to authenticate your Hugging Face account. We securely retrieve this from the environment.

This setup ensures a balance between training efficiency, frequent monitoring, and safe saving of model progress.



```python
training_args = TrainingArguments(
    output_dir="./t5_tag_generator",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,
    num_train_epochs=25,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    push_to_hub=True,
    hub_model_id="zamal/github-tag-generatorr",  # Replace with your Hugging Face username
    hub_token=os.environ['HUGGINGFACE_TOKEN']
)

```

#### 🧠 Initialize the Trainer

We now configure the `Trainer`, which abstracts away the training loop, evaluation steps, logging, and saving. It handles all of it for us using the parameters we've defined earlier.

We also pass in the `DataCollatorForSeq2Seq`, which ensures proper padding and batching during training and evaluation for sequence-to-sequence tasks like ours.

#### ⚠️ Warnings Explained:

- **`FutureWarning: 'tokenizer' is deprecated...`**  
  As of Transformers v5.0.0, the `tokenizer` argument in `Trainer` is deprecated. Instead, Hugging Face recommends using the `processing_class`, which refers to a processor that combines tokenization and potentially feature extraction. For now, it's safe to ignore this, but it's good practice to track deprecations for future compatibility.

- **`No label_names provided for model class 'PeftModelForSeq2SeqLM'`**  
  This warning appears because we’re using a [PEFT (Parameter-Efficient Fine-Tuning)](https://huggingface.co/docs/peft) wrapped model (`PeftModelForSeq2SeqLM`), and the `Trainer` cannot automatically determine the label field names in this case.  
  Since we're already formatting our dataset correctly (by explicitly setting `labels` during preprocessing), this warning can be safely ignored as well — training will still proceed correctly.

Now, we can initialize our `Trainer`:



```python
from transformers import Trainer
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
    )

```

    <ipython-input-16-9e14871e3c2a>:5: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
      trainer = Trainer(
    No label_names provided for model class `PeftModelForSeq2SeqLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.


#### 🚀 Start Training the Tag Generator Model

With everything set up — the model, tokenizer, dataset, LoRA configuration, training arguments, and the `Trainer` — we can now kick off the fine-tuning process by calling `trainer.train()`.

This will:
- Fine-tune our **T5 model** using the **parameter-efficient LoRA strategy**.
- Save checkpoints at regular intervals (`save_steps=50`).
- Evaluate on the validation set every 50 steps (`eval_steps=50`).
- Log metrics like loss to **Weights & Biases** or the Hugging Face Hub if integrated.

Training will take some time depending on the size of your dataset and GPU, but you’ll start to see metrics printed out step-by-step, such as:

- `Training Loss`: how well the model is fitting the training data.
- `Validation Loss`: how well the model performs on unseen data.

Let’s begin the fine-tuning! 👇



```python
trainer.train()
```

    wandb: WARNING The `run_name` is currently set to the same value as `TrainingArguments.output_dir`. If this was not intended, please specify a different run name by setting the `TrainingArguments.run_name` parameter.
    wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
    wandb: You can find your API key in your browser here: https://wandb.ai/authorize?ref=models
    wandb: Paste an API key from your profile and hit enter:
    wandb: WARNING If you're specifying your api key in code, ensure this code is not shared publicly.
    wandb: WARNING Consider setting the WANDB_API_KEY environment variable, or running `wandb login` from the command line.
    wandb: No netrc file found, creating one.
    wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc
    wandb: Currently logged in as: zamalbabar9866 (zamalbabar9866-fau-erlangen-n-rnberg) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
    Tracking run with wandb version 0.19.11
    Run data is saved locally in <code>/content/wandb/run-20250525_215048-3uv5wis6</code>
    Syncing run <strong><a href='https://wandb.ai/zamalbabar9866-fau-erlangen-n-rnberg/huggingface/runs/3uv5wis6' target="_blank">./t5_tag_generator</a></strong> to <a href='https://wandb.ai/zamalbabar9866-fau-erlangen-n-rnberg/huggingface' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>
    View project at <a href='https://wandb.ai/zamalbabar9866-fau-erlangen-n-rnberg/huggingface' target="_blank">https://wandb.ai/zamalbabar9866-fau-erlangen-n-rnberg/huggingface</a>
    View run at <a href='https://wandb.ai/zamalbabar9866-fau-erlangen-n-rnberg/huggingface/runs/3uv5wis6' target="_blank">https://wandb.ai/zamalbabar9866-fau-erlangen-n-rnberg/huggingface/runs/3uv5wis6</a>
    /usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py:741: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /pytorch/torch/csrc/utils/tensor_new.cpp:254.)
      batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
    Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. You should pass an instance