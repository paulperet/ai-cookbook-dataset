# Fine-tuning a Code LLM on Custom Code with a Single GPU

## Overview
Publicly available code LLMs like Codex, StarCoder, and Code Llama excel at general programming tasks but often lack awareness of organizational conventions or proprietary libraries. This guide demonstrates how to fine-tune a code LLM on private codebases to enhance its contextual awareness while optimizing resource usage to fit on a single GPU.

We'll use Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically LoRA (Low-Rank Adaptation), combined with 4-bit quantization to make this feasible with limited hardware.

## Prerequisites

First, install the required libraries:

```bash
pip install -q transformers datasets peft bitsandbytes flash-attn
```

If using a gated model like `bigcode/starcoderbase-1b`, authenticate with Hugging Face:

```python
from huggingface_hub import notebook_login

notebook_login()
```

## Configuration

Define your training configuration. These parameters can be adjusted based on your needs:

```python
# Model and Dataset
MODEL = "bigcode/starcoderbase-1b"  # Model checkpoint on Hugging Face Hub
DATASET = "smangrul/hf-stack-v1"    # Dataset on Hugging Face Hub
DATA_COLUMN = "content"             # Column containing code content

# Sequence and Training Parameters
SEQ_LENGTH = 2048                   # Sequence length
MAX_STEPS = 2000                    # Maximum training steps
BATCH_SIZE = 16                     # Batch size
GR_ACC_STEPS = 1                    # Gradient accumulation steps
LR = 5e-4                           # Learning rate
LR_SCHEDULER_TYPE = "cosine"        # Learning rate scheduler
WEIGHT_DECAY = 0.01                 # Weight decay
NUM_WARMUP_STEPS = 30               # Warmup steps
EVAL_FREQ = 100                     # Evaluation frequency
SAVE_FREQ = 100                     # Save frequency
LOG_FREQ = 25                       # Logging frequency
OUTPUT_DIR = "peft-starcoder-lora-a100"  # Output directory
BF16 = True                         # Use bfloat16
FP16 = False                        # Disable float16

# FIM (Fill-in-the-Middle) Transformations
FIM_RATE = 0.5                      # Proportion of samples with FIM
FIM_SPM_RATE = 0.5                  # Proportion of FIM using SPM variant

# LoRA Configuration
LORA_R = 8                          # LoRA rank
LORA_ALPHA = 32                     # LoRA alpha
LORA_DROPOUT = 0.0                  # LoRA dropout
LORA_TARGET_MODULES = "c_proj,c_attn,q_attn,c_fc,c_proj"  # Target modules

# Quantization Configuration
USE_NESTED_QUANT = True             # Use nested quantization
BNB_4BIT_COMPUTE_DTYPE = "bfloat16" # Compute dtype for 4-bit

SEED = 0                            # Random seed
```

Set the random seed for reproducibility:

```python
from transformers import set_seed

set_seed(SEED)
```

## Step 1: Prepare the Dataset

### 1.1 Load the Dataset with Streaming
We'll use the Hugging Face dataset containing code from top repositories. Streaming mode loads data progressively to avoid downloading the entire dataset at once:

```python
from datasets import load_dataset

dataset = load_dataset(
    DATASET,
    data_dir="data",
    split="train",
    streaming=True,
)

# Split into validation and training sets
valid_data = dataset.take(4000)
train_data = dataset.skip(4000)
train_data = train_data.shuffle(buffer_size=5000, seed=SEED)
```

### 1.2 Estimate Character-to-Token Ratio
Calculate the average characters per token to help with buffer sizing:

```python
from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())
    
    return total_characters / total_tokens

chars_per_token = chars_token_ratio(train_data, tokenizer, DATA_COLUMN)
print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
```

A ratio between 2.0 and 3.5 indicates good tokenization for code.

### 1.3 Define FIM Transformations (Optional)
FIM (Fill-in-the-Middle) transformations teach the model to infill code, not just generate left-to-right:

```python
import functools
import numpy as np

@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    try:
        FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = tokenizer.special_tokens_map["additional_special_tokens"][1:5]
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            tokenizer.vocab[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
        )
    except KeyError:
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = None, None, None, None
    return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id

def permute(
    sample,
    np_rng,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    fim_rate=0.5,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
):
    """
    Apply FIM transformation to a token sequence with probability fim_rate.
    """
    if np_rng.binomial(1, fim_rate):
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()
        
        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0] : boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1] :], dtype=np.int64)
        
        if truncate_or_pad:
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - len(sample)
            
            if diff > 0:
                if suffix.shape[0] <= diff:
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])
        
        # SPM variant: suffix, prefix, middle
        if np_rng.binomial(1, fim_spm_rate):
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        # PSM variant: prefix, suffix, middle
        else:
            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        new_sample = sample
    
    return list(new_sample), np_rng
```

Set `FIM_RATE = 0` if you want to skip FIM transformations.

### 1.4 Create Constant-Length Dataset
Create an iterable dataset that yields fixed-length token sequences:

```python
from torch.utils.data import IterableDataset
import random

class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset returning constant-length token chunks from text streams.
    """
    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed
        
        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = get_fim_token_ids(self.tokenizer)
        if not self.suffix_tok_id and self.fim_rate > 0:
            print("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0
    
    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        np_rng = np.random.RandomState(seed=self.seed)
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            
            for tokenized_input in tokenized_inputs:
                if self.fim_rate > 0:
                    tokenized_input, np_rng = permute(
                        tokenized_input,
                        np_rng,
                        self.suffix_tok_id,
                        self.prefix_tok_id,
                        self.middle_tok_id,
                        self.pad_tok_id,
                        fim_rate=self.fim_rate,
                        fim_spm_rate=self.fim_spm_rate,
                        truncate_or_pad=False,
                    )
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            
            random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }

# Create training and evaluation datasets
train_dataset = ConstantLengthDataset(
    tokenizer,
    train_data,
    infinite=True,
    seq_length=SEQ_LENGTH,
    chars_per_token=chars_per_token,
    content_field=DATA_COLUMN,
    fim_rate=FIM_RATE,
    fim_spm_rate=FIM_SPM_RATE,
    seed=SEED,
)

eval_dataset = ConstantLengthDataset(
    tokenizer,
    valid_data,
    infinite=False,
    seq_length=SEQ_LENGTH,
    chars_per_token=chars_per_token,
    content_field=DATA_COLUMN,
    fim_rate=FIM_RATE,
    fim_spm_rate=FIM_SPM_RATE,
    seed=SEED,
)
```

## Step 2: Prepare the Model

### 2.1 Configure Quantization
We'll use 4-bit quantization via `bitsandbytes` to reduce memory usage:

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import torch

load_in_8bit = False
compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=USE_NESTED_QUANT,
)
```

### 2.2 Load the Quantized Model
Load the model with the quantization configuration:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

### 2.3 Configure LoRA
Apply Parameter-Efficient Fine-Tuning with LoRA:

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES.split(","),
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

## Step 3: Configure Training

### 3.1 Set Training Arguments
Define the training configuration:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    dataloader_drop_last=True,
    evaluation_strategy="steps",
    max_steps=MAX_STEPS,
    eval_steps=EVAL_FREQ,
    save_steps=SAVE_FREQ,
    logging_steps=LOG_FREQ,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_steps=NUM_WARMUP_STEPS,
    gradient_accumulation_steps=GR_ACC_STEPS,
    gradient_checkpointing=True,
    fp16=FP16,
    bf16=BF16,
    weight_decay=WEIGHT_DECAY,
    run_name="starcoder-lora-finetune",
    report_to="none",
    ddp_find_unused_parameters=False,
)
```

### 3.2 Initialize Trainer
Create the Trainer instance:

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

## Step 4: Train the Model

### 4.1 Start Training
Begin the fine-tuning process:

```python
trainer.train()
```

### 4.2 Save the Model
Save the trained model and tokenizer:

```python
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
```

## Step 5: Evaluate and Use

### 5.1 Run Evaluation
Evaluate the model on the validation set:

```python
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
```

### 5.2 Generate Code
Test the fine-tuned model with code generation:

```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
)

prompt = "def fibonacci(n):"
result = generator(prompt, max_length=100, temperature=0.7)
print(result[0]['generated_text'])
```

## Conclusion

You've successfully fine-tuned a code LLM on custom code using a single GPU! By combining 4-bit quantization with LoRA, you've made efficient use of resources while adapting the model to your specific codebase. The fine-tuned model should now better understand your organization's coding conventions and proprietary libraries.

**Key takeaways:**
- Streaming datasets handle large codebases efficiently
- FIM transformations enable infilling capabilities
- 4-bit quantization significantly reduces memory requirements
- LoRA provides parameter-efficient fine-tuning
- The approach works on a single GPU with sufficient RAM

Experiment with different hyperparameters, try other base models, or adjust the FIM rate to further optimize performance for your use case.