# Prompt Tuning with PEFT: A Practical Guide

## Introduction

This guide demonstrates how to apply prompt tuning using the PEFT (Parameter-Efficient Fine-Tuning) library. Prompt tuning is an additive fine-tuning technique where you train additional prompt-related layers without modifying the original model's weights. This approach dramatically reduces trainable parameters, preserves the base model's knowledge, and enables efficient multi-task serving.

We'll train two specialized models from a single base model using different datasets, then compare their outputs before and after training.

## Prerequisites

First, install the required libraries:

```bash
pip install peft==0.8.2 datasets==2.14.5
```

## Step 1: Import Libraries and Load Base Model

Import the necessary components from Transformers and PEFT:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit, PeftModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import os
```

Choose your base model. We'll use a smaller Bloom model for faster training:

```python
model_name = "bigscience/bloomz-560m"
NUM_VIRTUAL_TOKENS = 4
NUM_EPOCHS = 6

tokenizer = AutoTokenizer.from_pretrained(model_name)
foundational_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
)
```

## Step 2: Define Helper Function for Model Inference

Create a function to generate text from the model:

```python
def get_outputs(model, inputs, max_new_tokens=100):
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.5,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id
    )
    return outputs
```

## Step 3: Test Base Model Performance

Let's see how the base model responds to our target prompts before fine-tuning:

```python
# Test with motivational coach prompt
input_prompt = tokenizer("I want you to act as a motivational coach. ", return_tensors="pt")
foundational_outputs_prompt = get_outputs(foundational_model, input_prompt, max_new_tokens=50)
print("Base model response to coach prompt:")
print(tokenizer.batch_decode(foundational_outputs_prompt, skip_special_tokens=True)[0])

# Test with inspirational sentence starter
input_sentences = tokenizer("There are two nice things that should matter to you:", return_tensors="pt")
foundational_outputs_sentence = get_outputs(foundational_model, input_sentences, max_new_tokens=50)
print("\nBase model response to sentence starter:")
print(tokenizer.batch_decode(foundational_outputs_sentence, skip_special_tokens=True)[0])
```

The base model produces reasonable but generic responses. We'll train it to produce more specialized outputs.

## Step 4: Prepare Training Datasets

We'll use two different datasets for our two specialized models:

```python
# Load and prepare the ChatGPT prompts dataset
dataset_prompt = "fka/awesome-chatgpt-prompts"
data_prompt = load_dataset(dataset_prompt)
data_prompt = data_prompt.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
train_sample_prompt = data_prompt["train"].select(range(50))

# Load and prepare the inspirational quotes dataset
dataset_sentences = load_dataset("Abirate/english_quotes")
data_sentences = dataset_sentences.map(lambda samples: tokenizer(samples["quote"]), batched=True)
train_sample_sentences = data_sentences["train"].select(range(25))
train_sample_sentences = train_sample_sentences.remove_columns(['author', 'tags'])
```

## Step 5: Configure PEFT for Prompt Tuning

Set up the prompt tuning configuration. This defines how the virtual prompt tokens will be initialized and trained:

```python
generation_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.RANDOM,
    num_virtual_tokens=NUM_VIRTUAL_TOKENS,
    tokenizer_name_or_path=model_name
)
```

## Step 6: Create PEFT Models

Create two PEFT models from the same base model:

```python
peft_model_prompt = get_peft_model(foundational_model, generation_config)
peft_model_sentences = get_peft_model(foundational_model, generation_config)

print("Trainable parameters for prompt model:")
print(peft_model_prompt.print_trainable_parameters())

print("\nTrainable parameters for sentences model:")
print(peft_model_sentences.print_trainable_parameters())
```

Notice the dramatic reduction in trainable parameters - we're only training about 0.0007% of the total parameters!

## Step 7: Set Up Training Configuration

Define helper functions for creating training arguments and trainers:

```python
def create_training_arguments(path, learning_rate=0.0035, epochs=6):
    training_args = TrainingArguments(
        output_dir=path,
        use_cpu=True,
        auto_find_batch_size=True,
        learning_rate=learning_rate,
        num_train_epochs=epochs
    )
    return training_args

def create_trainer(model, training_args, train_dataset):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    return trainer
```

Create output directories and training arguments:

```python
working_dir = "./"
output_directory_prompt = os.path.join(working_dir, "peft_outputs_prompt")
output_directory_sentences = os.path.join(working_dir, "peft_outputs_sentences")

# Create directories if they don't exist
for directory in [working_dir, output_directory_prompt, output_directory_sentences]:
    if not os.path.exists(directory):
        os.mkdir(directory)

training_args_prompt = create_training_arguments(output_directory_prompt, 0.003, NUM_EPOCHS)
training_args_sentences = create_training_arguments(output_directory_sentences, 0.003, NUM_EPOCHS)
```

## Step 8: Train Both Models

Train each model with its respective dataset:

```python
# Train the prompt model
trainer_prompt = create_trainer(peft_model_prompt, training_args_prompt, train_sample_prompt)
trainer_prompt.train()

# Train the sentences model
trainer_sentences = create_trainer(peft_model_sentences, training_args_sentences, train_sample_sentences)
trainer_sentences.train()
```

## Step 9: Save the Trained Models

Save both trained PEFT models:

```python
trainer_prompt.model.save_pretrained(output_directory_prompt)
trainer_sentences.model.save_pretrained(output_directory_sentences)
```

## Step 10: Load and Test the Fine-Tuned Models

Load the first fine-tuned model and test it:

```python
# Load the prompt-tuned model
loaded_model_prompt = PeftModel.from_pretrained(
    foundational_model,
    output_directory_prompt,
    is_trainable=False
)

# Test with the original prompt
loaded_model_prompt_outputs = get_outputs(loaded_model_prompt, input_prompt)
print("Fine-tuned model response to coach prompt:")
print(tokenizer.batch_decode(loaded_model_prompt_outputs, skip_special_tokens=True)[0])
```

Compare the results:
- **Base Model:** "I want you to act as a motivational coach. Don't be afraid of being challenged."
- **Fine-Tuned Model:** "I want you to act as a motivational coach. You will be helping students learn how they can improve their performance in the classroom and at school."

## Step 11: Load Multiple Adapters for Multi-Task Serving

One powerful feature of PEFT is loading multiple adapters into the same base model:

```python
# Load the second adapter into the same model
loaded_model_prompt.load_adapter(output_directory_sentences, adapter_name="quotes")
loaded_model_prompt.set_adapter("quotes")

# Test with the second prompt
loaded_model_sentences_outputs = get_outputs(loaded_model_prompt, input_sentences)
print("\nFine-tuned model response to sentence starter:")
print(tokenizer.batch_decode(loaded_model_sentences_outputs, skip_special_tokens=True)[0])
```

Compare the results:
- **Base Model:** "There are two nice things that should matter to you: the price and quality of your product."
- **Fine-Tuned Model:** "There are two nice things that should matter to you: the weather and your health."

## Conclusion

Prompt tuning with PEFT offers several advantages:

1. **Parameter Efficiency:** Train only a tiny fraction (0.0007%) of the model's parameters
2. **Knowledge Preservation:** The base model's original knowledge remains intact
3. **Multi-Task Serving:** Load multiple specialized adapters into a single base model
4. **Fast Training:** Train specialized models in minutes rather than hours
5. **Cost Effective:** Significantly reduced computational requirements

You can experiment with different base models from the Bloom family, adjust the number of virtual tokens, or modify training epochs to achieve different results. This technique is particularly valuable when you need to serve multiple specialized tasks efficiently while maintaining a single foundational model in memory.