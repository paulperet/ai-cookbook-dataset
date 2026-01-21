# Building a Documentation Chatbot with Meta Synthetic Data Kit

_Authored by: [Alan Ponnachan](https://huggingface.co/AlanPonnachan)_

This guide demonstrates a practical approach to building a domain-specific Question & Answering chatbot. We'll focus on creating a chatbot that can answer questions about a specific piece of documentation – in this case, LangChain's documentation on Chat Models.

**Goal:** To fine-tune a small, efficient Language Model (LLM) to understand and answer questions about the LangChain Chat Models documentation.

**Approach:**
1.  **Data Acquisition:** Obtain the text content from the target LangChain documentation page.
2.  **Synthetic Data Generation:** Use Meta's `synthetic-data-kit` to automatically generate Question/Answer pairs from this documentation.
3.  **Efficient Fine-tuning:** Employ Unsloth and Hugging Face's TRL SFTTrainer to efficiently fine-tune a Llama-3.2-3B model on the generated synthetic data.
4.  **Evaluation:** Test the fine-tuned model with specific questions about the documentation.

This method allows us to adapt an LLM to a niche domain without requiring a large, manually curated dataset.

**Hardware Used:** This notebook was run on Google Colab (Free Tier) with an NVIDIA T4 GPU.

---

## 1. Setup and Installation

First, we need to install the necessary libraries. We'll use `unsloth` for efficient model handling and training, and `synthetic-data-kit` for generating our training data.

```bash
pip install unsloth vllm==0.8.2
pip install synthetic-data-kit
```

If you are running this in a Google Colab environment, you may need to handle dependencies differently to avoid conflicts. The following cell manages this.

```python
import os
if "COLAB_" in "".join(os.environ.keys()):
    import sys, re, requests
    modules = list(sys.modules.keys())
    for x in modules:
        if "PIL" in x or "google" in x:
            sys.modules.pop(x)
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft "trl==0.15.2" triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf datasets huggingface_hub[hf_xet] hf_transfer

    # vLLM requirements - vLLM breaks Colab due to reinstalling numpy
    f = requests.get("https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/requirements/common.txt").content
    with open("vllm_requirements.txt", "wb") as file:
        file.write(re.sub(rb"(transformers|numpy|xformers|importlib_metadata)[^\n]{0,}\n", b"", f))
    !pip install -r vllm_requirements.txt
```

---

## 2. Synthetic Data Generation

We'll use `SyntheticDataKit` from Unsloth (which wraps Meta's `synthetic-data-kit`) to create Question/Answer pairs from our chosen documentation.

### 2.1 Initialize the Data Generator

First, initialize the `SyntheticDataKit` with the base model you intend to fine-tune. This generator will use the model to create synthetic Q&A pairs.

```python
from unsloth.dataprep import SyntheticDataKit

generator = SyntheticDataKit.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 2048,
)
```

### 2.2 Load the Documentation

Next, we need the raw text from the documentation we want the chatbot to learn. For this example, we'll use the LangChain Chat Models documentation. You can load text from a local file or fetch it from a URL.

```python
# Example: Loading documentation from a URL
import requests
from bs4 import BeautifulSoup

url = "https://python.langchain.com/docs/modules/model_io/chat/"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract the main text content (this selector may need adjustment)
text_content = soup.get_text(separator='\n', strip=True)
print(f"Loaded {len(text_content)} characters of documentation.")
```

### 2.3 Generate Synthetic Q&A Pairs

Now, use the generator to create a dataset of questions and answers based on the loaded documentation. The `generate` method will process the text in chunks and produce synthetic examples.

```python
# Generate synthetic data
synthetic_dataset = generator.generate(
    text=text_content,
    num_samples=100,  # Number of Q&A pairs to generate
    chunk_size=512,   # Size of text chunks to process
)

# The dataset is a list of dictionaries with 'question' and 'answer' keys
print(f"Generated {len(synthetic_dataset)} synthetic Q&A pairs.")
print("Example:")
print(synthetic_dataset[0])
```

The output will be a list of dictionaries, each containing a `question` and `answer` key. This synthetic dataset is now ready for the fine-tuning step.

---

## 3. Prepare Data for Fine-Tuning

Before training, we need to format the synthetic dataset into the instruction-following structure expected by the model.

### 3.1 Format the Dataset

We'll create a prompt template that instructs the model to answer questions based on the documentation.

```python
def format_instruction(example):
    # Create a prompt in a chat-like format
    prompt = f"""You are a helpful assistant specialized in LangChain Chat Models documentation.
Based on the provided documentation, answer the following question.

Question: {example['question']}

Answer: {example['answer']}"""
    return {"text": prompt}

# Apply formatting to the entire dataset
formatted_dataset = [format_instruction(ex) for ex in synthetic_dataset]
```

### 3.2 Split into Training and Validation Sets

It's good practice to hold out a portion of the data for validation to monitor training performance.

```python
from sklearn.model_selection import train_test_split

train_data, val_data = train_test_split(formatted_dataset, test_size=0.1, random_state=42)
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
```

---

## 4. Fine-Tune the Model

Now we'll fine-tune the Llama-3.2-3B-Instruct model on our synthetic dataset using Unsloth and TRL's `SFTTrainer`.

### 4.1 Load the Model and Tokenizer with Unsloth

Unsloth provides optimized versions of models for faster and more memory-efficient training.

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 2048,
    dtype = None, # Auto-detect
    load_in_4bit = True, # Use 4-bit quantization to reduce memory usage
)
```

### 4.2 Prepare the Model for LoRA Fine-Tuning

We'll use LoRA (Low-Rank Adaptation) to fine-tune the model efficiently, updating only a small subset of parameters.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,           # LoRA rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_rslora = False,
    loftq_config = None,
)
```

### 4.3 Set Up the Trainer

Configure the `SFTTrainer` with our training arguments.

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_data,
    eval_dataset = val_data,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
        report_to = "none", # Disable external logging for simplicity
    ),
)
```

### 4.4 Train the Model

Start the fine-tuning process.

```python
trainer_stats = trainer.train()
```

---

## 5. Evaluate the Fine-Tuned Model

After training, let's test the model with some specific questions about the LangChain Chat Models documentation.

### 5.1 Create an Inference Function

First, create a helper function to generate answers from the model.

```python
def ask_model(question, model, tokenizer):
    prompt = f"""You are a helpful assistant specialized in LangChain Chat Models documentation.
Based on the provided documentation, answer the following question.

Question: {question}

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part after the prompt
    answer = answer.split("Answer:")[-1].strip()
    return answer
```

### 5.2 Test with Sample Questions

Now, ask the model questions that should be answerable based on the documentation it was trained on.

```python
test_questions = [
    "What is a LangChain Chat Model?",
    "How do you initialize a ChatOpenAI model?",
    "What is the difference between ChatModels and LLMs in LangChain?",
]

for q in test_questions:
    answer = ask_model(q, model, tokenizer)
    print(f"Q: {q}")
    print(f"A: {answer}\n")
```

The model should now provide accurate, context-specific answers based on the LangChain documentation it was fine-tuned on.

---

## 6. Save and Share the Model (Optional)

If you want to save your fine-tuned model for later use or share it on the Hugging Face Hub, you can do so easily.

### 6.1 Save Locally

```python
model.save_pretrained("langchain_chat_model_assistant")
tokenizer.save_pretrained("langchain_chat_model_assistant")
```

### 6.2 Push to Hugging Face Hub

```python
from huggingface_hub import login
login() # You'll need to set your HF token first

model.push_to_hub("your-username/langchain-chat-model-assistant", tokenizer=tokenizer)
```

---

## Conclusion

You have successfully built a domain-specific documentation chatbot by:
1.  Generating a synthetic Q&A dataset from raw documentation using Meta's Synthetic Data Kit.
2.  Fine-tuning a Llama-3.2-3B model efficiently with Unsloth and LoRA.
3.  Evaluating the model's ability to answer questions within its specialized domain.

This approach demonstrates how to adapt a general-purpose LLM to a specific knowledge base without manual data labeling, making it a powerful technique for creating specialized AI assistants.