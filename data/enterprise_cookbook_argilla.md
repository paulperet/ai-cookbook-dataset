# Data Annotation with Argilla Spaces: Evaluating LLM Code Generation

_Authored by: [Moritz Laurer](https://huggingface.co/MoritzLaurer)_

This guide demonstrates a systematic workflow for evaluating LLM outputs and creating high-quality training data. You can use this process to first assess the zero-shot performance of your favorite LLM on a specific task. If performance needs improvement, you can then adapt the same workflow to generate and annotate training data efficiently.

**Example Use Case:** In this tutorial, we focus on **code generation**. We will create test and training data by comparing outputs from two different LLMs. This workflow is flexible and can be adapted to any other text-generation task relevant to your project.

**What you will accomplish:**
1.  Download a dataset containing code generation tasks.
2.  Prompt two different LLMs to generate code for these tasks, creating synthetic data for evaluation.
3.  Create an annotation interface on Hugging Face Spaces using Argilla.
4.  Upload the task prompts and LLM responses to the Argilla interface for human evaluation.
5.  Download the annotated data for analysis or training.

You can customize each step, such as swapping the LLM provider or modifying the annotation criteria.

## Prerequisites & Setup

First, install the required Python libraries and connect to the Hugging Face Hub.

```bash
pip install argilla~=2.0.0
pip install transformers~=4.40.0
pip install datasets~=2.19.0
pip install huggingface_hub~=0.23.2
```

```python
import huggingface_hub

# Login to the HF Hub. This method helps avoid storing your token in plain text.
import subprocess
subprocess.run(["git", "config", "--global", "credential.helper", "store"])
huggingface_hub.login(add_to_git_credential=True)
```

## Step 1: Download Example Task Data

We'll start by downloading a dataset containing code generation instructions. We'll use a small sample from the `bigcode/self-oss-instruct-sc2-exec-filter-50k` dataset, which was used to train the StarCoder2-Instruct model.

```python
from datasets import load_dataset

# Load a small sample for faster testing
dataset_codetask = load_dataset("bigcode/self-oss-instruct-sc2-exec-filter-50k", split="train[:3]")
print("Dataset structure:")
print(dataset_codetask, "\n")

# Extract the instruction/prompt text
instructions_lst = dataset_codetask["instruction"]
print("Example instructions:")
print(instructions_lst[:2])
```

## Step 2: Generate Code with Two LLMs

We will prompt two different LLMs to generate code based on the downloaded instructions. This creates the synthetic outputs we want to evaluate.

### 2.1 Format Instructions with Chat Templates

Before sending instructions to an LLM API, we must format them correctly using the model's specific `chat_template`. This involves wrapping the instruction with special tokens.

```python
from transformers import AutoTokenizer

# Define the two models we want to compare
models_to_compare = ["mistralai/Mixtral-8x7B-Instruct-v0.1", "meta-llama/Meta-Llama-3-70B-Instruct"]

def format_prompt(prompt, tokenizer):
    """Formats a single instruction using the tokenizer's chat template."""
    messages = [{"role": "user", "content": prompt}]
    # `tokenize=False` returns a string, not tensors
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted_prompt

# Format all instructions for each model
prompts_formatted_dic = {}
for model in models_to_compare:
    tokenizer = AutoTokenizer.from_pretrained(model)
    formatted_prompts = [format_prompt(instr, tokenizer) for instr in instructions_lst]
    prompts_formatted_dic[model] = formatted_prompts

# Inspect the formatted prompts
print(f"First prompt formatted for {models_to_compare[0]}:\n")
print(prompts_formatted_dic[models_to_compare[0]][0], "\n\n")
print(f"First prompt formatted for {models_to_compare[1]}:\n")
print(prompts_formatted_dic[models_to_compare[1]][0])
```

### 2.2 Define Generation Parameters

We'll configure parameters for the text generation to ensure consistent, high-quality outputs. Hugging Face's Inference API uses Text Generation Inference (TGI). The parameters below reduce creativity to favor more probable, correct code.

```python
generation_params = {
    "temperature": 0.2,        # Low temperature for less random outputs
    "top_p": 0.60,             # Nucleus sampling parameter
    "top_k": None,
    "repetition_penalty": 1.0,
    "do_sample": True,
    "max_new_tokens": 512 * 2, # Limit response length
    "return_full_text": False, # Return only the generated part
    "seed": 42,                # For reproducibility
    "max_time": None,
    "stream": False,
    "use_cache": False,
    "wait_for_model": False,
}
```

### 2.3 Query the Hugging Face Inference API

Now, we send the formatted prompts to the Hugging Face Inference API to get the LLM-generated code. The code below uses the serverless API, which is rate-limited and intended for testing. For production, consider using [Dedicated Endpoints](https://huggingface.co/docs/inference-endpoints).

> **Tip:** For faster processing, use asynchronous API calls and dedicated endpoints.

```python
import requests
from tqdm.auto import tqdm

def query(payload=None, api_url=None, headers=None):
    """Helper function to send a POST request to the HF Inference API."""
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

# Set up authorization headers using your HF token
headers = {"Authorization": f"Bearer {huggingface_hub.get_token()}"}

# Query each model and collect responses
output_dic = {}
for model in models_to_compare:
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    formatted_prompts = prompts_formatted_dic[model]
    
    output_lst = []
    for prompt in tqdm(formatted_prompts, desc=f"Querying {model}"):
        response = query(
            payload={
                "inputs": prompt,
                "parameters": generation_params
            },
            api_url=api_url,
            headers=headers
        )
        # Extract the generated text from the response
        generated_text = response[0]["generated_text"]
        output_lst.append(generated_text)
    
    output_dic[model] = output_lst

# Examine the first generated response from each model
print(f"--- First generation from {models_to_compare[0]}:\n")
print(output_dic[models_to_compare[0]][0])
print("\n" + "="*80 + "\n")
print(f"--- First generation from {models_to_compare[1]}:\n")
print(output_dic[models_to_compare[1]][0])
```

### 2.4 Store Results in a Dataset

Let's organize the original instructions and the corresponding LLM responses into a Hugging Face Dataset for easy management.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "instructions": instructions_lst,
    "response_model_1": output_dic[models_to_compare[0]],
    "response_model_2": output_dic[models_to_compare[1]]
})

print(dataset)
```

## Step 3: Set Up Argilla for Annotation

We will use [Argilla](https://argilla.io/), a collaborative tool for building high-quality datasets, to evaluate the LLM outputs. You can run Argilla via a Hugging Face Space with minimal setup.

### 3.1 Create an Argilla Space

1.  Follow the [Argilla Space quickstart instructions](https://docs.argilla.io/latest/getting_started/quickstart/) to create your own Space.
2.  Once your Space is running, note its URL and your API key (found under **"My Settings > API key"** in the UI).

### 3.2 Connect to Your Argilla Space

Connect your notebook to the Argilla backend to configure datasets and upload data programmatically.

```python
import argilla as rg

# Replace with your Space's URL and API key
client = rg.Argilla(
    api_url="https://your-username-spacename.hf.space",  # For local: "http://localhost:6900"
    api_key="your-argilla-api-key-here",
    # If using a private Space, also pass your HF token
    headers={"Authorization": f"Bearer {huggingface_hub.get_token()}"},
)

# Verify the connection
user = client.me
print(user)
```

### 3.3 Define Annotation Guidelines

Clear, comprehensive guidelines are crucial for consistent and high-quality annotations. Good guidelines should be simple, explicit, and reproducible.

**Key Principles for Guidelines:**
*   **Simple & Clear:** Understandable by someone new to the task.
*   **Reproducible & Explicit:** All necessary information must be in the text.
*   **Short & Comprehensive:** Be as brief as possible while covering all requirements.

We'll use a **cumulative rating system** where annotators add points for specific criteria. This is often better than a simple Likert scale (e.g., 1-5) because it forces explicit quality judgments.

```python
annotator_guidelines = """Your task is to evaluate the responses of two LLMs to code generation tasks.

First, you need to score each response on a scale from 0 to 7. You add points to your final score based on the following criteria:
- Add up to +2 points, if the code is properly commented, with inline comments and doc strings for functions.
- Add up to +2 points, if the code contains a good example for testing.
- Add up to +3 points, if the code runs and works correctly. Copy the code into an IDE and test it with at least two different inputs. Attribute one point if the code is overall correct, but has some issues. Attribute three points if the code is fully correct and robust against different scenarios.
Your resulting final score can be any value between 0 to 7.

If both responses have a final score of <= 4, select one response and correct it manually in the text field.
The corrected response must fulfill all criteria from above.
"""

# A shorter tooltip for the rating interface
rating_tooltip = """- Add up to +2 points for proper comments and docstrings.
- Add up to +2 points for a good testing example.
- Add up to +3 points for correct, runnable code.
"""
```

### 3.4 Create and Configure the Argilla Dataset

Now, we define the structure of the dataset within Argilla. We specify the fields (instructions, model responses) and the annotation task (rating and correction).

```python
# Define the fields that will be displayed to annotators
fields = [
    rg.TextField(name="instruction", title="Code Generation Instruction"),
    rg.TextField(name="response_1", title="Response from Model 1 (Mixtral)"),
    rg.TextField(name="response_2", title="Response from Model 2 (Llama 3)"),
]

# Define the questions/tasks for annotators
questions = [
    rg.RatingQuestion(
        name="rating_response_1",
        title="Rate the response from Model 1 (Mixtral):",
        description=rating_tooltip,
        required=True,
        values=[0, 1, 2, 3, 4, 5, 6, 7]  # 0-7 scale
    ),
    rg.RatingQuestion(
        name="rating_response_2",
        title="Rate the response from Model 2 (Llama 3):",
        description=rating_tooltip,
        required=True,
        values=[0, 1, 2, 3, 4, 5, 6, 7]
    ),
    rg.TextQuestion(
        name="corrected_response",
        title="If both ratings <=4, provide a corrected version here:",
        required=False,
        use_markdown=True
    )
]

# Create the dataset settings
dataset_settings = rg.DatasetSettings(
    fields=fields,
    questions=questions,
    guidelines=annotator_guidelines
)

# Create the dataset in Argilla (or update an existing one)
try:
    # Try to delete if it exists to start fresh
    rg.delete("llm_code_evaluation")
except Exception:
    pass

# Create the new dataset
dataset_rg = rg.FeedbackDataset(
    fields=fields,
    questions=questions,
    guidelines=annotator_guidelines
)

# Push the dataset configuration to Argilla
dataset_rg.push_to_argilla(name="llm_code_evaluation", workspace="admin")
print("Argilla dataset 'llm_code_evaluation' created successfully.")
```

## Step 4: Upload Data for Annotation

With the dataset configured, we can now upload our instructions and LLM responses as records for annotators to evaluate.

```python
# Prepare records from our earlier dataset
records = []
for i, row in enumerate(dataset):
    record = rg.FeedbackRecord(
        fields={
            "instruction": row["instructions"],
            "response_1": row["response_model_1"],
            "response_2": row["response_model_2"],
        },
        # Optional: add metadata like the dataset ID
        metadata={"dataset_id": i}
    )
    records.append(record)

# Upload records to Argilla
dataset_rg.add_records(records)
print(f"Uploaded {len(records)} records to Argilla.")
```

## Step 5: Download Annotated Data

After annotators have completed their evaluations in the Argilla UI, you can download the annotated data for analysis or to use as training data.

```python
# Fetch the annotated dataset from Argilla
annotated_dataset = rg.FeedbackDataset.from_argilla("llm_code_evaluation", workspace="admin")

# Convert to a format suitable for training (e.g., pandas DataFrame)
import pandas as pd

data_for_export = []
for record in annotated_dataset.records:
    # Check if the record has responses (annotations)
    if record.responses:
        # Assuming one annotator per record for simplicity
        resp = record.responses[0]
        data_for_export.append({
            "instruction": record.fields["instruction"],
            "response_1": record.fields["response_1"],
            "response_2": record.fields["response_2"],
            "rating_1": resp.values["rating_response_1"].value,
            "rating_2": resp.values["rating_response_2"].value,
            "corrected_response": resp.values["corrected_response"].value if "corrected_response" in resp.values else None
        })

df_annotated = pd.DataFrame(data_for_export)
print(df_annotated.head())

# Save to a CSV file
df_annotated.to_csv("annotated_llm_responses.csv", index=False)
print("Annotated data saved to 'annotated_llm_responses.csv'.")
```

## Next Steps

You now have a complete pipeline for generating, evaluating, and annotating LLM outputs. Here’s how you can extend this workflow:

*   **Scale Up:** Increase the number of instructions and use asynchronous API calls or dedicated endpoints for faster generation.
*   **Different Tasks:** Adapt the instructions and annotation guidelines for other tasks like summarization, question answering, or creative writing.
*   **Fine-Tuning:** Use the high-rated or corrected responses as training data to fine-tune a model for improved performance on your specific task.
*   **Quality Analysis:** Use the annotation scores to perform inter-annotator agreement studies and refine your guidelines further.

This cookbook provides a robust foundation for building high-quality, human-in-the-loop datasets to power your AI applications.