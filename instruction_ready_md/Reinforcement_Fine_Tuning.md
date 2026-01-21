# Reinforcement Fine-Tuning with Model Graders: A Practical Guide

This guide walks through applying Reinforcement Fine-Tuning (RFT) to the OpenAI `o4-mini` reasoning model using a medical dataset. You'll learn how to benchmark a base model, define a custom grader, run the RFT training job, and evaluate the fine-tuned model.

## Prerequisites

Ensure you have the required Python packages installed:

```bash
pip install openai datasets rapidfuzz tqdm
```

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Step 1: Load and Prepare the Dataset

We'll use a subset of the `medical-o1-verifiable-problem` dataset from Hugging Face, which contains clinical vignettes with questions and verified answers.

```python
import re
import json
import random
from datasets import load_dataset

# Load the dataset
ds = load_dataset("FreedomIntelligence/medical-o1-verifiable-problem")

def is_age_question(sample):
    """Filter samples that start with age descriptions like 'A 88-year-old'"""
    question = sample.get('Open-ended Verifiable Question', '')
    return re.match(r"^(A|An) \d{1,2}-year-old", question) is not None

# Apply the filter
filtered_samples = [s for s in ds["train"] if is_age_question(s)]
print(f"Filtered samples: {len(filtered_samples)}")
```

Next, split the data into training and test sets:

```python
# Set seed for reproducibility
random.seed(42)

# Randomly select 100 training samples
train_samples = random.sample(filtered_samples, min(100, len(filtered_samples)))

# Remove training samples to avoid overlap
remaining_samples = [s for s in filtered_samples if s not in train_samples]

# Randomly select 100 test samples
test_samples = random.sample(remaining_samples, min(100, len(remaining_samples)))

print(f"Training samples: {len(train_samples)}")
print(f"Test samples: {len(test_samples)}")
```

Standardize the answer format by converting all ground-truth answers to lowercase:

```python
for sample in train_samples:
    if 'Ground-True Answer' in sample and isinstance(sample['Ground-True Answer'], str):
        sample['Ground-True Answer'] = sample['Ground-True Answer'].lower()

for sample in test_samples:
    if 'Ground-True Answer' in sample and isinstance(sample['Ground-True Answer'], str):
        sample['Ground-True Answer'] = sample['Ground-True Answer'].lower()
```

Convert the samples to JSONL format, which is required by the RFT API:

```python
def convert_to_jsonl_format(samples, filename):
    """Convert samples to the JSONL format expected by the RFT API"""
    with open(filename, "w") as f:
        for sample in samples:
            user_content = sample.get("Open-ended Verifiable Question", "")
            reference_answer = sample.get("Ground-True Answer", "")
            json_obj = {
                "messages": [
                    {"role": "user", "content": user_content}
                ],
                "reference_answer": reference_answer
            }
            f.write(json.dumps(json_obj) + "\n")

def load_jsonl(filename):
    """Load samples from a JSONL file"""
    samples = []
    with open(filename, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

# Save datasets
convert_to_jsonl_format(train_samples, "data/medical_01_verifiable_problem_train.jsonl")
convert_to_jsonl_format(test_samples, "data/medical_01_verifiable_problem_val.jsonl")

# Load them back
train_samples_loaded = load_jsonl("data/medical_01_verifiable_problem_train.jsonl")
test_samples_loaded = load_jsonl("data/medical_01_verifiable_problem_val.jsonl")
```

## Step 2: Benchmark the Base Model

Before fine-tuning, establish a performance baseline. We'll create two grading functions: one for exact matches and another for fuzzy similarity.

```python
from rapidfuzz import fuzz, utils

def clinical_phrase_grader(sample: dict, item: dict) -> float:
    """Calculate fuzzy similarity between model output and reference answer"""
    score = fuzz.token_set_ratio(sample["output_text"], item["reference_answer"], 
                                 processor=utils.default_process)
    return score / 100.0

def clinical_phrase_binary_grader(sample: dict, item: dict) -> float:
    """Check for exact match between model output and reference answer"""
    return 1.0 if sample["output_text"] == item["reference_answer"] else 0.0

def combined_grader(sample: dict, item: dict, weights: list[float] = [0.85, 0.15]) -> float:
    """Combine fuzzy and exact grading with weighted scores"""
    clinical_phrase_score = clinical_phrase_grader(sample, item)
    binary_score = clinical_phrase_binary_grader(sample, item)
    return weights[0] * clinical_phrase_score + weights[1] * binary_score
```

The combined grader provides a balanced assessment: the fuzzy scorer (85% weight) accounts for semantic similarity, while the exact matcher (15% weight) rewards perfect responses.

Now, prepare the data with a system prompt:

```python
def prepend_system_prompt_to_first_user_message(samples, system_prompt, path=None):
    """Add a system prompt to the beginning of each conversation"""
    new_samples = []
    for sample in samples:
        sample_copy = json.loads(json.dumps(sample))
        messages = sample_copy.get("messages", [])
        if messages and messages[0].get("role") == "user" and isinstance(messages[0].get("content"), str):
            if not messages[0]["content"].startswith(system_prompt):
                messages[0]["content"] = f"{system_prompt}\n\n{messages[0]['content']}"
        new_samples.append(sample_copy)
    
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            for item in new_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return new_samples

# Define the system prompt
simple_prompt = """You are an expert clinician. For each clinical vignette, respond with exactly one phrase: the single most likely outcome or phenomenon, all in lowercase. 
- Do not add punctuation, articles, explanations, or commentary - output only the term itself.
- Sometimes, the expected answer can be a synonym of what you think.
- Use the standard clinical name (e.g. "thought withdrawal", "Toxoplasma encephalitis")."""

# Apply the prompt to both datasets
train_samples_loaded_simple_sys_prompt = prepend_system_prompt_to_first_user_message(
    train_samples_loaded, simple_prompt, path="data/medical_01_verifiable_problem_train_simple_prompt.jsonl"
)
test_samples_loaded_simple_sys_prompt = prepend_system_prompt_to_first_user_message(
    test_samples_loaded, simple_prompt, path="data/medical_01_verifiable_problem_val_simple_prompt.jsonl"
)
```

## Step 3: Generate Model Predictions

Create a function to generate predictions from the model:

```python
from openai import OpenAI
import concurrent.futures
from tqdm import tqdm
import os

client = OpenAI()

def generate_model_predictions(
    subset,
    prompt_type,
    model_name="o4-mini-2025-04-16",
    reasoning_effort="medium",
    n_runs=1,
    verbose=False,
):
    """Generate predictions from the model for evaluation"""
    if isinstance(subset, str):
        samples_path = f"data/medical_01_verifiable_problem_{subset}_{prompt_type}_prompt.jsonl"
        with open(samples_path, "r", encoding="utf-8") as f:
            test_samples = [json.loads(line) for line in f if line.strip()]
    else:
        test_samples = [subset]

    def run_inference(item):
        """Run a single inference call"""
        resp = client.responses.create(
            model=model_name,
            input=item["messages"],
            reasoning={"effort": reasoning_effort, "summary": "detailed"},
        )
        model_prediction = {'output_text': resp.output_text}
        reasoning_tokens_used = resp.usage.output_tokens_details.reasoning_tokens
        summaries = [seg.text for item in resp.output if item.type == "reasoning" for seg in item.summary]
        summaries_string = "\n".join(summaries)
        
        if verbose:
            print(f"Prompt: {item['messages'][0]['content']}")
            print(f"Model Sample: {model_prediction}\nSolution: {item['reference_answer']}\n")
        
        return {
            "model_prediction": model_prediction["output_text"],
            "input": item,
            "reasoning_tokens_used": reasoning_tokens_used,
            "reference_answer": item["reference_answer"],
            "summaries": summaries_string
        }

    # Ensure predictions directory exists
    predictions_dir = os.path.join("data", "rft", "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # Generate predictions for each run
    results_per_run = []
    for run_idx in range(n_runs):
        run_save_path = os.path.join(
            predictions_dir,
            f"{subset}_{prompt_type}_{model_name}_{reasoning_effort}_predictions_run{run_idx+1}.json"
        )
        
        # Load existing results if available
        if os.path.exists(run_save_path):
            print(f"Results for run {run_idx+1} already exist. Loading from disk.")
            with open(run_save_path, "r", encoding="utf-8") as f:
                run_results = json.load(f)
            results_per_run.append(run_results)
        else:
            # Generate new predictions
            if len(test_samples) == 1:
                run_results = [run_inference(test_samples[0])]
            else:
                run_results = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(run_inference, item) for item in test_samples]
                    for future in tqdm(futures, total=len(futures), 
                                      desc=f"Generating predictions (run {run_idx+1})"):
                        result = future.result()
                        run_results.append(result)
                
                # Save results
                with open(run_save_path, "w", encoding="utf-8") as f:
                    json.dump(run_results, f, ensure_ascii=False, indent=2)
            results_per_run.append(run_results)

    # Return flat list for single run, nested list for multiple runs
    return results_per_run[0] if n_runs == 1 else results_per_run
```

Generate predictions for both `o4-mini` and `o3` models:

```python
# Generate predictions for o4-mini
results_simple_o4mini = generate_model_predictions(
    subset="train",
    prompt_type="simple",
    model_name="o4-mini",
    reasoning_effort="medium",
    n_runs=3
)

# Generate predictions for o3
results_simple_o3 = generate_model_predictions(
    subset="train",
    prompt_type="simple",
    model_name="o3",
    reasoning_effort="medium",
    n_runs=3
)
```

## Step 4: Evaluate Predictions

Create an evaluation function to score the predictions using your grader:

```python
import functools

def evaluate_predictions_with_grader(predictions, grader_func=combined_grader):
    """Score predictions using the specified grader function"""
    results = []

    if isinstance(predictions, dict):
        predictions = [predictions]

    def run_grading(pred):
        """Grade a single prediction"""
        model_prediction = {"output_text": pred["model_prediction"]}
        item = pred["input"]
        score = grader_func(model_prediction, item)
        result = pred.copy()
        result["score"] = score
        return result

    if len(predictions) == 1:
        result = run_grading(predictions[0])
        results.append(result)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_grading, pred) for pred in predictions]
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), desc="Grading predictions"):
                results.append(future.result())

    # Calculate metrics
    total = len(results)
    correct = sum(r["score"] for r in results)
    accuracy = correct / total if total else 0.0

    metrics = {
        "total_samples": total,
        "accuracy": accuracy,
    }
    print(f"Evaluation Metrics: {metrics}")
    return metrics, results

def run_prediction_evaluation(
    model_name="o4-mini",
    reasoning_effort="medium",
    prompt_type="simple",
    subset="train",
    grader_func=combined_grader,
    num_runs=3,
):
    """Run evaluation across multiple runs and save results"""
    # Generate grader function name for file naming
    if isinstance(grader_func, functools.partial):
        name = grader_func.func.__name__
        mg = grader_func.keywords["model_grader"]
        mg_name = mg["name"]
        name = f"{name}_{mg_name}"
    else:
        name = getattr(grader_func, "__name__", 
                      getattr(grader_func, "__class__", type(grader_func)).__name__)
    grader_func_name = name.replace(" ", "_").replace(":", "_").replace("/", "_").replace(",", "_")

    for i in range(num_runs):
        # Load predictions
        preds_path = f"data/rft/predictions/{subset}_{prompt_type}_{model_name}_{reasoning_effort}_predictions_run{i+1}.json"
        with open(preds_path, "r") as f:
            preds = json.load(f)
        
        # Evaluate
        metrics, results_with_scores = evaluate_predictions_with_grader(preds, grader_func=grader_func)
        
        # Save results
        with open(f"data/rft/predictions/{subset}_{prompt_type}_{model_name}_{reasoning_effort}_{grader_func_name}_predictions_run_{i+1}_scored.json", "w") as f:
            json.dump(results_with_scores, f, indent=2)
        
        with open(f"data/rft/predictions/{subset}_{prompt_type}_{model_name}_{reasoning_effort}_{grader_func_name}_predictions_run_{i+1}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
```

## Step 5: Define Your Custom Grader

For RFT, you need a grader that provides detailed feedback. Here's an example using a model-as-a-judge approach:

```python
def model_grader(sample: dict, item: dict, model_grader_config: dict) -> float:
    """Use a model to grade the sample against the reference answer"""
    # Prepare the grading prompt
    grading_prompt = f"""
    You are evaluating a clinical model's response.
    
    Question: {item['messages'][0]['content']}
    Model's Answer: {sample['output_text']}
    Reference Answer: {item['reference_answer']}
    
    Score the model's answer on a scale from 0.0 to 1.0, where:
    - 1.0: Perfect match or semantically equivalent
    - 0.5: Partially correct or contains relevant information
    - 0.0: Completely incorrect or irrelevant
    
    Provide only the numerical score.
    """
    
    # Call the grading model
    response = client.chat.completions.create(
        model=model_grader_config["model"],
        messages=[{"role": "user", "content": grading_prompt}],
        temperature=0.0,
        max_tokens=10
    )
    
    try:
        score = float(response.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    except ValueError:
        return 0.0

# Create a partial function with your grader configuration
from functools import partial

model_grader_config = {
    "model": "gpt-4o-mini",
    "name": "gpt-4o-mini-judge"
}

custom_model_grader = partial(
    model_grader,
    model_grader_config=model_grader_config
)
```

## Step 6: Run Reinforcement Fine-Tuning

With your dataset prepared and grader defined, you can now start the RFT job:

```python
from openai import OpenAI

client = OpenAI()

# Upload your training file
with open("data/medical_01_verifiable_problem_train_simple_prompt.jsonl", "rb") as f:
    training_file = client.files.create(file=f, purpose="fine-tune")

# Upload your validation file
with open("data/medical_01_verifiable_problem_val_simple_prompt.jsonl", "rb") as f:
    validation_file = client.files.create(file=f, purpose="fine-tune")

# Create the RFT job
rft_job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file.id,
    model="o4-mini",
    hyperparameters={
        "n_epochs": 3,
        "learning_rate_multiplier": 0.1
    },
    suffix="medical-rft-demo"
)

print(f"RFT job created: {rft_job.id}")
print(f"Status: {rft_job.status}")
```

Monitor the job progress:

```python
# Check job status
job_status = client.fine_tuning.jobs.retrieve(rft_job.id)
print(f"Job status: {job_status.status}")

# List events (for debugging)
events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=rft_job.id, limit=10)
for event in events:
    print(f"{event.created_at}: {event.message}")
```

## Step 7: Evaluate the Fine-Tuned Model

Once training completes, evaluate your fine-tuned model:

```