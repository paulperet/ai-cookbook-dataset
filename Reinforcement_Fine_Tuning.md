# **Exploring Model Graders for Reinforcement Fine-Tuning**

*This guide is for developers and ML practitioners who already know their way around OpenAIʼs APIs, have a basic understanding of reinforcement fine-tuning (RFT), and wish to use their fine-tuned models for research or other appropriate uses. OpenAI’s services are not intended for the personalized treatment or diagnosis of any medical condition and are subject to our [applicable terms](https://openai.com/policies/).*

[Reinforcement fine-tuning (RFT)](https://platform.openai.com/docs/guides/reinforcement-fine-tuning) of reasoning models consists in running reinforcement learning on of top the models to improve their reasoning performance by exploring the solution space and reinforcing strategies that result in a higher reward. RFT helps the model make sharper decisions and interpret context more effectively. 

In this guide, weʼll walk through how to apply RFT to the OpenAI `o4-mini` reasoning model, using a task from the life sciences research domain: predicting outcomes from doctor-patient transcripts and descriptions, which is a necessary assessment in many health research studies. We'll use a subset of the medical-o1-verifiable-problem [dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-verifiable-problem/viewer/default/train?row=0). You will learn key steps to take in order to succesfully run RFT jobs for your use-cases.

Here’s what we’ll cover:

- **[1. Setup](#1-setup)**
- **[2. Gathering the dataset](#2-gathering-the-dataset)**
- **[3. Benchmarking the base model](#3-benchmarking-the-base-model)**
- **[4. Defining your grader](#4-defining-your-grader)**
- **[5. Training](#5-training)**
- **[6. Using your fine-tuned model](#6-using-your-fine-tuned-model)**

---

## **1. Setup**

Even strong reasoning models can miss the mark when it comes to expert-level behavior-especially in domains like medicine, where nuance and exactness matter. Imagine a model trying to extract [ICD-10](https://www.cms.gov/medicare/coding-billing/icd-10-codes) codes from a transcript: even if it understands the gist, it may not use the precise terminology expected by medical professionals. 

Other great candidates for RFT include topics like ledger normalization or tiering fraud risk- settings in which you want precise, reliable, and repeatable reasoning. Checkout our [RFT use-cases guide](https://platform.openai.com/docs/guides/rft-use-cases) for great examples. 

In our case, weʼll focus on teaching `o4-mini` to become better at predicting the outcomes of clinical conversations and descriptions. Specifically, we want to see if RFT can boost the accuracy of the prediction. 

Along the way, weʼll talk about how to write effective graders, how they guide the modelʼs learning, and how to watch out for classic reward-hacking pitfalls. 

---

## **2. Gathering the Dataset**

Letʼs start off by loading the dataset from Hugging Face. Weʼre interested in samples framed as a description of a patient case with an associated question, followed by the correct answer. These represent real world transcripts where a physician is summarizing a case and assigning an outcome. For any use-case, verifying the accuracy of the gold level answers is critical and requires careful consideration. Here, we will trust the dataset quality.

```python
import re
from datasets import load_dataset
ds = load_dataset("FreedomIntelligence/medical-o1-verifiable-problem")

def is_age_question(sample):
    question = sample.get('Open-ended Verifiable Question', '')
    # Match "A 88-year-old", "An 8-year-old", "A 23-year-old", etc. at the start
    return re.match(r"^(A|An) \d{1,2}-year-old", question) is not None

filtered_samples = [s for s in ds["train"] if is_age_question(s)]
print(f"Filtered samples: {len(filtered_samples)}")
```

One of the advantages of RFT is that it doesnʼt need thousands of samples to start making a difference. Thanks to trajectory sampling and the feedback loop during training, the model learns not just correct behaviors, but also patterns to avoid. This means we can see solid gains even with small datasets.

For this run, weʼll randomly sample 100 training and 100 test examples and slightly normalize them.

```python
import random

# Set a random seed for reproducibility
random.seed(42)

# Randomly select 100 training samples from filtered_samples
train_samples = random.sample(filtered_samples, min(100, len(filtered_samples)))

# Remove training samples from filtered_samples to avoid overlap
remaining_samples = [s for s in filtered_samples if s not in train_samples]

# Randomly select 100 test samples from the remaining samples (no overlap)
test_samples = random.sample(remaining_samples, min(100, len(remaining_samples)))

print(f"Number of training samples: {len(train_samples)}")
print(f"Number of test samples: {len(test_samples)}")
```

```python
# Standardize the 'Ground-True Answer' fields to all lowercase in train and test samples
for sample in train_samples:
    if 'Ground-True Answer' in sample and isinstance(sample['Ground-True Answer'], str):
        sample['Ground-True Answer'] = sample['Ground-True Answer'].lower()

for sample in test_samples:
    if 'Ground-True Answer' in sample and isinstance(sample['Ground-True Answer'], str):
        sample['Ground-True Answer'] = sample['Ground-True Answer'].lower()
```

We'll convert these samples to `jsonl` format, as expected by the [reinforcement finetuning API](https://platform.openai.com/docs/api-reference/fine-tuning/reinforcement-input).

```python
import json

def convert_to_jsonl_format(samples, filename):
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
    samples = []
    with open(filename, "r") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

# Save the datasets to jsonl files
convert_to_jsonl_format(train_samples, "data/medical_01_verifiable_problem_train.jsonl")
convert_to_jsonl_format(test_samples, "data/medical_01_verifiable_problem_val.jsonl")

# Load the datasets back from jsonl files
train_samples_loaded = load_jsonl("data/medical_01_verifiable_problem_train.jsonl")
test_samples_loaded = load_jsonl("data/medical_01_verifiable_problem_val.jsonl")
```

Next up: we’ll see how the base model performs out of the box-and where there’s room to grow.

---

## **3. Benchmarking the Base Model**

Before we fine-tune anything, we need to know where we’re starting from. Benchmarking gives us a clear picture of the model’s initial strengths and weaknesses-so we can later measure how far it’s come.

We’ll first lean on two simple yet powerful evaluators:

1. `clinical_phrase_binary_grader` - an exact-match checker.
2. `clinical_phrase_grader` - a softer, token-based similarity grader.

```python
from rapidfuzz import fuzz, utils

def clinical_phrase_grader(sample: dict, item: dict) -> float:
    from rapidfuzz import fuzz, utils
    score = fuzz.token_set_ratio(sample["output_text"], item["reference_answer"], processor=utils.default_process)
    return score / 100.0

def clinical_phrase_binary_grader(sample: dict, item: dict) -> float:
    return 1.0 if sample["output_text"] == item["reference_answer"] else 0.0

def combined_grader(sample: dict, item: dict, weights: list[float] = [0.85, 0.15]) -> float:
    clinical_phrase_score = clinical_phrase_grader(sample, item)
    binary_score = clinical_phrase_binary_grader(sample, item)
    return weights[0] * clinical_phrase_score + weights[1] * binary_score
```

This combination lets us track both strict correctness and partial lexical overlap. The binary grader gives a crisp 0 or 1: did the model produce an exact match? The softer one gives more nuance-how close did the output come to the gold answer? We use both because outcomes are often phrased in multiple valid ways. For instance, a model might respond with “gouty arthritis” instead of “gout.” While a human evaluator could consider this partially acceptable, a strict string match would not. Combining exact and fuzzy scoring ensures a more accurate and fair assessment of model outputs. 

We build a helper function to preprend the examples with a system prompt.

```python
def prepend_system_prompt_to_first_user_message(samples, system_prompt, path=None):
    new_samples = []
    for sample in samples:
        # Deep copy to avoid mutating the original
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
```

```python
simple_prompt = """You are an expert clinician. For each clinical vignette, respond with exactly one phrase: the single most likely outcome or phenomenon, all in lowercase. 
- Do not add punctuation, articles, explanations, or commentary - output only the term itself.
- Sometimes, the expected answer can be a synonym of what you think.
- Use the standard clinical name (e.g. “thought withdrawal”, “Toxoplasma encephalitis”)."""
train_samples_loaded_simple_sys_prompt = prepend_system_prompt_to_first_user_message(
    train_samples_loaded, simple_prompt, path="data/medical_01_verifiable_problem_train_simple_prompt.jsonl"
)
test_samples_loaded_simple_sys_prompt = prepend_system_prompt_to_first_user_message(
    test_samples_loaded, simple_prompt, path="data/medical_01_verifiable_problem_val_simple_prompt.jsonl"
)
```

Then build a helper function to generate and store the model's predictions.

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
    if isinstance(subset, str):
        samples_path = f"data/medical_01_verifiable_problem_{subset}_{prompt_type}_prompt.jsonl"
        with open(samples_path, "r", encoding="utf-8") as f:
            test_samples = [json.loads(line) for line in f if line.strip()]
    else:
        test_samples = [subset]

    def run_inference(item):
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
            print("Prompt: {}".format(item["messages"][0]["content"]))
            print(f"Model Sample: {model_prediction}\nSolution: {item['reference_answer']}\n")
        return {
            "model_prediction": model_prediction["output_text"],
            "input": item,
            "reasoning_tokens_used": reasoning_tokens_used,
            "reference_answer": item["reference_answer"],
            "summaries": summaries_string
        }

    # Ensure the predictions directory exists before any file operations
    predictions_dir = os.path.join("data", "rft", "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # Check if results already exist for all runs
    results_per_run = []
    for run_idx in range(n_runs):
        run_save_path = os.path.join(
            predictions_dir,
            f"{subset}_{prompt_type}_{model_name}_{reasoning_effort}_predictions_run{run_idx+1}.json"
        )
        if os.path.exists(run_save_path):
            print(f"Results for run {run_idx+1} already exist at {run_save_path}. Loading results.")
            with open(run_save_path, "r", encoding="utf-8") as f:
                run_results = json.load(f)
            results_per_run.append(run_results)
        else:
            if len(test_samples) == 1:
                run_results = [run_inference(test_samples[0])]
            else:
                run_results = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(run_inference, item) for item in test_samples]
                    for future in tqdm(futures, total=len(futures), desc=f"Generating predictions (run {run_idx+1})"):
                        result = future.result()
                        run_results.append(result)
                with open(run_save_path, "w", encoding="utf-8") as f:
                    json.dump(run_results, f, ensure_ascii=False, indent=2)
            results_per_run.append(run_results)

    # Return a flat list for backward compatibility
    if n_runs == 1:
        return results_per_run[0]
    else:
        return results_per_run
```

To generate the predictions, first make sure your API key is set:

```bash
export OPENAI_API_KEY=...
```

```python
# OpenAI o4-mini model
results_simple_o4mini = generate_model_predictions(
    subset="train",
    prompt_type="simple",
    model_name="o4-mini",
    reasoning_effort="medium",
    n_runs=3
)
```

```python
# OpenAI o3 model
results_simple_o3 = generate_model_predictions(
    subset="train",
    prompt_type="simple",
    model_name="o3",
    reasoning_effort="medium",
    n_runs=3
)
```

We now have predictions that are ready to be evaluated.
We'll build a helper function that allows us to easily swap in different scoring methods,

```python
import functools

def evaluate_predictions_with_grader(
    predictions,
    grader_func=combined_grader,
):
    results = []

    if isinstance(predictions, dict):
        predictions = [predictions]

    def run_grading(pred):
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
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Grading predictions"):
                results.append(future.result())

    total = len(results)
    correct = sum(r["score"] for r in results)
    accuracy = correct / total if total else 0.0

    metrics = {
        "total_samples": total,
        "accuracy": accuracy,
    }
    print(metrics)
    return metrics, results

def run_prediction_evaluation(
    model_name="o4-mini",
    reasoning_effort="medium",
    prompt_type="simple",
    subset="train",
    grader_func=combined_grader,
    num_runs=3,
):
    if isinstance(grader_func, functools.partial):
        name = grader_func.func.__name__
        mg = grader_func.keywords["model_grader"]
        mg_name = mg["name"]
        name = f"{name}_{mg_name}"
    else:
        name = getattr(grader_func, "__name__", getattr(grader_func, "__class__", type(grader_func)).__name__)
    grader_func_name = name.replace(" ", "_").replace(":", "_").replace("/", "_").replace(",", "_")

    for i in range(num_runs):
        preds_path = f"data/rft/predictions/{subset}_{prompt_type}_{model_name}_{reasoning_effort}_predictions_run{i+1}.json"
        with open(preds_path, "r") as f:
            preds = json.load(f)
        metrics, results_with_scores = evaluate_predictions_with_grader(preds, grader_func=grader_func)
        # Save the scored results
        with open(f"data/rft/predictions/{subset}_{prompt_type}_{model_name}_{reasoning_effort}_{grader_func_name}_predictions_run_{i+1}_scored.json", "w") as f:
            json.dump(results_with_scores, f, indent=2)
        # Save the metrics
        with open(f"data/rft/predictions/{subset}_{prompt_type}_{model_name}_{reasoning_effort}_{grader_func_name}_predictions_run_{i+1}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        # Save the scores (if present in results_with_scores)
        scores = [item.get("score") for item in results_with_scores if "score" in item]
        with open(f"data/r