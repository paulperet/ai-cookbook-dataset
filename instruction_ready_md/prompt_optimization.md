# Automated Prompt Optimization with Mistral Models

## Introduction

Prompt engineering is often a non-standard, trial-and-error process that's difficult to scale. Fortunately, we can automate this using **Prompt Optimization (PO)**, a technique that iteratively refines prompts to improve their performance for specific tasks.

This guide shows you how to implement PO using Mistral models to process unstructured survey responses, helping you efficiently filter through large volumes of applicant data.

## Prerequisites

First, let's set up our environment and define the prompts we want to optimize.

### 1. Define Task Prompts

We'll create two prompts for processing applicant data: one for classifying job titles and another for standardizing locations.

```python
# Overarching context for the task
context = (
    "I am working on recruiting people to advocate about the products of an AI company. "
    "The position is in close contact with the DevRel team, and we are looking for people "
    "to share on their personal social media about the company and its products. "
    "The company produces Large Language Models and is very followed, "
    "so I received a large number of applications that I need to process quickly. "
    "I can't process them manually, and there's little structure in the application form. "
    "Please help me extract structured information from what applicants declared."
)

# Prompt for classifying job titles
job_prompt = lambda job_title: (
    "Your task is to provide me with a direct classification of the person's job title into one of 4 categories. "
    "The categories you can decide are always: 'RESEARCH', 'ENGINEERING', 'BUSINESS', 'FOUNDER'. "
    "There is no possibility for mixed assignments. You always assign one and one only category to each subject. "
    "When in doubt, assign to 'OTHER'. You must strictly adhere to the categories I have mentioned, and nothing more. "
    "This means that you cannot use any other output apart from 'RESEARCH', 'ENGINEERING', 'BUSINESS', 'FOUNDER', 'OTHER'. "
    "Keep your answer very, very concise. Don't give context on your answer. As a matter of fact, only answer with one word "
    "based on the category you deem the most appropriate. Absolutely don't change this. You will be penalized if "
    "(1) you use a category outside of the ones I have mentioned and (2) you use more than 1 word in your output. "
    f"# INPUT declared title: the person job title is {job_title}"
)

# Prompt for standardizing locations
location_prompt = lambda location: (
    "Your task is basic. Your task is to disambiguate the respondent's answer in terms of the location used. "
    "Your output is always CITY, COUNTRY. Use always the English name of a city. Also, always use the international "
    "country code. Nothing else. For instance, if a user answered with 'Rome', you would output 'Rome, IT'. "
    "In the rare case when someone puts down multiple locations, make sure you always select the first one. Nothing more"
    f" #INPUT declared location: the respondent declared being located in {location}"
)
```

### 2. Install Dependencies

We'll use MetaGPT for prompt optimization. Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/geekan/MetaGPT

# Install dependencies
pip install -qUr MetaGPT/requirements.txt

# Navigate to the directory
cd MetaGPT
```

## Setting Up Prompt Optimization

### 3. Create Instruction Files

MetaGPT requires YAML template files for each prompt you want to optimize. Let's create a helper function to generate these files:

```python
from typing import Optional
import yaml

def prompt_to_dict(
    prompt: str,
    requirements: Optional[str],
    questions: list[str],
    answers: list[str],
    count: Optional[int] = None,
) -> dict:
    """Convert prompt information to dictionary format for YAML."""
    return {
        "prompt": prompt if isinstance(prompt, str) else prompt(""),
        "requirements": requirements,
        "count": count,
        "qa": [
            {"question": question, "answer": answer}
            for question, answer in zip(questions, answers)
        ]
    }

# Define prompts and their requirements
prompts = {
    "job": job_prompt,
    "location": location_prompt
}

requirements = [
    "The job title, categorized",
    "The location, disambiguated"
]

# Save template files for each prompt
path = "metagpt/ext/spo/settings"

for (name, prompt), requirement in zip(prompts.items(), requirements):
    with open(f"{path}/{name}.yaml", "w") as f:
        yaml.dump(
            prompt_to_dict(prompt, requirement, [""], [""]),
            f,
        )
```

### 4. Configure Model Settings

Create a configuration file specifying which Mistral models to use for execution, evaluation, and optimization:

```python
def models_dict(mistral_api_key: str) -> dict:
    """Create model configuration dictionary."""
    return {
        "llm": {
            "api_type": "openai",
            "model": "mistral-small-latest",
            "base_url": "https://api.mistral.ai/v1/",
            "api_key": mistral_api_key,
            "temperature": 0
        },
        "models": {
            "mistral-small-latest": {
                "api_type": "openai",
                "base_url": "https://api.mistral.ai/v1/",
                "api_key": mistral_api_key,
                "temperature": 0
            },
            "mistral-large-latest": {
                "api_type": "openai",
                "base_url": "https://api.mistral.ai/v1/",
                "api_key": mistral_api_key,
                "temperature": 0
            }
        }
    }

# Save the configuration
path = "config/config2.yaml"
MISTRAL_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key

with open(path, "w") as f:
    yaml.dump(models_dict(MISTRAL_API_KEY), f)
```

## Running Prompt Optimization

### 5. Create Optimization Script

Due to Jupyter notebook limitations with `asyncio`, we'll create a separate Python script:

```python
# spo.py
from metagpt.ext.spo.components.optimizer import PromptOptimizer
from metagpt.ext.spo.utils.llm_client import SPO_LLM

# Initialize LLM settings
SPO_LLM.initialize(
    optimize_kwargs={
        "model": "mistral-large-latest", 
        "temperature": 0.6
    },
    evaluate_kwargs={
        "model": "mistral-small-latest", 
        "temperature": 0.3
    },
    execute_kwargs={
        "model": "mistral-small-latest", 
        "temperature": 0
    }
)

template_name = "job.yaml"  # Change this for each prompt!

# Create and run optimizer
optimizer = PromptOptimizer(
    optimized_path="workspace",  # Output directory
    initial_round=1,             # Starting round
    max_rounds=5,               # Maximum optimization rounds
    template=template_name,      # Template file
    name="Mistral-Prompt-Opt",   # Project name
)

optimizer.optimize()
```

### 6. Execute Optimization

Run the optimization script:

```bash
python spo.py
```

The optimization process will run for 5 rounds, refining your prompt based on automated evaluation.

## Results Analysis

After optimization, compare the original and optimized prompts:

| Original Prompt | Optimized Prompt |
|-----------------|------------------|
| Your task is to provide me with a direct classification of the person's job title into one of 4 categories. The categories you can decide are always: 'RESEARCH', 'ENGINEERING', 'BUSINESS', 'FOUNDER'. There is no possibility for mixed assignments. You always assign one and one only category to each subject. When in doubt, assign to 'OTHER'. You must strictly adhere to the categories I have mentioned, and nothing more. This means that you cannot use any other output apart from 'RESEARCH', 'ENGINEERING', 'BUSINESS', 'FOUNDER', 'OTHER'. Keep your answer very, very concise. Don't give context on your answer. As a matter of fact, only answer with one word based on the category you deem the most appropriate. Absolutely don't change this. You will be penalized if (1) you use a category outside of the ones I have mentioned and (2) you use more than 1 word in your output. # INPUT declared title: the person job title is {job_title} | Your task is to classify the given job title into one of the following categories: 'RESEARCH', 'ENGINEERING', 'BUSINESS', 'FOUNDER'. If the job title does not fit any of these categories, classify it as 'OTHER'. You must strictly adhere to these categories. If a job title is ambiguous or could fit into multiple categories, choose the most relevant category based on common industry standards. For example, 'Data Scientist' could fit into both 'RESEARCH' and 'ENGINEERING', but is typically classified as 'RESEARCH'. Similarly, 'Data Analyst' is typically classified as 'BUSINESS'. Provide your answer using one word only, in all uppercase letters without any additional context or explanations.<br><br># INPUT: The person's job title is: {job_title}<br><br># Example:<br># INPUT: The person's job title is: Software Developer<br># OUTPUT: ENGINEERING |

## Key Improvements

The optimized prompt includes several enhancements:

1. **Few-shot examples**: Added concrete examples to guide the model
2. **Clearer formatting**: Improved structure with clear input/output examples
3. **Industry context**: Added guidance for ambiguous cases based on industry standards
4. **Simplified instructions**: Removed redundant warnings and penalties

## Next Steps

You can apply the same process to optimize other prompts:

1. Change `template_name = "job.yaml"` to `"location.yaml"` in the optimization script
2. Run the optimization for each prompt you need to refine
3. Experiment with different numbers of optimization rounds
4. Test the optimized prompts on real data to validate improvements

This automated approach to prompt optimization helps you create more effective prompts with less manual effort, making your AI workflows more reliable and scalable.