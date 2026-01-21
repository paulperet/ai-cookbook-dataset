# Generate a Preference Dataset with distilabel

_Authored by: [David Berenstein](https://huggingface.co/davidberenstein1957) and [Sara Han Díaz](https://huggingface.co/sdiazlor)_

In this guide, you will learn how to use the `distilabel` framework to generate a synthetic preference dataset suitable for training methods like DPO, ORPO, or RLHF. The pipeline will generate multiple responses to a set of instructions, evaluate their quality, and prepare the data for further human curation using Argilla.

## Prerequisites

Before you begin, ensure you have the necessary libraries installed and your Hugging Face token ready.

### 1. Install Dependencies

Install the required packages, including `distilabel` with extras for Hugging Face Inference Endpoints and Argilla.

```bash
pip install "distilabel[argilla, hf-inference-endpoints]"
pip install "transformers~=4.0" "torch~=2.0"
```

### 2. Import Libraries and Authenticate

Import the necessary modules from `distilabel` and log in to Hugging Face using your access token.

```python
import os
from huggingface_hub import login
from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadDataFromHub,
    GroupColumns,
    FormatTextGenerationDPO,
    PreferenceToArgilla,
)
from distilabel.steps.tasks import TextGeneration, UltraFeedback

# Authenticate with Hugging Face
login(token=os.getenv("HF_TOKEN"), add_to_git_credential=True)
```

**Note:** You need a Hugging Face account and an access token (`HF_TOKEN`). Set it as an environment variable or replace `os.getenv("HF_TOKEN")` with your token string.

### 3. (Optional) Deploy Argilla

For the data curation step, you can use [Argilla](https://docs.argilla.io/latest/getting_started/quickstart/). If you haven't deployed it yet, follow the official quickstart guide. This step is optional but recommended for ensuring high-quality data.

---

## Step 1: Load the Source Dataset

We'll start by loading a small dataset of prompts from the Hugging Face Hub. We use the `LoadDataFromHub` step to fetch data.

```python
# Initialize the data loading step
load_dataset = LoadDataFromHub(
    repo_id="argilla/10Kprompts-mini",
    num_examples=1,  # Load just one example for this tutorial
    pipeline=Pipeline(name="preference-dataset-pipeline"),
)

# Load and process the data
load_dataset.load()
data_batch, _ = next(load_dataset.process())

# Inspect the loaded data
print(data_batch)
```

**Output:**
```python
[{'instruction': 'How can I create an efficient and robust workflow that utilizes advanced automation techniques to extract targeted data, including customer information, from diverse PDF documents and effortlessly integrate it into a designated Google Sheet? Furthermore, I am interested in establishing a comprehensive and seamless system that promptly activates an SMS notification on my mobile device whenever a new PDF document is uploaded to the Google Sheet, ensuring real-time updates and enhanced accessibility.',
  'topic': 'Software Development'}]
```

This step loads the `instruction` and `topic` columns from the dataset. For a full pipeline, you would increase `num_examples`.

---

## Step 2: Generate Responses with Multiple LLMs

Next, we will generate responses for each instruction using two different language models via the Hugging Face Serverless Inference API. This creates the candidate responses we will later evaluate.

We'll use:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

Each model is wrapped in a `TextGeneration` task.

```python
# Define the response generation tasks
generate_responses = [
    TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3-8B-Instruct",
            generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
        ),
        pipeline=Pipeline(name="preference-dataset-pipeline"),
    ),
    TextGeneration(
        llm=InferenceEndpointsLLM(
            model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            tokenizer_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
        ),
        pipeline=Pipeline(name="preference-dataset-pipeline"),
    ),
]

# Process a sample instruction through both models
sample_instruction = [{"instruction": "Which are the top cities in Spain?"}]

for task in generate_responses:
    task.load()
    result = next(task.process(sample_instruction))
    print(f"Model: {result[0]['model_name']}")
    print(f"Generation: {result[0]['generation'][:200]}...\n")
```

**Output (truncated):**
```
Model: meta-llama/Meta-Llama-3-8B-Instruct
Generation: Spain is a country with a rich culture, history, and architecture, and it has many great cities to visit. Here are some of the top cities in Spain:

1. **Madrid**: The capital city of Spain, known for its vibrant nightlife, museums, and historic landmarks like the Royal Palace and Prado Museum.
2. **Barcelona**: The second-largest city in Spain, famous for its modernist architecture, beaches, and iconic landmarks like La Sagrada Família and Park Güell, designed by Antoni Gaudí.
3. **Valencia**: Located on the Mediterranean coast, Valencia is known for its beautiful beaches, City of Arts and Sciences, and delicious local cuisine, such as paella...

Model: mistralai/Mixtral-8x7B-Instruct-v0.1
Generation: Here are some of the top cities in Spain based on various factors such as tourism, culture, history, and quality of life:

1. Madrid: The capital and largest city in Spain, Madrid is known for its vibrant nightlife, world-class museums (such as the Prado Museum and Reina Sofia Museum), stunning parks (such as the Retiro Park), and delicious food.

2. Barcelona: Famous for its unique architecture, Barcelona is home to several UNESCO World Heritage sites designed by Antoni Gaudí, including the Sagrada Familia and Park Güell. The city also boasts beautiful beaches, a lively arts scene, and delicious Catalan cuisine...
```

Each task outputs a `generation`, along with metadata and the `model_name`.

---

## Step 3: Group Responses for Evaluation

The evaluation step requires all model responses to be in a single list. Currently, each model's output is in a separate subset (e.g., `text_generation_0`, `text_generation_1`). We use `GroupColumns` to merge them.

```python
group_responses = GroupColumns(
    columns=["generation", "model_name"],
    output_columns=["generations", "model_names"],
    pipeline=Pipeline(name="preference-dataset-pipeline"),
)

# Example: Simulate grouped output from two model responses
grouped = next(
    group_responses.process(
        [  # Output from first model (text_generation_0 subset)
            {
                "generation": "Madrid",
                "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            },
        ],
        [  # Output from second model (text_generation_1 subset)
            {
                "generation": "Barcelona",
                "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            }
        ],
    )
)

print(grouped)
```

**Output:**
```python
[{'generations': ['Madrid', 'Barcelona'],
  'model_names': ['meta-llama/Meta-Llama-3-8B-Instruct',
   'mistralai/Mixtral-8x7B-Instruct-v0.1']}]
```

The `generations` and `model_names` columns now contain lists, ready for the evaluation step.

---

## Step 4: Evaluate Responses with UltraFeedback

To build a preference dataset, we need to judge which response is better. We'll use the `UltraFeedback` task, which employs a powerful LLM (like `meta-llama/Meta-Llama-3-70B-Instruct`) to score responses across multiple dimensions: helpfulness, honesty, instruction-following, and truthfulness.

```python
# Define the evaluation task
evaluate_responses = UltraFeedback(
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3-70B-Instruct",
        generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
    ),
    pipeline=Pipeline(name="preference-dataset-pipeline"),
)

# Load the task
evaluate_responses.load()

# Process the grouped data (in a real pipeline, this connects automatically)
# Example input structure the step expects:
sample_for_evaluation = [
    {
        "instruction": "Which are the top cities in Spain?",
        "generations": ["Madrid", "Barcelona"],
        "model_names": ["meta-llama/Meta-Llama-3-8B-Instruct",
                       "mistralai/Mixtral-8x7B-Instruct-v0.1"]
    }
]

# The evaluation step would be called here in a full pipeline
# evaluation_results = next(evaluate_responses.process(sample_for_evaluation))
```

The `UltraFeedback` task will output ratings and, crucially, a preference ranking indicating which response is chosen as the "chosen" (preferred) and which is "rejected."

---

## Next Steps: Formatting and Curation

After evaluation, the remaining steps in a complete pipeline would typically be:

1.  **Format for DPO:** Use the `FormatTextGenerationDPO` step to structure the data into the standard DPO format: `(prompt, chosen_response, rejected_response)`.
2.  **Send to Argilla:** Use the `PreferenceToArgilla` step to push the preference pairs to an Argilla workspace for human review and curation, ensuring the highest quality before final dataset export.

## Summary

You have now built the core components of a `distilabel` pipeline to generate a synthetic preference dataset:
1.  **Loaded** a source dataset of instructions.
2.  **Generated** diverse responses using multiple LLMs.
3.  **Grouped** the responses for evaluation.
4.  **Prepared** to evaluate them using the `UltraFeedback` task.

To run this as an end-to-end pipeline, you would connect these steps using the `Pipeline` class and execute it. The final output is a dataset of instruction-preference pairs, ready for training alignment algorithms or for further human refinement in Argilla.