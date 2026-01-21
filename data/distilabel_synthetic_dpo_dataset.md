# Create a Mathematical Preference Dataset with `distilabel`

_Authored by: David Berenstein and Sara Han Díaz from [Argilla](https://github.com/argilla-io/argilla)_

In this tutorial, you will learn how to use different Mistral models to create a synthetic mathematical preference dataset using the `distilabel` framework. You will generate instructions, produce multiple answers, judge their quality, and optionally analyze the results in Argilla.

> [distilabel](https://github.com/argilla-io/distilabel/tree/main) is an AI Feedback (AIF) framework for generating and labeling datasets with LLMs. It is designed for robustness, efficiency, and scalability, enabling you to build synthetic datasets for various applications.

## Prerequisites

Before you begin, ensure you have the following:

1.  A **MistralAI API key**. You can obtain one by signing up at [MistralAI](https://mistral.ai/) and creating a key in the **API Keys** section.
2.  (Optional) An **Argilla Space** on Hugging Face for data visualization and annotation. Create one using the [Argilla template](https://huggingface.co/new-space?template=argilla/argilla-template-space). You will need the `ARGILLA_API_URL` (e.g., `https://[your-owner-name]-[your_space_name].hf.space`) and `ARGILLA_API_KEY` (found in your Space's **Settings**).

## Step 1: Setup and Installation

Install the required packages, including the `mistralai` and `argilla` extras for `distilabel`.

```bash
pip install distilabel[mistralai,argilla]==1.1.1
```

## Step 2: Import Libraries and Set Environment Variables

Import the necessary modules from `distilabel` and set your API keys as environment variables.

```python
import os
from distilabel.llms import MistralLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import ExpandColumns, LoadDataFromDicts, CombineColumns, PreferenceToArgilla
from distilabel.steps.tasks import TextGeneration, UltraFeedback, SelfInstruct
from distilabel.steps.formatting.dpo import FormatTextGenerationDPO

# Set your API keys (replace with your actual keys or use a secure method)
os.environ['MISTRAL_API_KEY'] = '<YOUR_MISTRAL_API_KEY>'
# Optional: Only set these if using Argilla
os.environ['ARGILLA_API_URL'] = '<YOUR_ARGILLA_API_URL>'
os.environ['ARGILLA_API_KEY'] = '<YOUR_ARGILLA_API_KEY>'
```

## Step 3: Define the Seed Data

You will start with a list of mathematical topics. For this tutorial, you'll use the first 10 topics to generate instructions. You can adjust the number to create more or less data.

```python
math_topics = [
    "Algebraic Expressions",
    "Linear Equations",
    "Quadratic Equations",
    "Polynomial Functions",
    "Rational Expressions",
    "Exponential Functions",
    "Logarithmic Functions",
    "Sequences and Series",
    "Matrices",
    "Determinants",
    # ... (full list truncated for brevity)
]

# Prepare the seed data as a list of dictionaries
data = [{"input": topic} for topic in math_topics[:10]]
```

## Step 4: Build the Pipeline

You will construct a `Pipeline` that defines the data generation workflow as a series of connected steps (a Directed Acyclic Graph). The pipeline will:

1.  Load the seed data.
2.  Generate diverse instructions from each topic using a `SelfInstruct` task.
3.  Expand the list of instructions into individual rows.
4.  Generate two different answers for each instruction using two Mistral models.
5.  Combine the generations into a single column.
6.  Judge the quality of each answer pair using the `UltraFeedback` task with a more capable model.
7.  (Optional) Send the data to Argilla for annotation.
8.  Format the final dataset for Direct Preference Optimization (DPO).

Here is the complete pipeline definition:

```python
with Pipeline(name="mistral-pipe", description="A pipeline to generate and score a dataset") as pipeline:

    # Step 1: Load the initial seed data
    load_dataset = LoadDataFromDicts(
        name="load_dataset",
        data=data,
    )

    # Step 2: Generate multiple instructions per input topic
    self_instruct_open_mistral = SelfInstruct(
        name="self_instruct_open_mistral",
        llm=MistralLLM(model="open-mistral-7b"),
    )

    # Step 3: Expand the list of instructions into separate rows
    expand_columns = ExpandColumns(
        name="expand_columns",
        columns={"instructions": "instruction"}
    )

    # Step 4: Generate answers using two different models
    tasks = []
    for llm in (MistralLLM(model="open-mistral-7b"),
                MistralLLM(model="open-mixtral-8x7b")):
        tasks.append(
            TextGeneration(name=f"generate_{llm.model_name}", llm=llm)
        )

    # Step 5: Combine outputs from both models
    combine_generations = CombineColumns(
        name="combine_generations",
        columns=["generation", "model_name"],
        output_columns=["generations", "model_names"],
    )

    # Step 6: Judge the generated answers
    ultrafeedback_mistral_large = UltraFeedback(
        name="ultrafeedback_mistral_large",
        llm=MistralLLM(model="mistral-large-latest"),
        aspect="overall-rating"
    )

    # Step 7 (Optional): Send data to Argilla for annotation
    to_argilla = PreferenceToArgilla(
        dataset_name="mathematical-dataset",
        dataset_workspace="admin",  # Default workspace
        num_generations=2
    )

    # Step 8: Format the dataset for DPO
    format_dpo = FormatTextGenerationDPO(name="format_dpo")

    # Connect all steps in the pipeline
    (load_dataset
     >> self_instruct_open_mistral
     >> expand_columns
     >> tasks
     >> combine_generations
     >> ultrafeedback_mistral_large
     >> [to_argilla, format_dpo])
```

> **Note:** If you do not wish to use Argilla, you can comment out the `to_argilla` step and remove it from the connection list at the end of the pipeline.

## Step 5: Execute the Pipeline

Run the pipeline with the `run` method. You can pass runtime parameters to customize generation, such as token limits and temperature. The complete run will take approximately 10 minutes.

```python
distiset = pipeline.run(
    parameters={
        "generate_open-mistral-7b": {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                }
            }
        },
        "generate_open-mixtral-8x7b": {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                }
            }
        }
    }
)
```

> **Tip:** For a quick test, you can use `pipeline.dry_run()` instead of `run`. This will process only a single batch.

## Step 6: Inspect and Share the Dataset

After the pipeline completes, you can inspect the resulting dataset.

```python
distiset
```

Finally, push your synthetic dataset to the Hugging Face Hub to share it with the community.

```python
distiset.push_to_hub(repo_id="<your-username>/math-preference-dataset")
```

If you used the Argilla step, you can now navigate to your Space to visualize, annotate, and improve the dataset quality.

## Conclusion

You have successfully created a synthetic mathematical preference dataset using Mistral models and `distilabel`. This tutorial walked you through generating instructions, producing multiple candidate answers, judging their quality, and formatting the data for downstream tasks like DPO.

🚀 Experiment with the pipeline by changing the seed topics, models, or generation parameters to create your own custom datasets.

### Further Resources
-   [distilabel Documentation](https://distilabel.argilla.io/latest/)
-   [Paper Implementations (e.g., DEITA)](https://distilabel.argilla.io/latest/sections/pipeline_samples/papers/deita/)
-   [Task Implementations](https://distilabel.argilla.io/latest/components-gallery/tasks/)