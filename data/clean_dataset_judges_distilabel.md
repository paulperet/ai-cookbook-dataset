# Clean an Existing Preference Dataset with LLMs as Judges

**Authors:** [David Berenstein](https://huggingface.co/davidberenstein1957), [Sara Han Díaz](https://huggingface.co/sdiazlor)

**Libraries:** [distilabel](https://github.com/argilla-io/distilabel), [Argilla](https://github.com/argilla-io/argilla), [Hugging Face Inference Endpoints](https://github.com/huggingface/huggingface_hub)

In this guide, you will learn how to clean a preference dataset using AI feedback. We'll use **distilabel** to build a pipeline that evaluates the quality of text responses using a large language model (LLM) as a judge, and then optionally curate the results with **Argilla** for human review.

## Prerequisites

Before starting, ensure you have the required libraries installed. Run the following commands in your environment:

```bash
pip install "distilabel[hf-inference-endpoints]"
pip install "transformers~=4.0" "torch~=2.0"
```

If you plan to use Argilla for data curation, install the extra dependency:

```bash
pip install "distilabel[argilla, hf-inference-endpoints]"
```

Now, import the necessary modules:

```python
import os
import random
from datasets import load_dataset
from huggingface_hub import login

from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    KeepColumns,
    LoadDataFromDicts,
    PreferenceToArgilla,
)
from distilabel.steps.tasks import UltraFeedback
```

You will need a Hugging Face token to access the Inference API. Log in using:

```python
login(token=os.getenv("HF_TOKEN"), add_to_git_credential=True)
```

## Step 1: Load and Prepare the Dataset

We'll use the [`Intel/orca_dpo_pairs`](https://huggingface.co/datasets/Intel/orca_dpo_pairs) dataset from the Hugging Face Hub. This dataset contains pairs of chosen and rejected responses for Direct Preference Optimization (DPO).

First, load a small sample of the dataset:

```python
dataset = load_dataset("Intel/orca_dpo_pairs", split="train[:20]")
```

To avoid any positional bias, we'll shuffle the `chosen` and `rejected` columns randomly and track their original order:

```python
def shuffle_and_track(chosen, rejected):
    pair = [chosen, rejected]
    random.shuffle(pair)
    order = ["chosen" if x == chosen else "rejected" for x in pair]
    return {"generations": pair, "order": order}

dataset = dataset.map(lambda x: shuffle_and_track(x["chosen"], x["rejected"]))
```

Convert the dataset to a list of dictionaries for processing:

```python
dataset = dataset.to_list()
```

**Optional – Create a Custom Step:** Instead of preprocessing the data manually, you can encapsulate the shuffling logic in a custom `GlobalStep`. Save the following code in a file named `shuffle_step.py`:

```python
# shuffle_step.py
from typing import TYPE_CHECKING, List
from distilabel.steps import GlobalStep, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput
    
import random

class ShuffleStep(GlobalStep):
    @property
    def inputs(self) -> List[str]:
        return ["instruction", "chosen", "rejected"]

    @property
    def outputs(self) -> List[str]:
        return ["instruction", "generations", "order"]

    def process(self, inputs: StepInput) -> "StepOutput":
        outputs = []
        for input in inputs:
            chosen = input["chosen"]
            rejected = input["rejected"]
            pair = [chosen, rejected]
            random.shuffle(pair)
            order = ["chosen" if x == chosen else "rejected" for x in pair]
            outputs.append({
                "instruction": input["instruction"],
                "generations": pair,
                "order": order
            })
        yield outputs
```

Then import it:

```python
from shuffle_step import ShuffleStep
```

## Step 2: Define the Cleaning Pipeline

We'll construct a distilabel pipeline with four main steps: loading the data, evaluating responses with an LLM, filtering columns, and optionally sending results to Argilla.

### 2.1 Load the Dataset

Use `LoadDataFromDicts` to load the shuffled data and rename the `question` column to `instruction` for consistency.

```python
load_dataset = LoadDataFromDicts(
    data=dataset,
    output_mappings={"question": "instruction"},
    pipeline=Pipeline(name="clean-dataset-pipeline"),
)
```

### 2.2 Evaluate Responses with an LLM Judge

We'll use the `UltraFeedback` task to evaluate each response pair. This task uses an LLM to judge responses across multiple dimensions (helpfulness, honesty, etc.). Here, we'll use the `meta-llama/Meta-Llama-3.1-70B-Instruct` model via the Hugging Face Inference API.

```python
evaluate_responses = UltraFeedback(
    aspect="overall-rating",
    llm=InferenceEndpointsLLM(
        model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
        generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
    ),
    pipeline=Pipeline(name="clean-dataset-pipeline"),
)
```

### 2.3 Keep Only Relevant Columns

After evaluation, we'll use `KeepColumns` to retain only the columns needed for further analysis and curation.

```python
keep_columns = KeepColumns(
    columns=[
        "instruction",
        "generations",
        "order",
        "ratings",
        "rationales",
        "model_name",
    ],
    pipeline=Pipeline(name="clean-dataset-pipeline"),
)
```

### 2.4 (Optional) Curate Data with Argilla

If you have an Argilla instance deployed, you can send the evaluated data for human review. Update the `api_url` and `api_key` with your own credentials.

```python
to_argilla = PreferenceToArgilla(
    dataset_name="cleaned-dataset",
    dataset_workspace="argilla",
    api_url="https://[your-owner-name]-[your-space-name].hf.space",
    api_key="[your-api-key]",
    num_generations=2
)
```

## Step 3: Assemble and Run the Pipeline

Now, connect all the steps into a pipeline and execute it.

```python
with Pipeline(name="clean-dataset") as pipeline:

    load_dataset = LoadDataFromDicts(
        data=dataset, output_mappings={"question": "instruction"}
    )

    evaluate_responses = UltraFeedback(
        aspect="overall-rating",
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            generation_kwargs={"max_new_tokens": 512, "temperature": 0.7},
        ),
    )

    keep_columns = KeepColumns(
        columns=[
            "instruction",
            "generations",
            "order",
            "ratings",
            "rationales",
            "model_name",
        ]
    )

    to_argilla = PreferenceToArgilla(
        dataset_name="cleaned-dataset",
        dataset_workspace="argilla",
        api_url="https://[your-owner-name]-[your-space-name].hf.space",
        api_key="[your-api-key]",
        num_generations=2,
    )

    # Connect the steps
    load_dataset.connect(evaluate_responses)
    evaluate_responses.connect(keep_columns)
    keep_columns.connect(to_argilla)

# Run the pipeline
distiset = pipeline.run()
```

The pipeline will process the dataset, evaluate each response pair, and output a cleaned dataset with AI-generated ratings and rationales.

## Step 4: Review and Share the Results

If you used Argilla, you can now log into the Argilla UI to review and annotate the data further.

Finally, you can push the cleaned dataset to the Hugging Face Hub to share with the community:

```python
distiset.push_to_hub("[your-owner-name]/example-cleaned-preference-dataset")
```

## Conclusion

In this tutorial, you built a distilabel pipeline to clean a preference dataset using AI feedback. You learned how to:

1. Load and preprocess a preference dataset.
2. Evaluate response quality using an LLM judge via the Hugging Face Inference API.
3. Filter and structure the results.
4. Optionally send data to Argilla for human curation.

This workflow can be adapted for other tasks, such as cleaning instruction-tuning (SFT) datasets or integrating custom evaluation steps. For more details, explore the [distilabel documentation](https://distilabel.argilla.io/latest/).