# Evals API: Image Inputs

This cookbook demonstrates how to use OpenAI's Evals framework for image-based tasks. Leveraging the Evals API, we will grade model-generated responses to an image and prompt by using **sampling** to generate model responses and **model grading** (LLM as a Judge) to score the model responses against the image, prompt, and reference answer.

In this example, we will evaluate how well our model can:
1. **Generate appropriate responses** to user prompts about images
3. **Align with reference answers** that represent high-quality responses

## Installing Dependencies + Setup

```python
# Install required packages
!pip install openai datasets pandas --quiet
```

```python
# Import libraries
from datasets import load_dataset
from openai import OpenAI
import os
import json
import time
import pandas as pd
```

## Dataset Preparation

We use the [VibeEval](https://huggingface.co/datasets/RekaAI/VibeEval) dataset that's hosted on Hugging Face. It contains a collection of user prompt, accompanying image, and reference answer data. First, we load the dataset.

```python
dataset = load_dataset("RekaAI/VibeEval")
```

We extract the relevant fields and put it in a json-like format to pass in as a data source in the Evals API. Input image data can be in the form of a web URL or a base64 encoded string. Here, we use the provided web URLs.

```python
evals_data_source = []

# select the first 3 examples in the dataset to use for this cookbook
for example in dataset["test"].select(range(3)):
    evals_data_source.append({
        "item": {
            "media_url": example["media_url"], # image web URL
            "reference": example["reference"], # reference answer
            "prompt": example["prompt"] # prompt
        }
    })
```

If you print the data source list, each item should be of a similar form to:

```python
{
  "item": {
    "media_url": "https://storage.googleapis.com/reka-annotate.appspot.com/vibe-eval/difficulty-normal_food1_7e5c2cb9c8200d70.jpg"
    "reference": "This appears to be a classic Margherita pizza, which has the following ingredients..."
    "prompt": "What ingredients do I need to make this?"
  }
}
```

## Eval Configuration

Now that we have our data source and task, we will create our evals. For the OpenAI Evals API docs, visit [API docs](https://platform.openai.com/docs/evals/overview).

```python
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
```

Evals have two parts, the "Eval" and the "Run". In the "Eval", we define the expected structure of the data and the testing criteria (grader).

### Data Source Config

Based on the data that we have compiled, our data source config is as follows:

```python
data_source_config = {
    "type": "custom",
    "item_schema": {
        "type": "object",
        "properties": {
          "media_url": { "type": "string" },
          "reference": { "type": "string" },
          "prompt": { "type": "string" }
        },
        "required": ["media_url", "reference", "prompt"]
      },
    "include_sample_schema": True, # enables sampling
}
```

### Testing Criteria

For our testing criteria, we set up our grader config. In this example, it is a model grader that takes in an image, reference answer, and sampled model response (in the `sample` namespace), and then outputs a score between 0 and 1 based on how closely the model response matches the reference answer and its general suitability for the conversation. For more info on model graders, visit [API Grader docs](https://platform.openai.com/docs/api-reference/graders).

Getting the both the data and the grader right are key for an effective evaluation. So, you will likely want to iteratively refine the prompts for your graders.

**Note**: The image url field / templating need to be placed in an input image object to be interpreted as an image. Otherwise, the image will be interpreted as a text string.

```python
grader_config = {
	    "type": "score_model",
        "name": "Score Model Grader",
        "input":[
            {
                "role": "system",
		        "content": "You are an expert grader. Judge how well the model response suits the image and prompt as well as matches the meaniing of the reference answer. Output a score of 1 if great. If it's somewhat compatible, output a score around 0.5. Otherwise, give a score of 0."
	        },
	        {
		        "role": "user",
		        "content": [{ "type": "input_text", "text": "Prompt: {{ item.prompt }}."},
							{ "type": "input_image", "image_url": "{{ item.media_url }}", "detail": "auto" },
							{ "type": "input_text", "text": "Reference answer: {{ item.reference }}. Model response: {{ sample.output_text }}."}
				]
	        }
		],
		"pass_threshold": 0.9,
	    "range": [0, 1],
	    "model": "o4-mini" # model for grading; check that the model you use supports image inputs
	}
```

Now, we create the eval object.

```python
eval_object = client.evals.create(
        name="Image Grading",
        data_source_config=data_source_config,
        testing_criteria=[grader_config],
    )
```

## Eval Run

To create the run, we pass in the eval object id, the data source (i.e., the data we compiled earlier), and the chat message input we will use for sampling to generate the model response. Note that EvalsAPI also supports stored completions and responses containing images as a data source. See the [Additional Info: Logs Data Source](#additional-info-logs-data-source) section for more info.

Here's the sampling message input we'll use for this example.

```python
sampling_messages = [{
    "role": "user",
    "type": "message",
    "content": {
        "type": "input_text",
        "text": "{{ item.prompt }}"
      }
  },
  {
    "role": "user",
    "type": "message",
    "content": {
        "type": "input_image",
        "image_url": "{{ item.media_url }}",
        "detail": "auto"
    }
  }]
```

We now kickoff an eval run.

```python
eval_run = client.evals.runs.create(
        name="Image Input Eval Run",
        eval_id=eval_object.id,
        data_source={
            "type": "responses", # sample using responses API
            "source": {
                "type": "file_content",
                "content": evals_data_source
            },
            "model": "gpt-4o-mini", # model used to generate the response; check that the model you use supports image inputs
            "input_messages": {
                "type": "template",
                "template": sampling_messages}
        }
    )
```

## Poll and Display the Results

When the run finishes, we can take a look at the result. You can also check in your org's OpenAI evals dashboard to see the progress and results.

```python
while True:
    run = client.evals.runs.retrieve(run_id=eval_run.id, eval_id=eval_object.id)
    if run.status == "completed" or run.status == "failed": # check if the run is finished
        output_items = list(client.evals.runs.output_items.list(
            run_id=run.id, eval_id=eval_object.id
        ))
        df = pd.DataFrame({
                "prompt": [item.datasource_item["prompt"]for item in output_items],
                "reference": [item.datasource_item["reference"] for item in output_items],
                "model_response": [item.sample.output[0].content for item in output_items],
                "grading_results": [item.results[0]["sample"]["output"][0]["content"]
                                    for item in output_items]
            })
        display(df)
        break
    time.sleep(5)
```

| prompt | reference | model_response | grading_results |
|--------|-----------|----------------|-----------------|
| Please provide latex code to replicate this table | Below is the latex code for your table:\n```te... | Certainly! Below is the LaTeX code to replicat... | {"steps":[{"description":"Assess if the provid... |
| What ingredients do I need to make this? | This appears to be a classic Margherita pizza,... | To make a classic Margherita pizza like the on... | {"steps":[{"description":"Check if model ident... |
| Is this safe for a vegan to eat? | Based on the image, this dish appears to be a ... | To determine if the dish is safe for a vegan t... | {"steps":[{"description":"Compare model respon... |

### Viewing Individual Output Items

To see a full output item, we can do the following. The structure of an output item is specified in the API docs [here](https://platform.openai.com/docs/api-reference/evals/run-output-item-object).

```python
first_item = output_items[0]

print(json.dumps(dict(first_item), indent=2, default=str))
```

```
{
  "id": "outputitem_687833f102ec8191a6e53a5461b970c2",
  "created_at": 1752708081,
  "datasource_item": {
    "prompt": "Please provide latex code to replicate this table",
    "media_url": "https://storage.googleapis.com/reka-annotate.appspot.com/vibe-eval/difficulty-normal_table0_b312eea68bcd0de6.png",
    "reference": "Below is the latex code for your table:\n```tex\n\\begin{table}\n\\begin{tabular}{c c c c} \\hline  & \\(S2\\) & Expert & Layman & PoelM \\\\ \\cline{2-4} \\(S1\\) & Expert & \u2013 & 54.0 & 62.7 \\\\  & Layman & 46.0 & \u2013 & 60.7 \\\\  &,PoelM,LM,LM,LM,LM,LM,,L,M,,L,M,,L,M,,L,M,,,\u2013&39.3 \\\\\n[-1ex] \\end{tabular}\n\\end{table}\n```."
  },
  "datasource_item_id": 1,
  "eval_id": "eval_687833d68e888191bc4bd8b965368f22",
  "object": "eval.run.output_item",
  "results": [
    {
      "name": "Score Model Grader-73fe48a0-8090-46eb-aa8e-d426ad074eb3",
      "sample": {
        "input": [
          {
            "role": "system",
            "content": "You are an expert grader. Judge how well the model response suits the image and prompt as well as matches the meaniing of the reference answer. Output a score of 1 if great. If it's somewhat compatible, output a score around 0.5. Otherwise, give a score of 0."
          },
          {
            "role": "user",
            "content": "Prompt: Please provide latex code to replicate this table. <image>https://storage.googleapis.com/reka-annotate.appspot.com/vibe-eval/difficulty-normal_table0_b312eea68bcd0de6.png</image> Reference answer: Below is the latex code for your table:\n```tex\n\\begin{table}\n\\begin{tabular}{c c c c} \\hline  & \\(S2\\) & Expert & Layman & PoelM \\\\ \\cline{2-4} \\(S1\\) & Expert & \u2013 & 54.0 & 62.7 \\\\  & Layman & 46.0 & \u2013 & 60.7 \\\\  &,PoelM,LM,LM,LM,LM,LM,,L,M,,L,M,,L,M,,L,M,,,\u2013&39.3 \\\\\n[-1ex] \\end{tabular}\n\\end{table}\n```.. Model response: Certainly! Below is the LaTeX code to replicate the table you provided:\n\n```latex\n\\documentclass{article}\n\\usepackage{array}\n\\usepackage{multirow}\n\\usepackage{booktabs}\n\n\\begin{document}\n\n\\begin{table}[ht]\n    \\centering\n    \\begin{tabular}{c|c|c|c}\n        \\multirow{2}{*}{S1} & \\multirow{2}{*}{S2} & \\multicolumn{3}{c}{Methods} \\\\ \n        \\cline{3-5}\n        & & Expert & Layman & PoeLM \\\\\n        \\hline\n        Expert & & - & 54.0 & 62.7 \\\\\n        Layman & & 46.0 & - & 60.7 \\\\\n        PoeLM & & 37.3 & 39.3 & - \\\\\n    \\end{tabular}\n    \\caption{Comparison of different methods}\n    \\label{tab:methods_comparison}\n\\end{table}\n\n\\end{document}\n```\n\n### Explanation:\n- The `multirow` package is used to create the multi-row header for `S1` and `S2`.\n- The `booktabs` package is used for improved table formatting (with `\\hline` for horizontal lines).\n- Adjust the table's caption and label as needed.."
          }
        ],
        "output": [
          {
            "role": "assistant",
            "content": "{\"steps\":[{\"description\":\"Assess if the provided LaTeX code correctly matches the structure of the target table, including the diagonal header, column counts, and alignment.\",\"conclusion\":\"The code fails to create the diagonal split between S1 and S2 and mismatches column counts (defines 4 columns but uses 5).\"},{\"description\":\"Check the header layout: the target table has a single diagonal cell spanning two axes and three following columns labeled Expert, Layman, PoeLM. The model uses \\\\multirow and a \\\\multicolumn block named 'Methods', which does not replicate the diagonal or correct labeling.\",\"conclusion\":\"Header structure is incorrect and does not match the prompt's table.\"},{\"description\":\"Verify the data rows: the model code includes two empty cells after S1 and before the data, misaligning all data entries relative to the intended columns.\",\"conclusion\":\"Data rows are misaligned due to incorrect column definitions.\"},{\"description\":\"Overall compatibility: the code is syntactically flawed for the target table and conceptually does not replicate the diagonal header or correct column count.\",\"conclusion\":\"The response does not satisfy the prompt.\"}],\"result\":0.0}"
          }
        ],
        "finish_reason": "stop",
        "model": "o4-mini-2025-04-16",
        "usage": {
          "total_tokens": 2185,
          "completion_tokens": 712,
          "prompt_tokens": 1473,
          "cached_tokens": 0
        },
        "error": null,
        "seed": null,
        "temperature": 1.0,
        "top_p": 1.0,
        "reasoning_effort": null,
        "max_completions_tokens": 4096
      },
      "passed": false,
      "score": 0.0
    }
  ],
  "run_id": "evalrun_687833dbadd081919a0f9fbfb817baf4",
  "sample": "Sample(error=None, finish_reason='stop', input=[SampleInput(content='Please provide latex code to replicate this table', role='user'), SampleInput(content='<image>https://storage.googleapis.com/reka-annotate.appspot.com/vibe-eval/difficulty-normal_table0_b312eea68bcd0de6.png</image>', role='user')], max_completion_tokens=None, model='gpt-4o-mini-2024-07-18', output=[SampleOutput(content=\"Certainly! Below is the LaTeX code to replicate the table you provided:\\n\\n```latex\\n\\\\documentclass{article}\\n\\\\usepackage{array}\\n\\\\usepackage{multirow}\\n\\\\usepackage{booktabs}\\n\\n\\\\begin{document}\\n\\n\\\\begin{table}[ht]\\n    \\\\centering\\n    \\\\begin{tabular}{c|c|c|c}\\n        \\\\multirow{2}{*}{S1} & \\\\multirow{2}{*}{S2} & \\\\multicolumn{3}{c}{Methods} \\\\\\\\ \\n        \\\\cline{3-5}\\n        & & Expert & Layman & PoeLM \\\\\\\\\\n        \\\\hline\\n        Expert & & - & 54.0 & 62.7 \\\\\\\\\\n        Layman & & 46.0 & - & 60.7 \\\\\\\\\\n        PoeLM & & 37.3 & 39.3 & - \\\\\\\\\\n    \\\\end{tabular}\\n    \\\\caption{Comparison of different methods}\\n    \\\\label{tab:methods_comparison}\\n\\\\end{table}\\n\\n\\\\end{document}\\n```\\n\\n### Explanation:\\n- The `multirow` package is used to create the multi-row header for `S1` and `S2`.\\n- The `booktabs` package is used for improved table formatting (with `\\\\hline` for horizontal lines).\\n- Adjust the table's caption and label as needed.\", role='assistant')], seed=None, temperature=1.0, top_p=1.0, usage=SampleUsage(cached_tokens=0, completion_tokens=295, prompt_tokens=14187, total_tokens=14482), max_completions_tokens=4096)",
  "status": "fail",
  "_datasource_item_content_hash": "bb2090df47ea2ca0aa67337709ce2ff7382d639118d3358068b0cc7031c12f82"
}
```

## Additional Info: Logs Data Source

As mentioned earlier, EvalsAPI supports