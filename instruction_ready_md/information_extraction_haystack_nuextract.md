# Information Extraction with Haystack and NuExtract

*Authored by: [Stefano Fiorucci](https://github.com/anakin87)*

In this guide, you will learn how to automate structured information extraction from text and web pages using a Language Model. You'll build an application that extracts specific data points following a user-defined schema.

**Goal:** Create a pipeline that fetches content from URLs and extracts structured information into a consistent JSON format.

**Stack:**
*   **[Haystack](https://haystack.deepset.ai)**: An orchestration framework for building LLM applications.
*   **[NuExtract](https://huggingface.co/numind/NuExtract)**: A 3.8B parameter language model fine-tuned for structured data extraction.

## 1. Setup and Installation

First, install the required Python libraries.

```bash
pip install haystack-ai trafilatura transformers pyvis
```

Now, import the necessary modules.

```python
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.builders import PromptBuilder
from haystack.components.converters import OutputAdapter
from haystack import Pipeline, Document
import torch
import json
```

## 2. Understanding the Pipeline Components

A Haystack pipeline is built from reusable **Components**. Let's examine each one you'll use.

### Component 1: Fetching Web Content
The `LinkContentFetcher` downloads content from URLs, and `HTMLToDocument` converts the raw HTML into clean text `Document` objects.

```python
# Initialize the components
fetcher = LinkContentFetcher()
converter = HTMLToDocument()

# Test them on a sample URL
streams = fetcher.run(urls=["https://example.com/"])["streams"]
docs = converter.run(sources=streams)
print(docs)
```

### Component 2: The Extraction Model
You'll use `HuggingFaceLocalGenerator` to load the NuExtract model. This model is specifically trained to output JSON matching a provided template.

```python
generator = HuggingFaceLocalGenerator(
    model="numind/NuExtract",
    huggingface_pipeline_kwargs={
        "model_kwargs": {"torch_dtype": torch.bfloat16} # Uses less memory
    }
)
# Load the model into memory
generator.warm_up()
```

**Note on Performance:** If your GPU supports it, you can install Flash Attention for a significant speed boost:
```bash
pip install flash-attn --no-build-isolation
```
Then add `"attn_implementation": "flash_attention_2"` to the `model_kwargs` dictionary.

### Component 3: Dynamic Prompt Creation
The `PromptBuilder` uses a Jinja2 template to create the precise prompt format the NuExtract model expects. Proper indentation of the JSON schema is crucial for good results.

```python
prompt_template = '''<|input|>
### Template:
{{ schema | tojson(indent=4) }}
{% for example in examples %}
### Example:
{{ example | tojson(indent=4) }}\n
{% endfor %}
### Text
{{documents[0].content}}
<|output|>
'''

prompt_builder = PromptBuilder(template=prompt_template)
```

### Component 4: Formatting the Output
The model's reply is a JSON string inside a list. The `OutputAdapter` converts this into a usable Python dictionary.

```python
adapter = OutputAdapter(
    template="""{{ replies[0]| replace("'",'"') | json_loads}}""",
    output_type=dict,
    custom_filters={"json_loads": json.loads}
)
```

## 3. Building the Information Extraction Pipeline

Now, connect all the components into a sequential pipeline.

```python
# Create a new pipeline
ie_pipe = Pipeline()

# Add all components
ie_pipe.add_component("fetcher", fetcher)
ie_pipe.add_component("converter", converter)
ie_pipe.add_component("prompt_builder", prompt_builder)
ie_pipe.add_component("generator", generator)
ie_pipe.add_component("adapter", adapter)

# Connect the components to define the data flow
ie_pipe.connect("fetcher", "converter")
ie_pipe.connect("converter", "prompt_builder")
ie_pipe.connect("prompt_builder", "generator")
ie_pipe.connect("generator", "adapter")
```

You can visualize your pipeline's structure:

```python
ie_pipe.show()
```

## 4. Defining Your Extraction Task

### Step 1: Choose Your Data Sources
Create a list of URLs you want to extract information from. Here are examples related to startup funding announcements.

```python
urls = [
    "https://techcrunch.com/2023/04/27/pinecone-drops-100m-investment-on-750m-valuation-as-vector-database-demand-grows/",
    "https://techcrunch.com/2023/04/27/replit-funding-100m-generative-ai/",
    "https://www.cnbc.com/2024/06/12/mistral-ai-raises-645-million-at-a-6-billion-valuation.html",
    # ... Add more URLs as needed
]
```

### Step 2: Define Your Extraction Schema
Create a JSON schema that defines the exact structure you want to extract. The keys in this dictionary will become the keys in your output.

```python
schema = {
    "Funding": {
        "New funding": "",
        "Investors": [],
    },
     "Company": {
        "Name": "",
        "Activity": "",
        "Country": "",
        "Total valuation": "",
        "Total funding": ""
    }
}
```

## 5. Running the Pipeline

Execute the pipeline for each URL. The process will: fetch the webpage, convert it to text, build a prompt with your schema, run the extraction model, and format the output.

```python
from tqdm import tqdm

extracted_data = []

for url in tqdm(urls):
    result = ie_pipe.run({
        "fetcher": {"urls": [url]},
        "prompt_builder": {"schema": schema}
    })
    extracted_data.append(result["adapter"]["output"])
```

## 6. Inspecting and Using the Results

The `extracted_data` list now contains structured dictionaries for each processed URL.

```python
# View the first two results
print(extracted_data[:2])
```

Example output:
```python
[
    {
        'Company': {
            'Activity': 'vector database',
            'Country': '',
            'Name': 'Pinecone',
            'Total funding': '$138 million',
            'Total valuation': '$750 million'
        },
        'Funding': {
            'Investors': ['Andreessen Horowitz', 'ICONIQ Growth', 'Menlo Ventures', 'Wing Venture Capital'],
            'New funding': '$100 million'
        }
    },
    # ... more results
]
```

You can now analyze this structured data, convert it to a Pandas DataFrame for further analysis, or store it in a database.

## Summary

You have successfully built an automated information extraction pipeline using Haystack and the NuExtract model. This pipeline can be easily adapted to extract different types of structured information from any text source by simply modifying the `schema` dictionary and the list of `urls`.

**Key Takeaways:**
1.  Haystack **Components** are modular building blocks for LLM applications.
2.  Haystack **Pipelines** chain components together to create complex workflows.
3.  The **NuExtract model** is specialized for converting unstructured text into structured JSON.
4.  The **PromptBuilder** dynamically creates model instructions based on your target schema.
5.  The entire process—from URL to structured data—is automated and reproducible.

To extend this application, consider adding a data validation step, integrating the results into a dashboard, or using the extracted data to power a search system.