# Guide: Implementing RAG with Source Highlighting via Structured Generation

## Overview
This guide demonstrates how to build a Retrieval-Augmented Generation (RAG) system that not only answers user queries but also highlights the specific source snippets supporting each answer. We achieve this using **structured generation** techniques, which force the LLM output to follow predefined constraints—in this case, a specific JSON schema.

We'll explore two approaches:
1. **Prompt-based structured generation** (naive approach)
2. **Grammar-constrained decoding** (robust approach)

## Prerequisites

First, install the required dependencies:

```bash
pip install pandas json huggingface_hub pydantic outlines accelerate -q
```

Then, import the necessary libraries:

```python
import pandas as pd
import json
from huggingface_hub import InferenceClient

pd.set_option("display.max_colwidth", None)
```

## Step 1: Initialize the LLM Client

We'll use HuggingFace's Inference API with a serverless endpoint. You can replace this with a dedicated endpoint if needed.

```python
repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llm_client = InferenceClient(model=repo_id, timeout=120)

# Test the connection
llm_client.text_generation(prompt="How are you today?", max_new_tokens=20)
```

## Step 2: Prompt-Based Structured Generation (Naive Approach)

### Define the Context and Prompt Template

First, let's create a sample context and a prompt template that instructs the model to output JSON:

```python
RELEVANT_CONTEXT = """
Document:

The weather is really nice in Paris today.
To define a stop sequence in Transformers, you should pass the stop_sequence argument in your pipeline or model.

"""

RAG_PROMPT_TEMPLATE_JSON = """
Answer the user query based on the source documents.

Here are the source documents: {context}


You should provide your answer as a JSON blob, and also provide all relevant short source snippets from the documents on which you directly based your answer, and a confidence score as a float between 0 and 1.
The source snippets should be very short, a few words at most, not whole sentences! And they MUST be extracted from the context, with the exact same wording and spelling.

Your answer should be built as follows, it must contain the "Answer:" and "End of answer." sequences.

Answer:
{{
  "answer": your_answer,
  "confidence_score": your_confidence_score,
  "source_snippets": ["snippet_1", "snippet_2", ...]
}}
End of answer.

Now begin!
Here is the user question: {user_query}.
Answer:
"""

USER_QUERY = "How can I define a stop sequence in Transformers?"
```

### Generate and Parse the Response

Now, format the prompt and generate a response:

```python
prompt = RAG_PROMPT_TEMPLATE_JSON.format(
    context=RELEVANT_CONTEXT, user_query=USER_QUERY
)

answer = llm_client.text_generation(
    prompt,
    max_new_tokens=1000,
)

# Extract the JSON portion
answer = answer.split("End of answer.")[0]
```

Parse the string into a dictionary:

```python
from ast import literal_eval
parsed_answer = literal_eval(answer)
```

### Display Results with Highlighting

Create a helper function to highlight the source snippets in the context:

```python
def highlight(s):
    return "\x1b[1;32m" + s + "\x1b[0m"

def print_results(answer, source_text, highlight_snippets):
    print("Answer:", highlight(answer))
    print("\n\n", "=" * 10 + " Source documents " + "=" * 10)
    for snippet in highlight_snippets:
        source_text = source_text.replace(snippet.strip(), highlight(snippet.strip()))
    print(source_text)

print_results(
    parsed_answer["answer"], RELEVANT_CONTEXT, parsed_answer["source_snippets"]
)
```

**Note:** This approach works well with powerful models at low temperatures. However, it can fail with less capable models or higher temperatures, as shown next.

### Testing the Limits

Let's increase the temperature to simulate a less reliable model:

```python
answer = llm_client.text_generation(
    prompt,
    max_new_tokens=250,
    temperature=1.6,
    return_full_text=False,
)
print(answer)
```

You'll notice the output may no longer be valid JSON, making it impossible to parse reliably.

## Step 3: Grammar-Constrained Decoding (Robust Approach)

To guarantee valid JSON output regardless of model behavior, we use **constrained decoding**. This forces the LLM to generate tokens that conform to a predefined grammar.

### Define the Output Schema with Pydantic

We'll define the exact structure we expect using Pydantic:

```python
from pydantic import BaseModel, confloat, StringConstraints
from typing import List, Annotated

class AnswerWithSnippets(BaseModel):
    answer: Annotated[str, StringConstraints(min_length=10, max_length=100)]
    confidence: Annotated[float, confloat(ge=0.0, le=1.0)]
    source_snippets: List[Annotated[str, StringConstraints(max_length=30)]]
```

Inspect the generated JSON schema to ensure it matches your requirements:

```python
AnswerWithSnippets.schema()
```

### Generate with Grammar Constraints

Now, generate a response while enforcing the grammar:

```python
# Using text_generation method
answer = llm_client.text_generation(
    prompt,
    grammar={"type": "json", "value": AnswerWithSnippets.schema()},
    max_new_tokens=250,
    temperature=1.6,
    return_full_text=False,
)
print(answer)
```

Alternatively, use the `post` method:

```python
data = {
    "inputs": prompt,
    "parameters": {
        "temperature": 1.6,
        "return_full_text": False,
        "grammar": {"type": "json", "value": AnswerWithSnippets.schema()},
        "max_new_tokens": 250,
    },
}
answer = json.loads(llm_client.post(json=data))[0]["generated_text"]
print(answer)
```

**Result:** Even with high temperature, the output is now guaranteed to be valid JSON with the correct keys and types.

## Step 4: Local Implementation with Outlines

[Outlines](https://github.com/outlines-dev/outlines/) is the library powering constrained generation in the Inference API. You can also use it locally.

### Set Up Local Model with Outlines

```python
import outlines

repo_id = "mustafaaljadery/gemma-2B-10M"
model = outlines.models.transformers(repo_id)

schema_as_str = json.dumps(AnswerWithSnippets.schema())
generator = outlines.generate.json(model, schema_as_str)

# Generate constrained output
result = generator(prompt)
print(result)
```

Outlines works by applying a bias to the model's logits, ensuring only tokens that conform to your grammar are selected.

## Additional Applications

Structured generation extends beyond RAG. For example, in LLM judge workflows, you can enforce outputs like:

```json
{
    "score": 1,
    "rationale": "The answer does not match the true answer at all.",
    "confidence_level": 0.85
}
```

This ensures consistent, machine-readable outputs for downstream processing.

## Conclusion

You've successfully built a RAG system with source highlighting using two structured generation methods:

1. **Prompt-based approach:** Simple but unreliable with weaker models or high temperatures.
2. **Grammar-constrained decoding:** Robust, guaranteeing valid JSON output regardless of model behavior.

The constrained decoding approach, whether via the Inference API or local Outlines, provides production-ready reliability for structured generation tasks.