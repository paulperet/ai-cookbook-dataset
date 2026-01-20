# Evaluating MCP-Based Answers with a Custom Dataset

This notebook evaluates a model's ability to answer questions about the [tiktoken](https://github.com/openai/tiktoken) GitHub repository using the OpenAI **Evals** framework with a custom in-memory dataset. 

We use a custom, in-memory dataset of Q&A pairs and compare two models: `gpt-4.1` and `o4-mini`, that leverage the **MCP** tool for repository-aware, contextually accurate answers.

**Goals:**
- Show how to set up and run an evaluation using OpenAI Evals with a custom dataset.
- Compare the performance of different models leveraging MCP-based tools.
- Provide best practices for professional, reproducible evaluation workflows.

_Next: We will set up our environment and import the necessary libraries._


```python
# Update OpenAI client
%pip install --upgrade openai --quiet
```

    
    [notice] A new release of pip is available: 24.0 -> 25.1.1
    [notice] To update, run: pip install --upgrade pip
    Note: you may need to restart the kernel to use updated packages.


## Environment Setup

We begin by importing the required libraries and configuring the OpenAI client.  
This step ensures we have access to the OpenAI API and all necessary utilities for evaluation.


```python
import os
import time

from openai import OpenAI

# Instantiate the OpenAI client (no custom base_url).
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("_OPENAI_API_KEY"),
)
```

## Define the Custom Evaluation Dataset

We define a small, in-memory dataset of question-answer pairs about the `tiktoken` repository.  
This dataset will be used to test the models' ability to provide accurate and relevant answers with the help of the MCP tool.

- Each item contains a `query` (the user’s question) and an `answer` (the expected ground truth).
- You can modify or extend this dataset to suit your own use case or repository.



```python
def get_dataset(limit=None):
    items = [
        {
            "query": "What is tiktoken?",
            "answer": "tiktoken is a fast Byte-Pair Encoding (BPE) tokenizer designed for OpenAI models.",
        },
        {
            "query": "How do I install the open-source version of tiktoken?",
            "answer": "Install it from PyPI with `pip install tiktoken`.",
        },
        {
            "query": "How do I get the tokenizer for a specific OpenAI model?",
            "answer": 'Call tiktoken.encoding_for_model("<model-name>"), e.g. tiktoken.encoding_for_model("gpt-4o").',
        },
        {
            "query": "How does tiktoken perform compared to other tokenizers?",
            "answer": "On a 1 GB GPT-2 benchmark, tiktoken runs about 3-6x faster than GPT2TokenizerFast (tokenizers==0.13.2, transformers==4.24.0).",
        },
        {
            "query": "Why is Byte-Pair Encoding (BPE) useful for language models?",
            "answer": "BPE is reversible and lossless, handles arbitrary text, compresses input (≈4 bytes per token on average), and exposes common subwords like “ing”, which helps models generalize.",
        },
    ]
    return items[:limit] if limit else items
```

### Define Grading Logic

To evaluate the model’s answers, we use two graders:

- **Pass/Fail Grader (LLM-based):**  
  An LLM-based grader that checks if the model’s answer matches the expected answer (ground truth) or conveys the same meaning.
- **Python MCP Grader:**  
  A Python function that checks whether the model actually used the MCP tool during its response (for auditing tool usage).

  > **Best Practice:**  
  > Using both LLM-based and programmatic graders provides a more robust and transparent evaluation.



```python
# LLM-based pass/fail grader: instructs the model to grade answers as "pass" or "fail".
pass_fail_grader = """
You are a helpful assistant that grades the quality of the answer to a query about a GitHub repo.
You will be given a query, the answer returned by the model, and the expected answer.
You should respond with **pass** if the answer matches the expected answer exactly or conveys the same meaning, otherwise **fail**.
"""

# User prompt template for the grader, providing context for grading.
pass_fail_grader_user_prompt = """
<Query>
{{item.query}}
</Query>

<Web Search Result>
{{sample.output_text}}
</Web Search Result>

<Ground Truth>
{{item.answer}}
</Ground Truth>
"""


# Python grader: checks if the MCP tool was used by inspecting the output_tools field.
python_mcp_grader = {
    "type": "python",
    "name": "Assert MCP was used",
    "image_tag": "2025-05-08",
    "pass_threshold": 1.0,
    "source": """
def grade(sample: dict, item: dict) -> float:
    output = sample.get('output_tools', [])
    return 1.0 if len(output) > 0 else 0.0
""",
}
```

## Define the Evaluation Configuration

We now configure the evaluation using the OpenAI Evals framework.  

This step specifies:
- The evaluation name and dataset.
- The schema for each item (what fields are present in each Q&A pair).
- The grader(s) to use (LLM-based and/or Python-based).
- The passing criteria and labels.

> **Best Practice:**  
> Clearly defining your evaluation schema and grading logic up front ensures reproducibility and transparency.


```python
# Create the evaluation definition using the OpenAI Evals client.
logs_eval = client.evals.create(
    name="MCP Eval",
    data_source_config={
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "answer": {"type": "string"},
            },
        },
        "include_sample_schema": True,
    },
    testing_criteria=[
        {
            "type": "label_model",
            "name": "General Evaluator",
            "model": "o3",
            "input": [
                {"role": "system", "content": pass_fail_grader},
                {"role": "user", "content": pass_fail_grader_user_prompt},
            ],
            "passing_labels": ["pass"],
            "labels": ["pass", "fail"],
        },
        python_mcp_grader
    ],
)
```

## Run Evaluations for Each Model

We now run the evaluation for each model (`gpt-4.1` and `o4-mini`).  

Each run is configured to:
- Use the MCP tool for repository-aware answers.
- Use the same dataset and evaluation configuration for fair comparison.
- Specify model-specific parameters (such as max completions tokens, and allowed tools).

> **Best Practice:**  
> Keeping the evaluation setup consistent across models ensures results are comparable and reliable.


```python
# Run 1: gpt-4.1 using MCP
gpt_4one_responses_run = client.evals.runs.create(
    name="gpt-4.1",
    eval_id=logs_eval.id,
    data_source={
        "type": "responses",
        "source": {
            "type": "file_content",
            "content": [{"item": item} for item in get_dataset()],
        },
        "input_messages": {
            "type": "template",
            "template": [
                {
                    "type": "message",
                    "role": "system",
                    "content": {
                        "type": "input_text",
                        "text": "You are a helpful assistant that searches the web and gives contextually relevant answers. Never use your tools to answer the query.",
                    },
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": {
                        "type": "input_text",
                        "text": "Search the web for the answer to the query {{item.query}}",
                    },
                },
            ],
        },
        "model": "gpt-4.1",
        "sampling_params": {
            "seed": 42,
            "temperature": 0.7,
            "max_completions_tokens": 10000,
            "top_p": 0.9,
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "gitmcp",
                    "server_url": "https://gitmcp.io/openai/tiktoken",
                    "allowed_tools": [
                        "search_tiktoken_documentation",
                        "fetch_tiktoken_documentation",
                    ],
                    "require_approval": "never",
                }
            ],
        },
    },
)
```


```python
# Run 2: o4-mini using MCP
gpt_o4_mini_responses_run = client.evals.runs.create(
    name="o4-mini",
    eval_id=logs_eval.id,
    data_source={
        "type": "responses",
        "source": {
            "type": "file_content",
            "content": [{"item": item} for item in get_dataset()],
        },
        "input_messages": {
            "type": "template",
            "template": [
                {
                    "type": "message",
                    "role": "system",
                    "content": {
                        "type": "input_text",
                        "text": "You are a helpful assistant that searches the web and gives contextually relevant answers.",
                    },
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": {
                        "type": "input_text",
                        "text": "Search the web for the answer to the query {{item.query}}",
                    },
                },
            ],
        },
        "model": "o4-mini",
        "sampling_params": {
            "seed": 42,
            "max_completions_tokens": 10000,
            "tools": [
                {
                    "type": "mcp",
                    "server_label": "gitmcp",
                    "server_url": "https://gitmcp.io/openai/tiktoken",
                    "allowed_tools": [
                        "search_tiktoken_documentation",
                        "fetch_tiktoken_documentation",
                    ],
                    "require_approval": "never",
                }
            ],
        },
    },
)
```

## Poll for Completion and Retrieve Outputs

After launching the evaluation runs, we can poll the run until they are complete.

This step ensures that we are analyzing results only after all model responses have been processed.

> **Best Practice:**  
> Polling with a delay avoids excessive API calls and ensures efficient resource usage.


```python
def poll_runs(eval_id, run_ids):
    while True:
        runs = [client.evals.runs.retrieve(rid, eval_id=eval_id) for rid in run_ids]
        for run in runs:
            print(run.id, run.status, run.result_counts)
        if all(run.status in {"completed", "failed"} for run in runs):
            break
        time.sleep(5)
    
# Start polling both runs.
poll_runs(logs_eval.id, [gpt_4one_responses_run.id, gpt_o4_mini_responses_run.id])
```

    [evalrun_684769b577488191863b5a51cf4db57a completed ResultCounts(errored=0, failed=5, passed=0, total=5), ..., evalrun_684769c1ad9c8191affea5aa02ef1215 completed ResultCounts(errored=0, failed=3, passed=2, total=5)]


## Display and Interpret Model Outputs

Finally, we display the outputs from each model for manual inspection and further analysis.

- Each model's answers are printed for each question in the dataset.
- You can compare the outputs side-by-side to assess quality, relevance, and correctness.

Below are screenshots from the OpenAI Evals Dashboard illustrating the evaluation outputs for both models:

For a comprehensive breakdown of the evaluation metrics and results, navigate to the "Data" tab in the dashboard:

Note that the 4.1 model was constructed to never use its tools to answer the query thus it never called the MCP server. The o4-mini model wasn't explicitly instructed to use it's tools either but it wasn't forbidden, thus it called the MCP server 3 times. We can see that the 4.1 model performed worse than the o4 model. Also notable is the one example that the o4-mini model failed was one where the MCP tool was not used.

We can also check a detailed analysis of the outputs from each model for manual inspection and further analysis.


```python
four_one_output = client.evals.runs.output_items.list(
    run_id=gpt_4one_responses_run.id, eval_id=logs_eval.id
)

o4_mini_output = client.evals.runs.output_items.list(
    run_id=gpt_o4_mini_responses_run.id, eval_id=logs_eval.id
)
```


```python
print('# gpt‑4.1 Output')
for item in four_one_output:
    print(item.sample.output[0].content)

print('\n# o4-mini Output')
for item in o4_mini_output:
    print(item.sample.output[0].content)
```

    # gpt‑4.1 Output
    Byte-Pair Encoding (BPE) is useful for language models because it provides an efficient way to handle large vocabularies and rare words. Here’s why it is valuable:
    
    1. **Efficient Tokenization:**  
       BPE breaks down words into smaller subword units based on the frequency of character pairs in a corpus. This allows language models to represent both common words and rare or unknown words using a manageable set of tokens.
    
    2. **Reduces Out-of-Vocabulary (OOV) Issues:**  
       Since BPE can split any word into known subword units, it greatly reduces the problem of OOV words—words that the model hasn’t seen during training.
    
    3. **Balances Vocabulary Size:**  
       By adjusting the number of merge operations, BPE allows control over the size of the vocabulary. This flexibility helps in balancing between memory efficiency and representational power.
    
    4. **Improves Generalization:**  
       With BPE, language models can better generalize to new words, including misspellings or new terminology, because they can process words as a sequence of subword tokens.
    
    5. **Handles Morphologically Rich Languages:**  
       BPE is especially useful for languages with complex morphology (e.g., agglutinative languages) where words can have many forms. BPE reduces the need to memorize every possible word form.
    
    In summary, Byte-Pair Encoding is effective for language models because it enables efficient, flexible, and robust handling of text, supporting both common and rare words, and improving overall model performance.
    **Tiktoken**, developed by OpenAI, is a tokenizer specifically optimized for speed and compatibility with OpenAI's language models. Here’s how it generally compares to other popular tokenizers:
    
    ### Performance
    - **Speed:** Tiktoken is significantly faster than most other Python-based tokenizers. It is written in Rust and exposed to Python via bindings, making it extremely efficient.
    - **Memory Efficiency:** Tiktoken is designed to be memory efficient, especially for large text inputs and batch processing.
    
    ### Accuracy and Compatibility
    - **Model Alignment:** Tiktoken is tailored to match the tokenization logic used by OpenAI’s GPT-3, GPT-4, and related models. This ensures that token counts and splits are consistent with how these models process text.
    - **Unicode Handling:** Like other modern tokenizers (e.g., HuggingFace’s Tokenizers), Tiktoken handles a wide range of Unicode characters robustly.
    
    ### Comparison to Other Tokenizers
    - **HuggingFace Tokenizers:** HuggingFace’s library is very flexible and supports a wide range of models (BERT, RoBERTa, etc.). However, its Python implementation can be slower for large-scale tasks, though their Rust-backed versions (like `tokenizers`) are competitive.
    - **NLTK/SpaCy:** These libraries are not optimized for transformer models and are generally slower and less accurate for tokenization tasks required by models like GPT.
    - **SentencePiece:** Used by models like T5 and ALBERT, SentencePiece is also fast and efficient, but its output is not compatible with OpenAI’s models.
    
    ### Use Cases
    - **Best for OpenAI Models:** If you are working with OpenAI’s APIs or models, Tiktoken is the recommended tokenizer due to its speed and alignment.
    - **General Purpose:** For non-OpenAI models, HuggingFace or SentencePiece might be preferable due to broader support.
    
    ### Benchmarks & Community Feedback
    - Multiple [community benchmarks](https://github.com/openai/tiktoken#performance) and [blog posts](https://www.philschmid.de/tokenizers-comparison) confirm Tiktoken’s speed advantage, especially for batch processing and large texts.
    
    **Summary:**  
    Tiktoken outperforms most tokenizers in speed when used with OpenAI models, with robust Unicode support and memory efficiency. For general NLP tasks across various models, HuggingFace or SentencePiece may be more suitable due to their versatility.
    
    **References:**  
    - [Tiktoken GitHub - Performance](https://github.com/openai/tiktoken#performance)
    - [Tokenizers Comparison Blog](https://www.philschmid.de/tokenizers-comparison)
    To get the tokenizer for a specific OpenAI model, you typically use the Hugging Face Transformers library, which provides easy access to tokenizers for OpenAI models like GPT-3, GPT-4, and others. Here’s how you can do it:
    
    **1. Using Hugging Face Transformers:**
    
    Install the library (if you haven’t already):
    ```bash
    pip install transformers
    ```
    
    **Example for GPT-3 (or GPT-4):**
    ```python
    from transformers import AutoTokenizer
    
    # For GPT-3 (davinci), use the corresponding model name
    tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
    
    # For GPT-4 (if available)
    # tokenizer = AutoTokenizer.from_pretrained("gpt-4")
    ```
    
    **2. Using OpenAI’s tiktoken library (for OpenAI API models):**
    
    Install tiktoken:
    ```bash
    pip install tiktoken
    ```
    
    Example for GPT-3.5-turbo or GPT-4:
    ```python
