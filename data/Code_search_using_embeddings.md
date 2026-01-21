# Guide: Implementing Semantic Code Search with Embeddings

This guide demonstrates how to build a semantic code search system using OpenAI's text embeddings. You'll learn to parse a code repository, generate embeddings for functions, and query them using natural language descriptions.

## Prerequisites

Ensure you have the following installed:

```bash
pip install pandas openai
```

You'll also need access to the OpenAI API and an API key set in your environment.

## Step 1: Set Up Helper Functions for Code Parsing

First, create utility functions to extract Python functions from your codebase.

```python
import pandas as pd
from pathlib import Path

DEF_PREFIXES = ['def ', 'async def ']
NEWLINE = '\n'

def get_function_name(code):
    """
    Extract function name from a line beginning with 'def' or 'async def'.
    """
    for prefix in DEF_PREFIXES:
        if code.startswith(prefix):
            return code[len(prefix): code.index('(')]

def get_until_no_space(all_lines, i):
    """
    Get all lines until a line outside the function definition is found.
    """
    ret = [all_lines[i]]
    for j in range(i + 1, len(all_lines)):
        if len(all_lines[j]) == 0 or all_lines[j][0] in [' ', '\t', ')']:
            ret.append(all_lines[j])
        else:
            break
    return NEWLINE.join(ret)

def get_functions(filepath):
    """
    Get all functions in a Python file.
    """
    with open(filepath, 'r') as file:
        all_lines = file.read().replace('\r', NEWLINE).split(NEWLINE)
        for i, l in enumerate(all_lines):
            for prefix in DEF_PREFIXES:
                if l.startswith(prefix):
                    code = get_until_no_space(all_lines, i)
                    function_name = get_function_name(code)
                    yield {
                        'code': code,
                        'function_name': function_name,
                        'filepath': filepath,
                    }
                    break

def extract_functions_from_repo(code_root):
    """
    Extract all .py functions from the repository.
    """
    code_files = list(code_root.glob('**/*.py'))

    num_files = len(code_files)
    print(f'Total number of .py files: {num_files}')

    if num_files == 0:
        print('Verify the repository exists and code_root is set correctly.')
        return None

    all_funcs = [
        func
        for code_file in code_files
        for func in get_functions(str(code_file))
    ]

    num_funcs = len(all_funcs)
    print(f'Total number of functions extracted: {num_funcs}')

    return all_funcs
```

These functions will traverse your repository, identify function definitions, and collect them into a structured format.

## Step 2: Load and Prepare Your Codebase

Now, point the script to your local copy of the `openai-python` repository (or any Python project) and extract the functions.

```python
# Set the root directory to your repository location
root_dir = Path.home()
code_root = root_dir / 'openai-python'

# Extract all functions from the repository
all_funcs = extract_functions_from_repo(code_root)
```

If successful, you'll see output like:

```
Total number of .py files: 51
Total number of functions extracted: 97
```

## Step 3: Generate Embeddings for Each Function

With the functions extracted, you'll now generate vector embeddings using OpenAI's `text-embedding-3-small` model. This step requires a helper utility for API calls.

First, ensure you have a `utils.embeddings_utils` module with a `get_embedding` function. Here's a simplified version you can use:

```python
# utils/embeddings_utils.py (example)
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding
```

Now, apply the embedding function to each code snippet and save the results.

```python
from utils.embeddings_utils import get_embedding

df = pd.DataFrame(all_funcs)
df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
df['filepath'] = df['filepath'].map(lambda x: Path(x).relative_to(code_root))
df.to_csv("data/code_search_openai-python.csv", index=False)
df.head()
```

The resulting DataFrame will have columns for the code, function name, filepath, and its vector embedding.

## Step 4: Implement the Semantic Search Function

To query the embedded functions, you need a search function that:
1. Embeds the query string.
2. Computes cosine similarity between the query embedding and all function embeddings.
3. Returns the top matches.

You'll need a `cosine_similarity` function. Add this to your `embeddings_utils`:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

Now, define the search function:

```python
from utils.embeddings_utils import cosine_similarity

def search_functions(df, code_query, n=3, pprint=True, n_lines=7):
    embedding = get_embedding(code_query, model='text-embedding-3-small')
    df['similarities'] = df.code_embedding.apply(lambda x: cosine_similarity(x, embedding))

    res = df.sort_values('similarities', ascending=False).head(n)

    if pprint:
        for r in res.iterrows():
            print(f"{r[1].filepath}:{r[1].function_name}  score={round(r[1].similarities, 3)}")
            print("\n".join(r[1].code.split("\n")[:n_lines]))
            print('-' * 70)

    return res
```

## Step 5: Test the Search with Example Queries

Now you can perform semantic searches using natural language descriptions.

### Example 1: Finding Fine-Tuning Validation Logic

```python
res = search_functions(df, 'fine-tuning input data validation logic', n=3)
```

**Output:**
```
openai/validators.py:format_inferrer_validator  score=0.453
def format_inferrer_validator(df):
    """
    This validator will infer the likely fine-tuning format of the data, and display it to the user if it is classification.
    It will also suggest to use ada and explain train/validation split benefits.
    """
    ft_type = infer_task_type(df)
    immediate_msg = None
----------------------------------------------------------------------
openai/validators.py:infer_task_type  score=0.37
def infer_task_type(df):
    """
    Infer the likely fine-tuning task type from the data
    """
    CLASSIFICATION_THRESHOLD = 3  # min_average instances of each class
    if sum(df.prompt.str.len()) == 0:
        return "open-ended generation"
----------------------------------------------------------------------
openai/validators.py:apply_validators  score=0.369
def apply_validators(
    df,
    fname,
    remediation,
    validators,
    auto_accept,
    write_out_file_func,
----------------------------------------------------------------------
```

### Example 2: Searching for String Manipulation Functions

```python
res = search_functions(df, 'find common suffix', n=2, n_lines=10)
```

**Output:**
```
openai/validators.py:get_common_xfix  score=0.487
def get_common_xfix(series, xfix="suffix"):
    """
    Finds the longest common suffix or prefix of all the values in a series
    """
    common_xfix = ""
    while True:
        common_xfixes = (
            series.str[-(len(common_xfix) + 1) :]
            if xfix == "suffix"
            else series.str[: len(common_xfix) + 1]
----------------------------------------------------------------------
openai/validators.py:common_completion_suffix_validator  score=0.449
def common_completion_suffix_validator(df):
    """
    This validator will suggest to add a common suffix to the completion if one doesn't already exist in case of classification or conditional generation.
    """
    error_msg = None
    immediate_msg = None
    optional_msg = None
    optional_fn = None

    ft_type = infer_task_type(df)
----------------------------------------------------------------------
```

### Example 3: Locating CLI-Related Code

```python
res = search_functions(df, 'Command line interface for fine-tuning', n=1, n_lines=20)
```

**Output:**
```
openai/cli.py:tools_register  score=0.391
def tools_register(parser):
    subparsers = parser.add_subparsers(
        title="Tools", help="Convenience client side tools"
    )

    def help(args):
        parser.print_help()

    parser.set_defaults(func=help)

    sub = subparsers.add_parser("fine_tunes.prepare_data")
    sub.add_argument(
        "-f",
        "--file",
        required=True,
        help="JSONL, JSON, CSV, TSV, TXT or XLSX file containing prompt-completion examples to be analyzed."
        "This should be the local file path.",
    )
    sub.add_argument(
        -q,
----------------------------------------------------------------------
```

## Summary

You've successfully built a semantic code search engine. By embedding function code and querying with natural language, you can quickly locate relevant code snippets without exact keyword matches. This approach is highly adaptable—you can apply it to any codebase by adjusting the parsing logic and scaling the embedding process.