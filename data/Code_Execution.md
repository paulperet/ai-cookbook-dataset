# Gemini API: Code Execution Guide

The Gemini API's **code execution** feature allows the model to generate and run Python code based on your plain-text instructions, including the ability to output graphs. It can learn iteratively from execution results to refine its output.

This guide walks you through:
*   Setting up and using the code execution feature.
*   Making single API calls with code execution.
*   Using File I/O with local files or files uploaded via the Gemini File API.
*   Applying code execution in chat interactions.
*   Handling multimodal scenarios.

## Prerequisites & Setup

### Step 1: Install the SDK
First, install the latest `google-genai` SDK from PyPI.

```bash
pip install -q -U "google-genai>=1.0.0"
```

### Step 2: Configure Your API Key
You need a Gemini API key. Store it in an environment variable named `GOOGLE_API_KEY`. If you don't have a key, you can [create one here](https://aistudio.google.com/app/apikey).

```python
import os
from google import genai

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
```

### Step 3: Choose a Model
Select a Gemini model that supports code execution. The `gemini-2.5-flash-preview` or `gemini-2.5-pro` models are good choices for this feature.

```python
MODEL_ID = "gemini-2.5-flash-preview"
```

### Step 4: Create a Display Helper
When using code execution, the model's response contains multiple parts (text, code, execution results). This helper function formats and displays them clearly.

```python
from IPython.display import Image, Markdown, Code, HTML

def display_code_execution_result(response):
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            display(Markdown(part.text))
        if part.executable_code is not None:
            code_html = f'<pre style="background-color: #e8f5e9;">{part.executable_code.code}</pre>'
            display(HTML(code_html))
        if part.code_execution_result is not None:
            display(Markdown(part.code_execution_result.output))
        if part.inline_data is not None:
            display(Image(data=part.inline_data.data, width=800, format="png"))
        display(Markdown("---"))
```

## Using Code Execution in a Single API Call

To enable code execution, you must pass `code_execution` as a tool in the request configuration. This tells the model it can generate and run Python code.

### Example: Calculate the Sum of Prime Numbers
Let's ask the model to calculate the sum of the first 50 prime numbers.

```python
from google.genai import types

prompt = """
    What is the sum of the first 50 prime numbers?
    Generate and run code for the calculation, and make sure you get all 50.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())]
    )
)

display_code_execution_result(response)
```

**How it works:**
1.  The model receives your prompt and the instruction to use the `code_execution` tool.
2.  It generates a plan and writes the necessary Python code.
3.  The code is executed in a secure sandbox.
4.  The execution result (the list of primes and their sum) is returned and displayed.

**Expected Output:**
The model will output its reasoning, the generated code, and the result showing the first 50 primes and their sum (5117).

## Code Execution with File I/O

You can use code execution to analyze data from files. The model can read uploaded files, perform analysis, and generate visualizations.

### Step 1: Prepare a Dataset
We'll use a sample dataset from the `scikit-learn` library: the California Housing dataset.

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load the dataset and save it as a CSV file
california_housing = fetch_california_housing(as_frame=True)
california_housing.frame.to_csv('houses.csv', index=False)

# Read a subset of the data for demonstration
houses_data = pd.read_csv('houses.csv', nrows=5000)
houses_data.to_csv('houses.csv', index=False)
print(houses_data.head())
```

### Step 2: Upload the File to Gemini
Use the Gemini File API to upload your CSV file, making it accessible to the model.

```python
# Upload the CSV file
houses_file = client.files.upload(
    file='houses.csv',
    config=types.FileDict(display_name='Blocks Data')
)

print(f"Uploaded file '{houses_file.display_name}' as: {houses_file.uri}")
```

### Step 3: Analyze the Data with Code Execution
Now, ask the model to analyze the uploaded file. You can include the file object directly in the request contents.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        "This dataset provides information on various blocks in California.",
        "Generate a scatterplot comparing the houses age with the median house value for the top-20 most expensive blocks.",
        "Use each block as a different color, and include a legend of what each color represents.",
        "Plot the age as the x-axis, and the median house value as the y-axis.",
        "In addition, point out on the graph which points could be anomalies? Circle the anomaly in red on the graph.",
        "Then save the plot as an image file and display the image.",
        houses_file  # The uploaded file reference
    ],
    config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())]
    )
)

display_code_execution_result(response)
```

**What happens:**
1.  The model receives the file and your complex, multi-part instruction.
2.  It generates code to load the CSV, filter the top 20 most expensive blocks, create a scatter plot with a unique color for each block, and attempt to identify anomalies.
3.  The code is executed, and the resulting plot is saved and displayed inline.

This demonstrates the power of combining natural language instructions with programmatic execution to perform sophisticated data analysis tasks directly through the API.

## Next Steps
You can extend this pattern to:
*   **Chat Interactions:** Maintain a conversation where the model uses code execution across multiple turns, learning from previous results.
*   **Multimodal Inputs:** Combine images, PDFs, or other file types with instructions for the model to analyze using generated code.
*   **Iterative Refinement:** Ask the model to modify its code based on initial results to improve accuracy or visualization.

Remember, the `code_execution` tool is a powerful feature for tasks requiring computation, data analysis, or visualization. Always ensure you understand the code the model generates before using it in production environments.