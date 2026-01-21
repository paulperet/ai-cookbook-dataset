# End-to-End Image Processing with NVIDIA NIM and Phi-3-Vision

This guide demonstrates how to automate image processing by using a Large Multimodal Model (LMM) to generate and execute Python code. You will use the `microsoft/phi-3-vision-128k-instruct` model, accessible via NVIDIA's NIM endpoints, to create a script that processes an input image and saves a new version.

## Prerequisites

Before you begin, ensure you have:
- An NVIDIA API key. You can obtain one from the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/).
- A Python environment (3.8 or later) with `pip` installed.
- An image file named `demo.png` located in an `./imgs/` directory relative to your script.

## Step 1: Setup and Installation

First, install the required LangChain integration package for NVIDIA AI endpoints.

```bash
pip install langchain_nvidia_ai_endpoints -U
```

## Step 2: Import Required Modules

Import the necessary libraries for the workflow.

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import getpass
import os
import base64
```

## Step 3: Configure the API Key

Securely set your NVIDIA API key as an environment variable. The script will prompt you for it if it's not already set.

```python
if not os.getenv("NVIDIA_API_KEY"):
    os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter your NVIDIA API key: ")
```

## Step 4: Initialize the Model and Define Image Path

Create an instance of the `ChatNVIDIA` client, specifying the Phi-3-Vision model, and define the path to your source image.

```python
model = 'microsoft/phi-3-vision-128k-instruct'
chat = ChatNVIDIA(model=model)
img_path = './imgs/demo.png'
```

## Step 5: Create the Instruction Prompt

Define the text instruction that will guide the model. This prompt asks the model to generate Python code for image processing.

```python
text = "Please create Python code for image, and use plt to save the new picture under imgs/ and name it phi-3-vision.jpg."
```

## Step 6: Encode the Image

To send the image to the model, you need to encode it in base64 format and wrap it in an HTML tag, which is a common format for multimodal inputs.

```python
with open(img_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()
image = f'<img src="data:image/png;base64,{image_b64}" />'
```

## Step 7: Construct the Full Prompt

Combine the text instruction and the encoded image into a single prompt string.

```python
prompt = f"{text} {image}"
```

## Step 8: Generate the Python Code

Stream the prompt to the model. The response will be a markdown block containing the Python code. You will collect and print the response in chunks.

```python
code = ""
for chunk in chat.stream(prompt):
    print(chunk.content, end="")
    code += chunk.content
```

## Step 9: Extract the Raw Code

The model's response typically includes markdown code fences (` ```python `). You need to extract just the Python code from within these fences.

```python
begin = code.index('```python') + 9
code = code[begin:]
end = code.index('```')
code = code[:end]
```

## Step 10: Execute the Generated Code

Use Python's `subprocess` module to run the extracted code string in a separate process. This method safely executes the dynamically generated code.

```python
import subprocess
result = subprocess.run(["python", "-c", code], capture_output=True)
```

**Note:** If the generated code has dependencies (like `matplotlib`), ensure they are installed in your environment before this step.

## Step 11: Display the Results

Finally, use IPython's display utilities to show both the original and the newly processed image, confirming the operation was successful.

```python
from IPython.display import Image, display
display(Image(filename='./imgs/phi-3-vision.jpg'))
display(Image(filename='./imgs/demo.png'))
```

## Summary

You have successfully automated an image processing task using an AI model. This workflow involved:
1.  Setting up the environment and authenticating with the NVIDIA API.
2.  Preparing a multimodal prompt with an instruction and an image.
3.  Using the Phi-3-Vision model to generate executable Python code.
4.  Parsing and running the generated code to process the image.
5.  Displaying the output to verify the result.

This pattern can be adapted for various automation tasks, leveraging AI to bridge the gap between a natural language request and functional code execution.