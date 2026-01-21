# Efficient Multimodality and Function Calling with Mistral AI

This guide demonstrates how to build an AI workflow that combines two powerful capabilities: extracting structured data from images and using function calling to query that data. You will learn to use Mistral's multimodal model to convert an image of a table into a structured JSON format, then use function calling to answer specific questions about the extracted data.

## Prerequisites

Before you begin, ensure you have the following:

1.  A Mistral AI API key. Store it in an environment variable named `MISTRAL_KEY`.
2.  An image file named `table.png` in your working directory. This image should contain a table of data (e.g., transaction records).

## Setup and Installation

First, install the required Python packages and import the necessary modules.

```bash
pip install requests python-dotenv pandas pillow --upgrade mistralai
```

```python
import os
import json
import base64
import pandas as pd
import requests
from io import BytesIO
from pprint import pprint
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
from PIL import Image as PILImage
from mistralai import Mistral

# Load environment variables (which contains your MISTRAL_KEY)
load_dotenv()
api_key = os.getenv("MISTRAL_KEY")
```

## Part 1: Extracting Structured Data from an Image

In this section, you will use Mistral's multimodal model to read a table from an image and output it as a JSON object.

### Step 1: Encode the Image

To send an image to the API, you must first encode it as a base64 string. The following helper function handles this process.

```python
def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Encode your image
image_path = "./table.png"
base64_image = encode_image(image_path)
```

### Step 2: Call the Multimodal Model

Now, initialize the Mistral client and construct a request asking the model to extract text from the image and format it as JSON.

```python
# Initialize the client and specify the model
client = Mistral(api_key=api_key)
model = "pixtral-12b-2409"

# Define the chat messages
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Please extract the text from this image and output as a json object."
            },
            {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{base64_image}"
            }
        ]
    }
]

# Get the chat response
chat_response = client.chat.complete(
    model=model,
    messages=messages,
    temperature=0.0,
    response_format={"type": "json_object"}
)

# Print the extracted JSON
print(chat_response.choices[0].message.content)
```

The model will return a JSON string. For our example, let's assume the output contains a list of transactions under a key named `'transactions'`.

### Step 3: Convert JSON to a DataFrame

To work with the data more easily, let's convert the JSON response into a pandas DataFrame.

```python
from typing import Union

def json_to_dataframe(json_data: Union[str, dict], key: str = None) -> pd.DataFrame:
    # If json_data is a string, parse it into a dictionary
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    
    # If a key is provided, extract the list of records from the JSON object
    if key is not None:
        data = json_data[key]
    else:
        data = json_data
    
    # Convert the list of records to a pandas DataFrame
    df = pd.DataFrame(data)
    return df

# Convert the model's response to a DataFrame
df = json_to_dataframe(chat_response.choices[0].message.content, key='transactions')
print(df.head())
```

You now have a structured DataFrame (`df`) containing the data from your image. This emulates an external database that the LLM cannot access directly.

## Part 2: Querying Data with Function Calling

Function calling allows Mistral models to connect to external tools (like your DataFrame) to answer specific questions. The process involves four steps:
1.  **User:** Specify the available tools and ask a question.
2.  **Model:** Decide which tool to use and generate the necessary arguments.
3.  **User:** Execute the chosen function with the provided arguments.
4.  **Model:** Generate a final answer using the function's result.

### Step 1: Define Your Tools (Functions)

First, create the functions that will act as tools for querying your data. In this example, we'll create tools to look up a transaction's status and its payment date.

```python
import json

def retrieve_payment_status(df: pd.DataFrame, transaction_id: str) -> str:
    """Tool to get the payment status for a given transaction ID."""
    if transaction_id in df.transaction_id.values:
        return json.dumps({'status': df[df.transaction_id == transaction_id].payment_status.item()})
    return json.dumps({'error': 'transaction id not found.'})

def retrieve_payment_date(df: pd.DataFrame, transaction_id: str) -> str:
    """Tool to get the payment date for a given transaction ID."""
    if transaction_id in df.transaction_id.values:
        return json.dumps({'date': df[df.transaction_id == transaction_id].payment_date.item()})
    return json.dumps({'error': 'transaction id not found.'})
```

### Step 2: Describe the Tools for the Model

The model needs a structured description of each tool to understand when and how to use them. This is done using a JSON schema.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_status",
            "description": "Get payment status of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_payment_date",
            "description": "Get payment date of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    }
                },
                "required": ["transaction_id"],
            },
        },
    }
]
```

Next, create a mapping from function names to the actual callable functions. We use `functools.partial` to pre-bind the DataFrame (`df`) to each function.

```python
import functools

names_to_functions = {
  'retrieve_payment_status': functools.partial(retrieve_payment_status, df=df),
  'retrieve_payment_date': functools.partial(retrieve_payment_date, df=df)
}
```

### Step 3: Ask a Question and Let the Model Choose a Tool

Now, pose a question that requires querying the data. The model will analyze the question and the available tool descriptions to decide which tool to call.

```python
# Start a new conversation with a user query
messages = [{"role": "user", "content": "What's the status of my transaction T1001?"}]

# Send the query and tool descriptions to the model
response = client.chat.complete(
    model = model,
    messages = messages,
    tools = tools,
    tool_choice = "any", # Let the model decide which tool to use
)

# Append the model's response (which includes the tool call) to the message history
messages.append(response.choices[0].message)
```

### Step 4: Execute the Chosen Function

The model's response contains a `tool_calls` object specifying which function to run and with what arguments. It's your responsibility to execute this function.

```python
# Extract the tool call details from the model's response
tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_params = json.loads(tool_call.function.arguments)

print(f"Function to call: {function_name}")
print(f"Function arguments: {function_params}")

# Execute the function using our mapping
function_result = names_to_functions[function_name](**function_params)
print(f"Function result: {function_result}")
```

### Step 5: Provide the Result Back to the Model

Add the function's result to the message history, specifying it as output from a `tool`. This gives the model the context it needs to formulate its final answer.

```python
messages.append({
    "role": "tool",
    "name": function_name,
    "content": function_result,
    "tool_call_id": tool_call.id
})
```

### Step 6: Get the Final Answer

Finally, send the updated conversation history (which now includes the tool's result) back to the model. It will synthesize the information and provide a natural language answer.

```python
final_response = client.chat.complete(
    model = model,
    messages = messages
)

print("Final Answer:", final_response.choices[0].message.content)
```

## Summary

You have successfully built a pipeline that:
1.  **Extracts Data:** Uses a multimodal LLM to convert an image of a table into structured JSON data.
2.  **Queries Data:** Implements function calling to allow an LLM to use custom Python functions as tools to query the extracted data.
This pattern is extremely powerful for building AI applications that can interact with private data, external APIs, or custom business logic.