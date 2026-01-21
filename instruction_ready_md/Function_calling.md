# Mistral AI Function Calling Tutorial: Querying a Payment Database

## Overview
This tutorial demonstrates how to use Mistral AI's function calling capability to connect LLMs with external tools. You'll learn to create a system where a Mistral model can query a payment transaction database through custom functions, enabling it to answer specific business questions it couldn't address directly.

## Prerequisites

First, install the required packages:

```bash
pip install pandas mistralai
```

Then import the necessary libraries:

```python
import pandas as pd
import json
import functools
import os
from mistralai import Mistral
```

## Step 1: Set Up Your Data

Create a sample payment transactions DataFrame to simulate an external database:

```python
# Sample payment transaction data
data = {
    'transaction_id': ['T1001', 'T1002', 'T1003', 'T1004', 'T1005'],
    'customer_id': ['C001', 'C002', 'C003', 'C002', 'C001'],
    'payment_amount': [125.50, 89.99, 120.00, 54.30, 210.20],
    'payment_date': ['2021-10-05', '2021-10-06', '2021-10-07', '2021-10-05', '2021-10-08'],
    'payment_status': ['Paid', 'Unpaid', 'Paid', 'Paid', 'Pending']
}

# Create DataFrame
df = pd.DataFrame(data)
```

## Step 2: Define Your Tools (Functions)

Create the functions that will serve as your tools. These functions query the DataFrame and return results in JSON format.

```python
def retrieve_payment_status(df: pd.DataFrame, transaction_id: str) -> str:
    """Retrieve payment status for a given transaction ID."""
    if transaction_id in df.transaction_id.values:
        return json.dumps({'status': df[df.transaction_id == transaction_id].payment_status.item()})
    return json.dumps({'error': 'transaction id not found.'})

def retrieve_payment_date(df: pd.DataFrame, transaction_id: str) -> str:
    """Retrieve payment date for a given transaction ID."""
    if transaction_id in df.transaction_id.values:
        return json.dumps({'date': df[df.transaction_id == transaction_id].payment_date.item()})
    return json.dumps({'error': 'transaction id not found.'})
```

## Step 3: Create Tool Specifications

Define JSON schemas that describe your functions to the Mistral model. This metadata helps the model understand when and how to use each tool.

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

## Step 4: Map Function Names to Executable Functions

Create a dictionary that maps function names to their executable counterparts. This allows you to call the appropriate function based on the model's selection.

```python
names_to_functions = {
    'retrieve_payment_status': functools.partial(retrieve_payment_status, df=df),
    'retrieve_payment_date': functools.partial(retrieve_payment_date, df=df)
}
```

## Step 5: Initialize the Mistral Client

Set up your Mistral API client. Make sure you have your API key stored in your environment variables.

```python
api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"
client = Mistral(api_key=api_key)
```

## Step 6: Define the User Query

Create a message containing the user's question. This query requires accessing external data, which the LLM cannot answer directly without tools.

```python
messages = [{"role": "user", "content": "What's the status of my transaction T1001?"}]
```

## Step 7: Let the Model Choose and Parameterize a Tool

Send the user query along with the tool specifications to the Mistral model. The model will analyze which tool is appropriate and generate the necessary parameters.

```python
response = client.chat.complete(
    model=model,
    messages=messages,
    tools=tools,
    tool_choice="any",
)

# Add the model's response to the conversation history
messages.append(response.choices[0].message)
```

The model responds with a tool call request, indicating it wants to use `retrieve_payment_status` with `transaction_id` set to "T1001".

## Step 8: Execute the Selected Function

Extract the function name and parameters from the model's response, then execute the corresponding function.

```python
# Extract tool call information
tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_params = json.loads(tool_call.function.arguments)

print(f"Function to execute: {function_name}")
print(f"Parameters: {function_params}")

# Execute the function
function_result = names_to_functions[function_name](**function_params)
print(f"Function result: {function_result}")
```

You should see output similar to:
```
Function to execute: retrieve_payment_status
Parameters: {'transaction_id': 'T1001'}
Function result: {"status": "Paid"}
```

## Step 9: Provide the Tool Result Back to the Model

Add the function's result to the conversation history as a tool response.

```python
messages.append({
    "role": "tool",
    "name": function_name,
    "content": function_result,
    "tool_call_id": tool_call.id
})
```

## Step 10: Get the Final Answer from the Model

Now that the model has access to the tool's result, it can generate a complete, natural language answer.

```python
response = client.chat.complete(
    model=model,
    messages=messages
)

final_answer = response.choices[0].message.content
print(f"Final answer: {final_answer}")
```

The model will respond with something like:
```
Final answer: The payment for transaction T1001 has been successfully paid.
```

## Summary

You've successfully implemented a function calling workflow with Mistral AI:

1. **Defined tools** as Python functions that query your data
2. **Created tool specifications** in JSON format for the model to understand
3. **Let the model choose** the appropriate tool and generate parameters
4. **Executed the function** locally with the model's parameters
5. **Provided results back** to the model for final answer generation

This pattern enables Mistral models to interact with any external system—databases, APIs, or custom business logic—through well-defined interfaces, significantly expanding their practical applications.