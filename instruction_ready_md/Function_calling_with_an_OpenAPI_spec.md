# AI Cookbook: Function-Calling with OpenAPI Specifications

## Introduction

Much of the internet is powered by RESTful APIs. By enabling GPT models to call these APIs, we unlock powerful automation capabilities. This guide demonstrates how to leverage OpenAPI specifications to enable intelligent API calls through chained function calls.

The [OpenAPI Specification (OAS)](https://swagger.io/specification/) is a universal standard for describing RESTful APIs in a machine-readable format. This allows both humans and computers to understand service capabilities, making it perfect for teaching GPT how to call APIs.

This tutorial covers two main sections:

1. Converting an OpenAPI specification into function definitions for the chat completions API
2. Using the chat completions API to intelligently invoke these functions based on user instructions

## Prerequisites

Before starting, ensure you have the necessary libraries installed:

```bash
pip install jsonref openai requests
```

## Setup

First, import the required libraries and set up your OpenAI client:

```python
import os
import json
import jsonref
from openai import OpenAI
import requests
from pprint import pp

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
```

## Part 1: Converting OpenAPI Specifications to Function Definitions

### Understanding the OpenAPI Specification

We'll use a sample OpenAPI specification that describes an events API. This spec includes endpoints for:

- Listing all events
- Creating a new event
- Retrieving an event by ID
- Deleting an event by ID
- Updating an event name by ID

Each operation has an `operationId` that we'll use as the function name when parsing the spec.

### Loading the OpenAPI Specification

Let's load and examine the sample specification:

```python
# Load the OpenAPI specification
with open('./data/example_events_openapi.json', 'r') as f:
    openapi_spec = jsonref.loads(f.read())  # Important: use jsonref to resolve references

# Display the specification structure
print("OpenAPI Specification Structure:")
print(json.dumps(openapi_spec, indent=2)[:1000] + "...")  # Show first 1000 characters
```

### Creating the Conversion Function

Now we'll create a function to convert the OpenAPI specification into GPT-compatible function definitions. Each function will include:

- **name**: The operation identifier from the OpenAPI spec
- **description**: A summary of what the function does
- **parameters**: The schema defining expected input parameters

```python
def openapi_to_functions(openapi_spec):
    """
    Convert an OpenAPI specification to GPT function definitions.
    
    Args:
        openapi_spec: The OpenAPI specification dictionary
        
    Returns:
        List of function definitions compatible with GPT's tools parameter
    """
    functions = []

    for path, methods in openapi_spec["paths"].items():
        for method, spec_with_ref in methods.items():
            # 1. Resolve JSON references to avoid duplication
            spec = jsonref.replace_refs(spec_with_ref)

            # 2. Extract function name from operationId
            function_name = spec.get("operationId")

            # 3. Extract description and parameters
            desc = spec.get("description") or spec.get("summary", "")
            
            # Build parameter schema
            schema = {"type": "object", "properties": {}}

            # Handle request body if present
            req_body = (
                spec.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema")
            )
            if req_body:
                schema["properties"]["requestBody"] = req_body

            # Handle URL parameters if present
            params = spec.get("parameters", [])
            if params:
                param_properties = {
                    param["name"]: param["schema"]
                    for param in params
                    if "schema" in param
                }
                schema["properties"]["parameters"] = {
                    "type": "object",
                    "properties": param_properties,
                }

            # Add function to list
            functions.append(
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": desc,
                        "parameters": schema
                    }
                }
            )

    return functions
```

### Testing the Conversion

Let's convert our OpenAPI specification and examine the results:

```python
# Convert OpenAPI spec to function definitions
functions = openapi_to_functions(openapi_spec)

# Display each function definition
print("Generated Function Definitions:")
for i, function in enumerate(functions, 1):
    print(f"\nFunction {i}:")
    pp(function)
```

## Part 2: Intelligent Function Calling with GPT

### Setting Up the Chat Completion System

The chat completions API doesn't execute functions directly—it generates JSON that you can use to call functions in your code. Let's set up our system:

```python
SYSTEM_MESSAGE = """
You are a helpful assistant.
Respond to the following prompt by using function_call and then summarize actions.
Ask for clarification if a user request is ambiguous.
"""

# Maximum number of function calls to prevent infinite loops
MAX_CALLS = 5

def get_openai_response(functions, messages):
    """
    Get a response from OpenAI's chat completion API.
    
    Args:
        functions: List of function definitions
        messages: Conversation history
        
    Returns:
        OpenAI chat completion response
    """
    return client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        tools=functions,
        tool_choice="auto",  # Model can choose between message or function call
        temperature=0,
        messages=messages,
    )
```

### Processing User Instructions with Function Chaining

Now we'll create a function that processes user instructions by chaining multiple function calls:

```python
def process_user_instruction(functions, instruction):
    """
    Process a user instruction by chaining function calls through GPT.
    
    Args:
        functions: List of function definitions
        instruction: User's natural language instruction
    """
    num_calls = 0
    messages = [
        {"content": SYSTEM_MESSAGE, "role": "system"},
        {"content": instruction, "role": "user"},
    ]

    while num_calls < MAX_CALLS:
        # Get response from OpenAI
        response = get_openai_response(functions, messages)
        message = response.choices[0].message
        
        print(f"\nResponse from GPT:")
        print(f"Role: {message.role}")
        
        try:
            # Check if GPT wants to call a function
            if message.tool_calls:
                print(f"\n>> Function call #{num_calls + 1}")
                print("Function call details:")
                pp(message.tool_calls)
                
                # Add GPT's message to conversation history
                messages.append(message)
                
                # In a real implementation, you would:
                # 1. Extract function name and arguments
                # 2. Call the actual API endpoint
                # 3. Add the results to messages
                
                # For this example, simulate success
                messages.append(
                    {
                        "role": "tool",
                        "content": "success",
                        "tool_call_id": message.tool_calls[0].id,
                    }
                )
                
                num_calls += 1
            else:
                # GPT responded with a message instead of function call
                print("\n>> Message from GPT:")
                print(message.content)
                break
                
        except AttributeError:
            # Handle case where there are no tool calls
            print("\n>> Message from GPT:")
            print(message.content)
            break

    if num_calls >= MAX_CALLS:
        print(f"\nReached maximum chained function calls: {MAX_CALLS}")
```

### Testing the Function Calling System

Let's test our system with a complex user instruction:

```python
USER_INSTRUCTION = """
Instruction: Get all the events.
Then create a new event named AGI Party.
Then delete event with id 2456.
"""

print("Processing user instruction...")
print(f"Instruction: {USER_INSTRUCTION}")
print("-" * 50)

process_user_instruction(functions, USER_INSTRUCTION)
```

## Complete Example: End-to-End Workflow

Here's a complete example that puts everything together:

```python
def complete_workflow(openapi_spec_path, user_instruction):
    """
    Complete workflow from OpenAPI spec to function execution.
    
    Args:
        openapi_spec_path: Path to OpenAPI specification JSON file
        user_instruction: Natural language instruction for GPT
    """
    # 1. Load OpenAPI specification
    with open(openapi_spec_path, 'r') as f:
        openapi_spec = jsonref.loads(f.read())
    
    # 2. Convert to function definitions
    functions = openapi_to_functions(openapi_spec)
    print(f"Converted {len(functions)} API endpoints to function definitions")
    
    # 3. Process user instruction
    process_user_instruction(functions, user_instruction)

# Run the complete workflow
complete_workflow('./data/example_events_openapi.json', USER_INSTRUCTION)
```

## Conclusion

You've successfully learned how to:

1. **Parse OpenAPI specifications** into GPT-compatible function definitions
2. **Handle JSON references** to avoid duplication in API schemas
3. **Enable intelligent function calling** where GPT determines which API endpoints to call
4. **Chain multiple function calls** to execute complex workflows from natural language instructions

### Next Steps and Extensions

Consider extending this system with:

1. **Real API Integration**: Replace the simulated success responses with actual API calls
2. **Error Handling**: Implement robust error handling for failed API calls
3. **Conditional Logic**: Enable GPT to make decisions based on API responses
4. **Validation**: Add input validation before making API calls
5. **Rate Limiting**: Implement proper rate limiting for production use
6. **Authentication**: Handle API authentication tokens securely

This approach enables powerful automation where users can describe complex workflows in natural language, and GPT intelligently determines and executes the necessary API calls.