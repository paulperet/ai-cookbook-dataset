# Guide: Function Calling with Mistral 7B via REST API and Ollama

## Overview
This guide demonstrates how to implement function calling with the Mistral 7B model using Ollama. Function calling enables LLMs to interact with external tools and APIs, allowing them to access real-time data or execute specific business logic. We'll walk through a practical example using a Pet Store API to retrieve pet and user information.

## Prerequisites

### Install Required Packages
First, install the necessary Python libraries:

```bash
pip install --upgrade ollama mistral-common pandas prance openapi-spec-validator requests
```

### Import Libraries
Import the required modules:

```python
import json
import os
import functools
from typing import List

import requests
import prance
import ollama
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_common.protocol.instruct.messages import UserMessage
```

## Step 1: Define Your API Functions

We'll create two functions to interact with the Pet Store API—one to retrieve pet information by ID and another to get user details by username.

```python
def getPetById(petId: int) -> str:
    """
    Retrieve pet information by ID from the Pet Store API.
    
    Args:
        petId (int): The ID of the pet to look up.
    
    Returns:
        str: JSON string containing pet data or an error message.
    """
    try:
        url = f'https://petstore3.swagger.io/api/v3/pet/{petId}'
        response = requests.get(url)
        response.raise_for_status()
        
        if response.ok:
            json_response = response.json()
            if petId == json_response['id']:
                return json_response
        return json.dumps({'error': 'Pet id not found.'})
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return json.dumps({'error': 'Pet id not found.'})
        else:
            return json.dumps({'error': 'Error with API.'})

def getUserByName(username: str) -> str:
    """
    Retrieve user information by username from the Pet Store API.
    
    Args:
        username (str): The username to look up.
    
    Returns:
        str: JSON string containing user data or an error message.
    """
    try:
        url = f'https://petstore3.swagger.io/api/v3/user/{username}'
        response = requests.get(url)
        response.raise_for_status()
        
        if response.ok:
            json_response = response.json()
            if username == json_response['username']:
                return json_response
        return json.dumps({'error': 'Username id not found.'})
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return json.dumps({'error': 'Username not found.'})
        else:
            return json.dumps({'error': 'Error with API.'})
```

### Create a Function Mapping
Map function names to their implementations for easy access:

```python
names_to_functions = {
    'getPetById': functools.partial(getPetById, petId=''),
    'getUserByName': functools.partial(getUserByName, username='')
}
```

## Step 2: Generate Tool Definitions from OpenAPI Spec

To enable the Mistral model to understand our functions, we need to define them in a format it can process. We'll parse an OpenAPI specification to automatically generate tool definitions.

```python
def generate_tools(objs: List[List[str]], function_end_point: str) -> List[Tool]:
    """
    Generate Tool objects from an OpenAPI specification.
    
    Args:
        objs: List of [resource, field] pairs (e.g., [['pet', 'petId']]).
        function_end_point: Path to the OpenAPI spec file (JSON or YAML).
    
    Returns:
        List[Tool]: A list of Tool objects ready for the Mistral model.
    """
    params = ['operationId', 'description', 'parameters']
    parser = prance.ResolvingParser(function_end_point, backend='openapi-spec-validator')
    spec = parser.specification
    
    user_tools = []
    for obj in objs:
        resource, field = obj
        path = f'/{resource}/{{{field}}}'
        
        # Extract function metadata from the OpenAPI spec
        function_name = spec['paths'][path]['get'][params[0]]
        function_description = spec['paths'][path]['get'][params[1]]
        function_parameters = spec['paths'][path]['get'][params[2]]
        
        # Build the parameter schema
        func_parameters = {
            "type": "object",
            "properties": {
                function_parameters[0]['name']: {
                    "type": function_parameters[0]['schema']['type'],
                    "description": function_parameters[0]['description']
                }
            },
            "required": [function_parameters[0]['name']]
        }
        
        # Create the Function and Tool objects
        user_function = Function(
            name=function_name,
            description=function_description,
            parameters=func_parameters
        )
        user_tool = Tool(function=user_function)
        user_tools.append(user_tool)
    
    return user_tools
```

## Step 3: Prepare User Queries

Create a helper function to convert user queries into the proper message format:

```python
def get_user_messages(queries: List[str]) -> List[UserMessage]:
    """
    Convert a list of user queries into UserMessage objects.
    
    Args:
        queries: List of user query strings.
    
    Returns:
        List[UserMessage]: Formatted messages for the chat completion request.
    """
    user_messages = []
    for query in queries:
        user_message = UserMessage(content=query)
        user_messages.append(user_message)
    return user_messages
```

## Step 4: Execute the Function Calling Pipeline

Now, let's bring everything together. We'll define a function that:
1. Takes user queries
2. Generates appropriate tools
3. Sends a request to the Ollama endpoint
4. Processes the model's response

```python
def execute_generator():
    """
    Main execution function for the function calling workflow.
    """
    # Define user queries and corresponding API endpoints
    queries = [
        "What's the status of my Pet 1?",
        "Find information of user user1?",
        "What's the status of my Store Order 3?"
    ]
    return_objs = [
        ['pet', 'petId'],
        ['user', 'username'],
        ['store/order', 'orderId']
    ]
    
    # Path to your OpenAPI specification file
    function_end_point = 'path/to/your/openapi.json'  # Update this path
    
    # Prepare messages and tools
    user_messages = get_user_messages(queries)
    user_tools = generate_tools(return_objs, function_end_point)
    
    # Tokenize the request
    tokenizer = MistralTokenizer.v3()
    completion_request = ChatCompletionRequest(
        tools=user_tools,
        messages=user_messages
    )
    tokenized = tokenizer.encode_chat_completion(completion_request)
    _, text = tokenized.tokens, tokenized.text
    
    # Configure Ollama endpoint
    ollama_endpoint_env = os.environ.get('OLLAMA_ENDPOINT', 'http://localhost:11434')
    model = "mistral:7b"
    ollama_endpoint = f"{ollama_endpoint_env}/api/generate"
    
    # Send request to Ollama
    response = requests.post(
        ollama_endpoint,
        json={
            'model': model,
            'prompt': text,
            'stream': False,
            'raw': True
        },
        stream=False
    )
    response.raise_for_status()
    result = response.json()
    
    # Process and display results
    process_results(result, user_messages)
```

## Step 5: Process the Model's Response

The model will return function calls that need to be executed. This function extracts the function names and arguments, then executes the corresponding functions:

```python
def process_results(result: dict, messages: List[UserMessage]):
    """
    Parse the model's response and execute the requested functions.
    
    Args:
        result: The JSON response from Ollama.
        messages: The original user messages for context.
    """
    result_format = result['response'].split("\n\n")
    result_tool_calls = result_format[0].replace("[TOOL_CALLS] ", "")
    
    tool_calls = json.loads(result_tool_calls)
    index = 0
    
    try:
        for tool_call in tool_calls:
            function_name = tool_call["name"]
            function_params = tool_call["arguments"]
            
            # Display the original query
            print(f"Query: {messages[index].content}")
            
            # Execute the function
            function_result = names_to_functions[function_name](**function_params)
            print(f"Result: {function_result}\n")
            
            index += 1
    except KeyError:
        print(f"{function_name} is not defined in names_to_functions mapping")
```

## Step 6: Run the Example

Execute the main function to see function calling in action:

```python
if __name__ == "__main__":
    execute_generator()
```

### Expected Output
When you run the code, you should see output similar to:

```
Query: What's the status of my Pet 1?
Result: {'id': 1, 'category': {'id': -2, 'name': 'Кошка'}, 'name': 'Фей-Фей', 'photoUrls': ['string'], 'tags': [{'id': 3, 'name': 'Русская Голубая'}], 'status': 'pending'}

Query: Find information of user user1?
Result: {'id': 1, 'username': 'user1', 'firstName': 'first name 1', 'lastName': 'last name 1', 'email': 'email1@test.com', 'password': 'XXXXXXXXXXX', 'phone': '123-456-7890', 'userStatus': 1}

Query: What's the status of my Store Order 3?
getOrderById is not defined
```

## Key Takeaways

1. **Function Mapping**: The model selects appropriate functions based on the user's query and available tool definitions.
2. **Dynamic Tool Generation**: By parsing OpenAPI specifications, you can automatically create tool definitions for any compliant API.
3. **Error Handling**: The example includes basic error handling for API failures and undefined functions.
4. **Extensibility**: You can easily add more functions by updating the `names_to_functions` dictionary and OpenAPI specification.

## Next Steps

- Add more sophisticated error handling and retry logic.
- Implement streaming responses for better user experience.
- Extend the tool set to include write operations (POST, PUT, DELETE).
- Add authentication mechanisms for secured APIs.

This guide provides a foundation for building AI-powered applications that can interact with external systems through function calling, enabling more dynamic and data-aware conversational experiences.