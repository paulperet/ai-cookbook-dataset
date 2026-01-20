# Function Calling Rest API with Mistral7Bv3 using Ollama

Function calling allows Mistral models to connect to external tools. By integrating Mistral models with external tools such as user defined functions or APIs, users can easily build applications catering to specific use cases and practical problems. In this guide, for instance, we wrote two functions for tracking a Pet Store's Pets and User info. We can use these two tools to provide answers for pet-related queries.

At a glance, there are four steps with function calling:

- User: specify tools and query
- Model: Generate function arguments if applicable
- User: Execute function to obtain tool results
- Model: Generate final answer

In this guide, we will walk through a simple example to demonstrate how function calling works with Mistral models in these four steps.

Before we get started, let’s assume we have an OpenAPI spec end-points consisting of Pet store information. When users ask questions about this API, they can use certain tools to answer questions about this data. This is just an example to emulate an external database via API that the LLM cannot directly access.

```python
!pip install --upgrade ollama mistral-common pandas
!pip install --upgrade  prance openapi-spec-validator
```

```python
import prance
from typing import List
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import Function, Tool
import ollama
from mistral_common.protocol.instruct.messages import UserMessage
import json
import requests
import functools
import os
```

Setup functions to make REST API call. We take example of pet store from [Swagger Editor](https://editor.swagger.io/)  
We download the openapi.json specification.

Example curl query to get information of a Pet by PetID

`
curl -X 'GET' \
  'https://petstore3.swagger.io/api/v3/pet/1' \
  -H 'accept: application/json'
`

Example curl query to get information of a User by username
`
curl -X 'GET' \
  'https://petstore3.swagger.io/api/v3/user/user1' \
  -H 'accept: application/json'
`  

# Function Calling for REST API

## Step 1. User: specify tools and query

### Tools

Users can define all the necessary tools for their use cases.

- In many cases, we might have multiple tools at our disposal. For example, let’s consider we have two functions as our two tools: `retrieve_pet_info` and `retreive_user_info` to retrieve pet and user info given `petID` and `username`.
- Then we organize the two functions into a dictionary where keys represent the function name, and values are the function with the df defined. This allows us to call each function based on its function name.

```python
def getPetById(petId: int) -> str:
    try:
        method = 'GET'
        headers=None
        data=None
        url =  'https://petstore3.swagger.io/api/v3/pet/' + str(petId)
        response = requests.request(method, url, headers=headers, data=data)
        # Raise an exception if the response was unsuccessful
        response.raise_for_status()
        #response = make_api_call('GET', url + str(petId))
        if response.ok :
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
    try:
        url = 'https://petstore3.swagger.io/api/v3/user/' + username
        response = requests.get(url)
        # Raise an exception if the response was unsuccessful
        response.raise_for_status()
        if response.ok :
            json_response = response.json()
            if username == json_response['username']:
                return json_response
        return json.dumps({'error': 'Username id not found.'})
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return json.dumps({'error': 'Username not found.'})
        else:
            return json.dumps({'error': 'Error with API.'})

names_to_functions = {
  'getPetById': functools.partial(getPetById, petId=''),
  'getUserByName': functools.partial(getUserByName, username='')  
}
```

- In order for Mistral models to understand the functions, we need to outline the function specifications with a JSON schema. Specifically, we need to describe the type, function name, function description, function parameters, and the required parameter for the function. Since we have two functions here, let’s list two function specifications in a list.
- Tool Generator -
parse open api spec for dynamic tool definition creation. Download openai.json from https://editor.swagger.io/

```python
def generate_tools(objs, function_end_point)-> List[Tool]:
    params = ['operationId', 'description',  'parameters']
    parser = prance.ResolvingParser(function_end_point, backend='openapi-spec-validator')
    spec = parser.specification
    
    user_tools = []
    for obj in objs:
        resource, field = obj
        path = '/' + resource + '/{' + field + '}'
        function_name=spec['paths'][path]['get'][params[0]]
        function_description=spec['paths'][path]['get'][params[1]]
        function_parameters=spec['paths'][path]['get'][params[2]]
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
        user_function= Function(name = function_name, description = function_description, parameters = func_parameters, )
        user_tool = Tool(function = user_function)
        user_tools.append(user_tool)
    return user_tools
```

### User query

Suppose a user asks the following question: “What’s the status of my Pet 1?” A standalone LLM would not be able to answer this question, as it needs to query the business logic backend to access the necessary data. But what if we have an exact tool we can use to answer this question? We could potentially provide an answer!

```python
def get_user_messages(queries: List[str]) -> List[UserMessage]:
    user_messages=[]
    for query in queries:
        user_message = UserMessage(content=query)
        user_messages.append(user_message)
    return user_messages
```

For external ollama endpoint, set the environment variable "OLLAMA_ENDPOINT"

export OLLAMA_ENDPOINT="YOUR-Ollama-IP:Port"

```python
def execute_generator():
    queries = ["What's the status of my Pet 1?", "Find information of user user1?" ,  "What's the status of my Store Order 3?"]
    return_objs = [['pet','petId'], ['user', 'username'], ['store/order','orderId']]
    function_end_point
    user_messages=get_user_messages(queries)
    user_tools = generate_tools(return_objs, function_end_point)

    #create tokens for message and tools prompt
    tokenizer = MistralTokenizer.v3()
    completion_request = ChatCompletionRequest(tools=user_tools, messages=user_messages,)
    tokenized = tokenizer.encode_chat_completion(completion_request)
    _, text = tokenized.tokens, tokenized.text

    ollama_endpoint_env = os.environ.get('OLLAMA_ENDPOINT')
    model = "mistral:7b"
    prompt = text 

    if ollama_endpoint_env is None:
        ollama_endpoint_env = 'http://localhost:11434'
    ollama_endpoint = ollama_endpoint_env +  "/api/generate"  # replace with localhost

    response = requests.post(ollama_endpoint,
                      json={
                          'model': model,
                          'prompt': prompt,
                          'stream':False,
                          'raw': True
                      }, stream=False
                      )
    
    response.raise_for_status()
    result = response.json()

    process_results(result, user_messages)
```

## Step 3. User: Execute function to obtain tool results

How do we execute the function? Currently, it is the user’s responsibility to execute these functions and the function execution lies on the user side. In the future, we may introduce some helpful functions that can be executed server-side.

Let’s extract some useful function information from model response including function_name and function_params. It’s clear here that our Mistral model has chosen to use the function `getPetId` with the parameter `petId` set to 1.

```python
def process_results(result, messages):

    result_format = result['response'].split("\n\n")
    result_tool_calls = result_format[0].replace("[TOOL_CALLS] ","")

    tool_calls = json.loads(result_tool_calls)
    index = 0 
    try:
        for tool_call in tool_calls:
            function_name = tool_call["name"]
            function_params = (tool_call["arguments"]) 
            print(messages[index].content)
            function_result = names_to_functions[function_name](**function_params)
            print(function_result)
            index = index + 1
    except:
        print(function_name + " is not defined")
```

```python
execute_generator()
```

What's the status of my Pet 1?
{'id': 1, 'category': {'id': -2, 'name': 'Кошка'}, 'name': 'Фей-Фей', 'photoUrls': ['string'], 'tags': [{'id': 3, 'name': 'Русская Голубая'}], 'status': 'pending'}
Find information of user user1?
{'id': 1, 'username': 'user1', 'firstName': 'first name 1', 'lastName': 'last name 1', 'email': 'email1@test.com', 'password': 'XXXXXXXXXXX', 'phone': '123-456-7890', 'userStatus': 1}
What's the status of my Store Order 3?
getOrderById is not defined

```python

```