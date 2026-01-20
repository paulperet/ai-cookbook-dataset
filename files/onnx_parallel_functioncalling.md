# Phi-4 Mini ONNX Parallel Function Calling Tutorial

This notebook demonstrates how to use Phi-4 Mini with ONNX Runtime GenAI for parallel function calling. Function calling allows the model to intelligently invoke external tools and APIs based on user requests.

## Overview

In this tutorial, you'll learn how to:
- Set up Phi-4 Mini with ONNX Runtime GenAI
- Define function schemas for booking flights and hotels
- Use guided generation with Lark grammar for structured output
- Execute parallel function calls for complex travel booking scenarios

## Prerequisites

Before running this notebook, ensure you have:
- Downloaded the Phi-4 Mini ONNX model
- Installed `onnxruntime-genai` package
- Basic understanding of function calling concepts

## Step 1: Import Required Libraries

First, we'll import the necessary libraries for our function calling implementation.


```python
import json
```


```python
import onnxruntime_genai as og
```

## Step 2: Model Setup and Configuration

Now we'll configure the Phi-4 Mini ONNX model. Make sure to replace the path with your actual model directory.


```python
# TODO: Replace with your actual Phi-4 Mini ONNX model path
# Download from: https://huggingface.co/microsoft/Phi-4-mini-onnx
path = 'Your phi-4-mini-onnx path'  # Update this path!
```


```python
config = og.Config(path)
```


```python
model = og.Model(config)
```


```python
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()
```

## Step 3: Configure Generation Parameters

Set up the generation parameters to control the model's output behavior. These settings ensure deterministic and focused responses for function calling.


```python
# Configure generation parameters for deterministic function calling
search_options = {}
search_options['max_length'] = 4096      # Maximum tokens to generate
search_options['temperature'] = 0.00001  # Very low temperature for deterministic output
search_options['top_p'] = 1.0            # Nucleus sampling parameter
search_options['do_sample'] = False       # Disable sampling for consistent results
```

## Step 4: Define Available Functions

Here we define the functions that our AI assistant can call. In this example, we have two travel-related functions:
1. **booking_flight_tickets**: For booking flights between airports
2. **booking_hotels**: For booking hotel accommodations

The function definitions follow OpenAI's function calling schema format.


```python
tool_list = '[{"name": "booking_flight_tickets", "description": "booking flights", "parameters": {"origin_airport_code": {"description": "The name of Departure airport code", "type": "string"}, "destination_airport_code": {"description": "The name of Destination airport code", "type": "string"}, "departure_date": {"description": "The date of outbound flight", "type": "string"}, "return_date": {"description": "The date of return flight", "type": "string"}}}, {"name": "booking_hotels", "description": "booking hotel", "parameters": {"destination": {"description": "The name of the city", "type": "string"}, "check_in_date": {"description": "The date of check in", "type": "string"}, "checkout_date": {"description": "The date of check out", "type": "string"}}}]'
```

## Step 5: Grammar Generation Helper Functions

These helper functions convert our function definitions into Lark grammar format, which is used by ONNX Runtime GenAI for guided generation. This ensures the model outputs valid function calls in the correct JSON format.


```python
def get_lark_grammar(input_tools):
    tools_list = get_tools_list(input_tools)
    prompt_tool_input = create_prompt_tool_input(tools_list)
    if len(tools_list) == 1:
        # output = ("start: TEXT | fun_call\n" "TEXT: /[^{](.|\\n)*/\n" " fun_call: <|tool_call|> %json " + json.dumps(tools_list[0]))
        output = ("start: TEXT | fun_call\n" "TEXT: /[^{](.|\\n)*/\n" " fun_call: <|tool_call|> %json " + json.dumps(convert_tool_to_grammar_input(tools_list[0])))
        return prompt_tool_input, output
    else:
        return prompt_tool_input, "start: TEXT | fun_call \n TEXT: /[^{](.|\n)*/ \n fun_call: <|tool_call|> %json {\"anyOf\": [" + ','.join([json.dumps(tool) for tool in tools_list]) + "]}"

```


```python
def get_tools_list(input_tools):
    # input_tools format: '[{"name": "fn1", "description": "fn details", "parameters": {"p1": {"description": "details", "type": "string"}}},
    # {"fn2": 2},{"fn3": 3}]'
    tools_list = []
    try:
        tools_list = json.loads(input_tools)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for tools list, expected format: '[{\"name\": \"fn1\"},{\"name\": \"fn2\"}]'")
    if len(tools_list) == 0:
        raise ValueError("Tools list cannot be empty")
    return tools_list
```


```python
def create_prompt_tool_input(tools_list):
    tool_input = str(tools_list[0])
    for tool in tools_list[1:]:
        tool_input += ',' + str(tool)
    return tool_input
```


```python
def convert_tool_to_grammar_input(tool):
    param_props = {}
    required_params = []
    for param_name, param_info in tool.get("parameters", {}).items():
        param_props[param_name] = {
            "type": param_info.get("type", "string"),
            "description": param_info.get("description", "")
        }
        required_params.append(param_name)
    output_schema = {
        "description": tool.get('description', ''),
        "type": "object",
        "required": ["name", "parameters"],
        "additionalProperties": False,
        "properties": {
            "name": { "const": tool["name"] },
            "parameters": {
                "type": "object",
                "properties": param_props,
                "required": required_params,
                "additionalProperties": False
            }
        }
    }
    if len(param_props) == 0:
        output_schema["required"] = ["name"]
    return output_schema
```


```python
get_lark_grammar(tool_list)
```




    ("{'name': 'booking_flight_tickets', 'description': 'booking flights', 'parameters': {'origin_airport_code': {'description': 'The name of Departure airport code', 'type': 'string'}, 'destination_airport_code': {'description': 'The name of Destination airport code', 'type': 'string'}, 'departure_date': {'description': 'The date of outbound flight', 'type': 'string'}, 'return_date': {'description': 'The date of return flight', 'type': 'string'}}},{'name': 'booking_hotels', 'description': 'booking hotel', 'parameters': {'destination': {'description': 'The name of the city', 'type': 'string'}, 'check_in_date': {'description': 'The date of check in', 'type': 'string'}, 'checkout_date': {'description': 'The date of check out', 'type': 'string'}}}",
     'start: TEXT | fun_call \n TEXT: /[^{](.|\n)*/ \n fun_call: <|tool_call|> %json {"anyOf": [{"name": "booking_flight_tickets", "description": "booking flights", "parameters": {"origin_airport_code": {"description": "The name of Departure airport code", "type": "string"}, "destination_airport_code": {"description": "The name of Destination airport code", "type": "string"}, "departure_date": {"description": "The date of outbound flight", "type": "string"}, "return_date": {"description": "The date of return flight", "type": "string"}}},{"name": "booking_hotels", "description": "booking hotel", "parameters": {"destination": {"description": "The name of the city", "type": "string"}, "check_in_date": {"description": "The date of check in", "type": "string"}, "checkout_date": {"description": "The date of check out", "type": "string"}}}]}')



## Step 6: Test Grammar Generation

Let's test our grammar generation functions to see how they convert our tool definitions into the proper format.


```python
prompt_tool_input, guidance_input = get_lark_grammar(tool_list)
```

## Step 7: Prepare System Prompt and Generator

Now we'll create the system prompt that tells the model about available tools and set up the generator with guided generation parameters.


```python
# Define the system prompt that introduces the assistant and its capabilities
system_prompt = "You are a helpful assistant with these tools."
```


```python
# Format the system message with tools information
messages = f"""[{{"role": "system", "content": "{system_prompt}", "tools": "{prompt_tool_input}"}}]"""
```


```python
# Apply chat template to format the system prompt properly
tokenizer_input_system_prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=False)
```


```python
tokenizer_input_system_prompt
```




    "<|system|>You are a helpful assistant with these tools.<|tool|>{'name': 'booking_flight_tickets', 'description': 'booking flights', 'parameters': {'origin_airport_code': {'description': 'The name of Departure airport code', 'type': 'string'}, 'destination_airport_code': {'description': 'The name of Destination airport code', 'type': 'string'}, 'departure_date': {'description': 'The date of outbound flight', 'type': 'string'}, 'return_date': {'description': 'The date of return flight', 'type': 'string'}}},{'name': 'booking_hotels', 'description': 'booking hotel', 'parameters': {'destination': {'description': 'The name of the city', 'type': 'string'}, 'check_in_date': {'description': 'The date of check in', 'type': 'string'}, 'checkout_date': {'description': 'The date of check out', 'type': 'string'}}}<|/tool|><|end|><|endoftext|>"




```python
input_tokens = tokenizer.encode(tokenizer_input_system_prompt)
```


```python
input_tokens = input_tokens[:-1]
```


```python
system_prompt_length = len(input_tokens)
```

## Step 8: Initialize Generator with Guided Generation

Now we'll create the generator with our configured parameters and apply the Lark grammar for guided generation.


```python
# Create generator parameters and apply search options
params = og.GeneratorParams(model)
params.set_search_options(**search_options)
```


```python
# Apply Lark grammar for guided generation to ensure valid function call format
params.set_guidance("lark_grammar", guidance_input)
```


```python

```


```python
generator = og.Generator(model, params)
```


```python
generator.append_tokens(input_tokens)
```

## Step 9: Test Parallel Function Calling

Now let's test our setup with a complex travel booking scenario that requires calling multiple functions.


```python
# Complex travel booking request that requires both flight and hotel booking
text = "book flight ticket from Beijing to Paris(using airport code) in 2025-12-04 to 2025-12-10 , then book hotel from 2025-12-04 to 2025-12-10 in Paris"
```


```python
# Format user message and apply chat template
messages = f"""[{{"role": "user", "content": "{text}"}}]"""

# Apply Chat Template for user input
user_prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)
input_tokens = tokenizer.encode(user_prompt)
generator.append_tokens(input_tokens)
```


```python
user_prompt
```




    '<|user|>book flight ticket from Beijing to Paris(using airport code) in 2025-12-04 to 2025-12-10 , then book hotel from 2025-12-04 to 2025-12-10 in Paris<|end|><|assistant|>'



### View the formatted user prompt

### Generate Function Calls

The model will now generate structured function calls based on our user request. Thanks to guided generation, the output will be in valid JSON format that can be directly executed.

**Expected Output**: The model should generate function calls for:
1. `booking_flight_tickets` with Beijing (PEK) to Paris (CDG) details
2. `booking_hotels` with Paris accommodation details

Run the cell below to see the live generation:


```python
# Generate tokens one by one and stream the output
while not generator.is_done():
    generator.generate_next_token()
    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end='', flush=True)
```

    [{"name": "booking_flight_tickets", "arguments": {"origin_airport_code": "PEKK", "destination_airport_code": "CDG", "departure_date": "2025-12-04", "return_date": "2025-12-10"}}, {"name": "booking_hotels", "arguments": {"destination": "Paris", "check_in_date": "2025-12-04", "checkout_date": "2025-12-10"}}]

## Conclusion

🎉 **Congratulations!** You've successfully implemented parallel function calling with Phi-4 Mini using ONNX Runtime GenAI.

### What You've Learned:

1. **Model Setup**: How to configure Phi-4 Mini with ONNX Runtime GenAI
2. **Function Definition**: How to define function schemas for AI function calling
3. **Guided Generation**: How to use Lark grammar for structured output generation
4. **Parallel Function Calls**: How to handle complex requests requiring multiple function calls

### Key Benefits:

- ✅ **Structured Output**: Guided generation ensures valid JSON function calls
- ✅ **Parallel Processing**: Handle multiple function calls in a single request
- ✅ **High Performance**: ONNX Runtime provides optimized inference
- ✅ **Flexible Schema**: Easy to add or modify function definitions

### Resources:

- [Phi-4 Mini Documentation](https://huggingface.co/microsoft/Phi-4-mini-onnx)
- [ONNX Runtime GenAI Documentation](https://onnxruntime.ai/docs/genai/)
- [Function Calling Best Practices](https://platform.openai.com/docs/guides/function-calling)