# Guide: Parallel Function Calling with Phi-4 Mini ONNX

This guide demonstrates how to implement parallel function calling using the Phi-4 Mini model with ONNX Runtime GenAI. You will learn to set up the model, define function schemas, and use guided generation to produce structured JSON outputs for executing multiple tool calls in a single request.

## Prerequisites

Before you begin, ensure you have:

1.  Downloaded the Phi-4 Mini ONNX model from [Hugging Face](https://huggingface.co/microsoft/Phi-4-mini-onnx).
2.  Installed the `onnxruntime-genai` Python package.

## Step 1: Import Required Libraries

Start by importing the necessary library for handling JSON data.

```python
import json
import onnxruntime_genai as og
```

## Step 2: Configure the Model and Tokenizer

Initialize the model, tokenizer, and a token stream. Replace the placeholder path with the actual location of your downloaded Phi-4 Mini ONNX model.

```python
# Update this path to your local model directory
model_path = 'Your phi-4-mini-onnx path'

# Load the model configuration and create the model object
config = og.Config(model_path)
model = og.Model(config)

# Create a tokenizer and a streaming decoder
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()
```

## Step 3: Set Generation Parameters

Configure the text generation parameters to ensure deterministic and focused outputs, which is crucial for reliable function calling.

```python
search_options = {
    'max_length': 4096,      # Maximum tokens to generate
    'temperature': 0.00001,  # Very low for deterministic output
    'top_p': 1.0,            # Nucleus sampling parameter
    'do_sample': False       # Disable sampling for consistency
}
```

## Step 4: Define Your Tool Schemas

Define the functions your AI assistant can call. This example uses a travel booking scenario with two tools. The schema follows the OpenAI function-calling format.

```python
tool_list = '[{"name": "booking_flight_tickets", "description": "booking flights", "parameters": {"origin_airport_code": {"description": "The name of Departure airport code", "type": "string"}, "destination_airport_code": {"description": "The name of Destination airport code", "type": "string"}, "departure_date": {"description": "The date of outbound flight", "type": "string"}, "return_date": {"description": "The date of return flight", "type": "string"}}}, {"name": "booking_hotels", "description": "booking hotel", "parameters": {"destination": {"description": "The name of the city", "type": "string"}, "check_in_date": {"description": "The date of check in", "type": "string"}, "checkout_date": {"description": "The date of check out", "type": "string"}}}]'
```

## Step 5: Create Helper Functions for Grammar Generation

To guide the model's output into valid JSON, you need to convert the tool schemas into a Lark grammar format. The following helper functions handle this conversion.

```python
def get_tools_list(input_tools):
    """
    Parses a JSON string containing a list of tool definitions.
    """
    try:
        tools_list = json.loads(input_tools)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for tools list.")
    if len(tools_list) == 0:
        raise ValueError("Tools list cannot be empty")
    return tools_list

def create_prompt_tool_input(tools_list):
    """
    Creates a formatted string of tools for the system prompt.
    """
    tool_input = str(tools_list[0])
    for tool in tools_list[1:]:
        tool_input += ',' + str(tool)
    return tool_input

def convert_tool_to_grammar_input(tool):
    """
    Converts a single tool definition into a JSON schema suitable for grammar.
    """
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

def get_lark_grammar(input_tools):
    """
    Main function to generate the prompt input and Lark grammar string.
    """
    tools_list = get_tools_list(input_tools)
    prompt_tool_input = create_prompt_tool_input(tools_list)

    if len(tools_list) == 1:
        # Grammar for a single tool
        output = ("start: TEXT | fun_call\n"
                  "TEXT: /[^{](.|\\n)*/\n"
                  " fun_call: <|tool_call|> %json " + json.dumps(convert_tool_to_grammar_input(tools_list[0])))
        return prompt_tool_input, output
    else:
        # Grammar for multiple tools (parallel calls)
        grammar_tools = [json.dumps(convert_tool_to_grammar_input(tool)) for tool in tools_list]
        output = ("start: TEXT | fun_call \n"
                  "TEXT: /[^{](.|\n)*/ \n"
                  "fun_call: <|tool_call|> %json {\"anyOf\": [" + ','.join(grammar_tools) + "]}")
        return prompt_tool_input, output
```

Test the grammar generation function to see its output.

```python
prompt_tool_input, guidance_input = get_lark_grammar(tool_list)
print("Grammar Generated Successfully.")
```

## Step 6: Prepare the System Prompt

Create a system prompt that informs the model about its capabilities and the available tools.

```python
system_prompt = "You are a helpful assistant with these tools."

# Format the message with the tool list
messages = f"""[{{"role": "system", "content": "{system_prompt}", "tools": "{prompt_tool_input}"}}]"""

# Apply the model's chat template to the system prompt
tokenizer_input_system_prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=False)

# Encode the prompt into tokens
input_tokens = tokenizer.encode(tokenizer_input_system_prompt)
input_tokens = input_tokens[:-1]  # Remove the end-of-text token
system_prompt_length = len(input_tokens)
```

## Step 7: Initialize the Guided Generator

Set up the text generator, applying the search parameters and the Lark grammar you created to guide the output.

```python
# Create generator parameters
params = og.GeneratorParams(model)
params.set_search_options(**search_options)

# Apply the Lark grammar for guided, structured generation
params.set_guidance("lark_grammar", guidance_input)

# Initialize the generator and feed it the system prompt tokens
generator = og.Generator(model, params)
generator.append_tokens(input_tokens)
```

## Step 8: Execute a Parallel Function Call

Now, test the setup with a user request that requires both a flight and a hotel booking.

```python
# User query requiring multiple function calls
user_text = "book flight ticket from Beijing to Paris(using airport code) in 2025-12-04 to 2025-12-10 , then book hotel from 2025-12-04 to 2025-12-10 in Paris"

# Format and tokenize the user message
messages = f"""[{{"role": "user", "content": "{user_text}"}}]"""
user_prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)
input_tokens = tokenizer.encode(user_prompt)

# Append the user prompt tokens to the generator
generator.append_tokens(input_tokens)
```

## Step 9: Generate and Stream the Function Calls

Run the generation loop. The guided grammar ensures the model outputs a valid JSON array containing calls to both defined functions.

```python
print("Generating function calls...")
while not generator.is_done():
    generator.generate_next_token()
    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end='', flush=True)
```

**Expected Output:**
The model should generate a JSON array with two objects: one for the flight booking and one for the hotel booking.

```json
[{"name": "booking_flight_tickets", "arguments": {"origin_airport_code": "PEK", "destination_airport_code": "CDG", "departure_date": "2025-12-04", "return_date": "2025-12-10"}}, {"name": "booking_hotels", "arguments": {"destination": "Paris", "check_in_date": "2025-12-04", "checkout_date": "2025-12-10"}}]
```

## Conclusion

You have successfully implemented parallel function calling using Phi-4 Mini and ONNX Runtime GenAI. This guide covered:

1.  **Model Setup**: Loading and configuring the Phi-4 Mini ONNX model.
2.  **Tool Definition**: Creating schemas for executable functions.
3.  **Guided Generation**: Using Lark grammar to enforce valid JSON output.
4.  **Parallel Execution**: Handling a single user query that triggers multiple, structured function calls.

This approach provides a robust foundation for building AI assistants that can reliably interact with external tools and APIs.

### Next Steps
-   Experiment with adding more complex tools to the `tool_list`.
-   Integrate the generated JSON output with actual backend APIs to execute the bookings.
-   Explore the [ONNX Runtime GenAI documentation](https://onnxruntime.ai/docs/genai/) for advanced features like different search strategies.