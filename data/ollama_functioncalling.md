# Implementing Function Calling with Phi-4-mini and Ollama

This guide walks you through setting up and using function calling with Microsoft's Phi-4-mini model via Ollama. You'll learn how to configure the model for function calling, execute single function calls, and handle parallel function calls for multi-step workflows.

## Prerequisites

Before you begin, ensure you have:

1. **Ollama version 0.5.13 or higher** installed
2. **Phi-4-mini:3.8b-fp16** model pulled and ready
3. **Python 3.7+** with the `requests` library installed

## Step 1: Set Up Ollama with Custom Template

Phi-4-mini requires a specific template configuration to enable function calling. Follow these steps:

### 1.1 Run the Model
First, verify your model is working correctly:

```bash
ollama run phi4-mini:3.8b-fp16
```

### 1.2 Create a Modelfile
Create a file called `Modelfile` with the following content:

```txt
FROM phi4-mini:3.8b-fp16

TEMPLATE """
{{- if .Messages }}
{{- if or .System .Tools }}<|system|>

{{ if .System }}{{ .System }}
{{- end }}
In addition to plain text responses, you can chose to call one or more of the provided functions.

Use the following rule to decide when to call a function:
  * if the response can be generated from your internal knowledge (e.g., as in the case of queries like "What is the capital of Poland?"), do so
  * if you need external information that can be obtained by calling one or more of the provided functions, generate a function calls

If you decide to call functions:
  * prefix function calls with functools marker (no closing marker required)
  * all function calls should be generated in a single JSON list formatted as functools[{"name": [function name], "arguments": [function arguments as JSON]}, ...]
  * follow the provided JSON schema. Do not hallucinate arguments or values. Do to blindly copy values from the provided samples
  * respect the argument type formatting. E.g., if the type if number and format is float, write value 7 as 7.0
  * make sure you pick the right functions that match the user intent

Available functions as JSON spec:
{{- if .Tools }}
{{ .Tools }}
{{- end }}<|end|>
{{- end }}
{{- range .Messages }}
{{- if ne .Role "system" }}<|{{ .Role }}|>
{{- if and .Content (eq .Role "tools") }}

{"result": {{ .Content }}}
{{- else if .Content }}

{{ .Content }}
{{- else if .ToolCalls }}

functools[
{{- range .ToolCalls }}{{ "{" }}"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}{{ "}" }}
{{- end }}]
{{- end }}<|end|>
{{- end }}
{{- end }}<|assistant|>

{{ else }}
{{- if .System }}<|system|>

{{ .System }}<|end|>{{ end }}{{ if .Prompt }}<|user|>

{{ .Prompt }}<|end|>{{ end }}<|assistant|>

{{ end }}{{ .Response }}{{ if .Response }}<|user|>{{ end }}
"""
```

### 1.3 Create the Custom Model
Create a new model with your custom template:

```bash
ollama create phi4-mini-func-call -f /path/to/your/Modelfile
```

Replace `/path/to/your/Modelfile` with the actual path to your Modelfile.

## Step 2: Set Up Your Python Environment

Import the necessary libraries:

```python
import requests
import json
```

## Step 3: Implement Single Function Calling

Let's start with a simple weather function example.

### 3.1 Define Your Function
First, create a mock weather function:

```python
def get_current_weather(location, format="json"):
    return "Today " + location + " is sunny and 20 degrees " + format
```

### 3.2 Configure the API Request
Set up the API endpoint and payload:

```python
url = "http://localhost:11434/api/chat"

payload = {
    "model": "phi4-mini-func-call",  # Use your custom model name
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant with some tools."
        },
        {
            "role": "user",
            "content": "What is the weather today in Paris?"
        }
    ],
    "stream": False,
    "options": {
        "max_new_tokens": 1024,
        "return_full_text": False,
        "temperature": 0.00001,
        "top_p": 1.0,
        "do_sample": False
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the weather for, e.g. San Francisco, CA"
                        },
                        "format": {
                            "type": "string",
                            "description": "The format to return the weather in, e.g. celsius or fahrenheit",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location", "format"]
                }
            }
        }
    ]
}
```

### 3.3 Send the Request and Parse the Response
Execute the API call and examine the response:

```python
response = requests.post(url, json=payload)
response.raise_for_status()  # Raise an exception for bad status codes

# Parse the response
result = response.json()
print("Response:")
print(json.dumps(result, indent=2, ensure_ascii=False))

# Extract tool calls
if "message" in result and "tool_calls" in result["message"]:
    print("\nAssistant's tool calls:")
    print(result["message"]["tool_calls"])
```

You should see output similar to:

```json
{
  "model": "phi4-mini-func-call",
  "created_at": "2025-03-10T06:21:45.070769Z",
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "get_current_weather",
          "arguments": {
            "format": "celsius",
            "location": "Paris"
          }
        }
      }
    ]
  },
  "done_reason": "stop",
  "done": true
}
```

### 3.4 Execute the Function Call
Now, map the tool call to your actual function and execute it:

```python
# Create a mapping between function names and actual functions
tools_mapping_functions = {
    "get_current_weather": get_current_weather
}

# Process each tool call
for item in result["message"]["tool_calls"]:
    if item["function"]["name"] == "get_current_weather":
        print("\nTool call details:")
        print(json.dumps(item, indent=2, ensure_ascii=False))
        
        print("\nTool call arguments:")
        print(json.dumps(item["function"]["arguments"], indent=2, ensure_ascii=False))
        
        # Execute the function
        func_call = tools_mapping_functions[item["function"]["name"]]
        tool_response = func_call(**item["function"]["arguments"])
        print("\nFunction execution result: " + tool_response)
```

The output will show:

```
Function execution result: Today Paris is sunny and 20 degrees celsius
```

## Step 4: Implement Parallel Function Calling

Now let's implement a more complex example with multiple functions that can be called in parallel.

### 4.1 Define Multiple Functions
Create functions for booking flights and hotels:

```python
def booking_flight_tickets(origin_airport_code, destination_airport_code, departure_date, return_date):
    return "Your book flights from " + origin_airport_code + " to " + destination_airport_code + " on " + departure_date + " and return on " + return_date

def booking_hotels(destination, check_in_date, checkout_date):
    return "Your book hotels in " + destination + " from " + check_in_date + " to " + checkout_date
```

### 4.2 Configure the API Request with Multiple Tools
Set up a payload with multiple available functions:

```python
payload = {
    "model": "phi4-mini-func-call",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant using the provided tools."
        },
        {
            "role": "user",
            "content": "book a hotel and flight ticket from Beijing to Paris(using airport code) in 2025-12-04 to 2025-12-10"
        }
    ],
    "stream": False,
    "options": {
        "max_new_tokens": 1024,
        "return_full_text": False,
        "temperature": 0.00001,
        "top_p": 1.0,
        "do_sample": False
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "booking_flight_tickets",
                "description": "booking flights",
                "parameters": {
                    "origin_airport_code": {
                        "description": "The name of Departure airport code",
                        "type": "str"
                    },
                    "destination_airport_code": {
                        "description": "The name of Destination airport code",
                        "type": "str"
                    },
                    "departure_date": {
                        "description": "The date of outbound flight",
                        "type": "str"
                    },
                    "return_date": {
                        "description": "The date of return flight",
                        "type": "str"
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "booking_hotels",
                "description": "booking hotel",
                "parameters": {
                    "destination": {
                        "description": "The name of the city",
                        "type": "str"
                    },
                    "check_in_date": {
                        "description": "The date of check in",
                        "type": "str"
                    },
                    "checkout_date": {
                        "description": "The date of check out",
                        "type": "str"
                    }
                }
            }
        }
    ]
}
```

### 4.3 Send the Request and Process Parallel Tool Calls
Execute the API call and handle multiple tool calls:

```python
# Update the function mapping
tools_mapping_functions = {
    "booking_flight_tickets": booking_flight_tickets,
    "booking_hotels": booking_hotels
}

# Send the request
response = requests.post(url, json=payload)
response.raise_for_status()

# Parse the response
result = response.json()
print("Response:")
print(json.dumps(result, indent=2, ensure_ascii=False))

# Show the tool calls
if "message" in result and "tool_calls" in result["message"]:
    print("\nAssistant's parallel tool calls:")
    print(result["message"]["tool_calls"])
```

You'll see output showing both functions were called:

```json
{
  "model": "phi4-mini-func-call",
  "created_at": "2025-03-10T06:21:48.387487Z",
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "booking_hotels",
          "arguments": {
            "check_in_date": "2025-12-04",
            "checkout_date": "2025-12-10",
            "destination": "Paris"
          }
        }
      },
      {
        "function": {
          "name": "booking_flight_tickets",
          "arguments": {
            "departure_date": "2025-12-04",
            "destination_airport_code": "CDG",
            "origin_airport_code": "PEK",
            "return_date": "2025-12-10"
          }
        }
      }
    ]
  },
  "done_reason": "stop",
  "done": true
}
```

### 4.4 Execute All Function Calls
Process and execute each tool call:

```python
for item in result["message"]["tool_calls"]:
    func_call = tools_mapping_functions[item["function"]["name"]]
    tool_response = func_call(**item["function"]["arguments"])
    print("\nFunction execution: " + tool_response)
```

The output will show both bookings were processed:

```
Function execution: Your book hotels in Paris from 2025-12-04 to 2025-12-10

Function execution: Your book flights from PEK to CDG on 2025-12-04 and return on 2025-12-10
```

## Key Takeaways

1. **Template Configuration**: Phi-4-mini requires a specific template format to enable function calling capabilities through Ollama.

2. **Function Definition**: Each function must be defined with a clear name, description, and parameter schema that matches the tool definition in the API call.

3. **Tool Mapping**: Create a dictionary that maps function names from the LLM's tool calls to your actual Python functions.

4. **Parallel Execution**: The model can call multiple functions in a single response when appropriate, allowing for complex multi-step workflows.

5. **Schema Consistency**: Ensure your function parameter schemas in the `tools` array match the actual function signatures in your code.

## Next Steps

- Experiment with more complex function schemas and parameter types
- Implement error handling for function execution failures
- Create a loop to continue the conversation after function execution
- Explore streaming responses for real-time function calling

By following this guide, you've successfully implemented both single and parallel function calling with Phi-4-mini through Ollama, enabling your AI assistant to interact with external tools and services.