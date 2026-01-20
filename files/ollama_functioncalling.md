# **Function Calling Sample**

Phi-4-mini supports Function Calling. We can deploy it locally through Ollama and call Function Calling through the API interface.

Note:

1. Ollama version 0.5.13+

2. Using Phi4-mini:3.8b-fp16 

```bash
ollama run phi4-mini:3.8b-fp16 
```

3. Modify the template of Modelfile

FROM Your phi4-mini:3.8b-fp16 location

```txt
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

And Run this code

```bash
ollama create phi4-mini:3.8b-fp16 -f /Users/lokinfey/Desktop/Tools/slm/phi_family/ollamadev/Modelfile
```

```python
import requests
import json
```

## **Single Function Calling**

```python
def get_current_weather(location, format="json"):
    return "Today " + location + " is sunny and 20 degrees "+ format
```

```python
url = "http://localhost:11434/api/chat"
```

```python
payload = {
        "model": "phi4-mini:3.8b-fp16",
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

```python
response = requests.post(url, json=payload)
response.raise_for_status()  # Raise an exception for bad status codes
        
# Parse and print the response
result = response.json()
print("Response:")
print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Print just the assistant's message
if "message" in result:
    print("\nAssistant's message:")
    print(result["message"]["tool_calls"])
```

    Response:
    {
      "model": "phi4-mini:3.8b-fp16",
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
      "done": true,
      "total_duration": 1436226250,
      "load_duration": 35039375,
      "prompt_eval_count": 351,
      "prompt_eval_duration": 518000000,
      "eval_count": 32,
      "eval_duration": 878000000
    }
    
    Assistant's message:
    [{'function': {'name': 'get_current_weather', 'arguments': {'format': 'celsius', 'location': 'Paris'}}}]

```python
tools_mapping_functions = {
    "get_current_weather": get_current_weather
}
```

```python
for item in result["message"]["tool_calls"]:
    if item["function"]["name"] == "get_current_weather":
        print("\nTool call:")
        print(json.dumps(item, indent=2, ensure_ascii=False))
        print("\nTool call arguments:")
        print(json.dumps(item["function"]["arguments"], indent=2, ensure_ascii=False))
        func_call = tools_mapping_functions[item["function"]["name"]]
        tool_reponse = func_call(**item["function"]["arguments"])
        print("\nFunctiong execution:" + tool_reponse)
```

    
    Tool call:
    {
      "function": {
        "name": "get_current_weather",
        "arguments": {
          "format": "celsius",
          "location": "Paris"
        }
      }
    }
    
    Tool call arguments:
    {
      "format": "celsius",
      "location": "Paris"
    }
    
    Functiong execution:Today Paris is sunny and 20 degrees celsius

# **Parallel Function Calling**

```python
def booking_flight_tickets(origin_airport_code, destination_airport_code, departure_date, return_date):
    return "Your book flights from " + origin_airport_code + " to " + destination_airport_code + " on " + departure_date + " and return on " + return_date

def booking_hotels(destination, check_in_date, checkout_date):
    return "Your book hotels in " + destination + " from " + check_in_date + " to " + checkout_date
```

```python
payload = {
        "model": "phi4-mini:3.8b-fp16",
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
                { "type": "function",
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

```python
tools_mapping_functions = {
    "booking_flight_tickets": booking_flight_tickets,
    "booking_hotels": booking_hotels
}
```

```python
response = requests.post(url, json=payload)
response.raise_for_status()  # Raise an exception for bad status codes
        
# Parse and print the response
result = response.json()
print("Response:")
print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Print just the assistant's message
if "message" in result:
    print("\nAssistant's message:")
    print(result["message"]["tool_calls"])
```

    Response:
    {
      "model": "phi4-mini:3.8b-fp16",
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
      "done": true,
      "total_duration": 3276915375,
      "load_duration": 16144792,
      "prompt_eval_count": 339,
      "prompt_eval_duration": 307000000,
      "eval_count": 106,
      "eval_duration": 2949000000
    }
    
    Assistant's message:
    [{'function': {'name': 'booking_hotels', 'arguments': {'check_in_date': '2025-12-04', 'checkout_date': '2025-12-10', 'destination': 'Paris'}}}, {'function': {'name': 'booking_flight_tickets', 'arguments': {'departure_date': '2025-12-04', 'destination_airport_code': 'CDG', 'origin_airport_code': 'PEK', 'return_date': '2025-12-10'}}}]

```python
for item in result["message"]["tool_calls"]:
    func_call = tools_mapping_functions[item["function"]["name"]]
    tool_reponse = func_call(**item["function"]["arguments"])
    print("\nFunctiong execution:" + tool_reponse)
```

    
    Functiong execution:Your book hotels in Paris from 2025-12-04 to 2025-12-10
    
    Functiong execution:Your book flights from PEK to CDG on 2025-12-04 and return on 2025-12-10