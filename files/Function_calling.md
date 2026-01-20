##### Copyright 2025 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: Function calling with Python

<a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Function_calling.ipynb"></a>

 Function calling lets developers create a description of a function in their code, then pass that description to a language model in a request. The response from the model includes the name of a function that matches the description and the arguments to call it with. Function calling lets you use functions as tools in generative AI applications, and you can define more than one function within a single request.

This notebook provides code examples to help you get started. The documentation's [quickstart](https://ai.google.dev/gemini-api/docs/function-calling#python) is also a good place to start understanding function calling.

## Setup

### Install dependencies


```
%pip install -qU 'google-genai>=1.0.0'
```

### Set up your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](../quickstarts/Authentication.ipynb) quickstart for an example.


```
from google import genai
from google.colab import userdata

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
```

### Choose a model

Function calling works on all Gemini models.


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

## Setting up Functions as Tools

To use function calling, pass a list of functions to the `tools` parameter when creating a [`GenerativeModel`](https://ai.google.dev/api/python/google/generativeai/GenerativeModel). The model uses the function name, docstring, parameters, and parameter type annotations to decide if it needs the function to best answer a prompt.

> Important: The SDK converts function parameter type annotations to a format the API understands (`genai.types.FunctionDeclaration`). The API only supports a limited selection of parameter types, and the Python SDK's automatic conversion only supports a subset of that: `AllowedTypes = int | float | bool | str | list['AllowedTypes'] | dict`


**Example: Lighting System Functions**

Here are 3 functions controlling a hypothetical lighting system. Note the docstrings and type hints.


```
def enable_lights():
    """Turn on the lighting system."""
    print("LIGHTBOT: Lights enabled.")


def set_light_color(rgb_hex: str):
    """Set the light color. Lights must be enabled for this to work."""
    print(f"LIGHTBOT: Lights set to {rgb_hex}.")

def stop_lights():
    """Stop flashing lights."""
    print("LIGHTBOT: Lights turned off.")

light_controls = [enable_lights, set_light_color, stop_lights]
instruction = """
  You are a helpful lighting system bot. You can turn
  lights on and off, and you can set the color. Do not perform any
  other tasks.
"""
```

## Basic Function Calling with Chat

Function calls naturally fit into multi-turn conversations. The Python SDK's `ChatSession (client.chats.create(...))` is ideal for this, as it automatically handles conversation history.

Furthermore, `ChatSession` simplifies function calling execution via its `automatic_function_calling` feature (enabled by default), which will be explored more later. For now, let's see a basic interaction where the model decides to call a function.


```
chat = client.chats.create(
    model=MODEL_ID,
    config={
        "tools": light_controls,
        "system_instruction": instruction,
        # automatic_function_calling defaults to enabled
    }
)

response = chat.send_message("It's awful dark in here...")

print(response.text)
```

    LIGHTBOT: Lights enabled.
    No problem, I can turn the lights on for you.
    


## Examining Function Calls and Execution History

To understand what happened in the background, you can examine the chat history.

The `Chat.history` property stores a chronological record of the conversation between the user and the Gemini model. You can get the history using `Chat.get_history()`. Each turn in the conversation is represented by a `genai.types.Content` object, which contains the following information:

**Role**: Identifies whether the content originated from the "user" or the "model".

**Parts**: A list of genai.types.Part objects that represent individual components of the message. With a text-only model, these parts can be:

* **Text**: Plain text messages.
* **Function Call (genai.types.FunctionCall)**: A request from the model to execute a specific function with provided arguments.
* **Function Response (genai.types.FunctionResponse)**: The result returned by the user after executing the requested function.



```
from IPython.display import Markdown, display

def print_history(chat):
  for content in chat.get_history():
      display(Markdown("###" + content.role + ":"))
      for part in content.parts:
          if part.text:
              display(Markdown(part.text))
          if part.function_call:
              print("Function call: {", part.function_call, "}")
          if part.function_response:
              print("Function response: {", part.function_response, "}")
      print("-" * 80)

print_history(chat)
```


###user:



It's awful dark in here...


    --------------------------------------------------------------------------------



###model:


    Function call: { id=None args={} name='enable_lights' }
    --------------------------------------------------------------------------------



###user:


    Function response: { id=None name='enable_lights' response={'result': None} }
    --------------------------------------------------------------------------------



###model:



No problem, I can turn the lights on for you.



    --------------------------------------------------------------------------------


This history shows the flow:

1. **User**: Sends the message.

2. **Model**: Responds not with text, but with a `FunctionCall` requesting `enable_lights`.

3. **User (SDK)**: The `ChatSession` automatically executes `enable_lights()` because `automatic_function_calling` is enabled. It sends the result back as a `FunctionResponse`.

4. **Model**: Uses the function's result ("Lights enabled.") to formulate the final text response.

## Automatic Function Execution (Python SDK Feature)

As demonstrated above, the `ChatSession` in the Python SDK has a powerful feature called Automatic Function Execution. When enabled (which it is by default), if the model responds with a FunctionCall, the SDK will:

1. Find the corresponding Python function in the provided `tools`.

2. Execute the function with the arguments provided by the model.

3. Send the function's return value back to the model in a `FunctionResponse`.

4. Return only the model's final response (usually text) to your code.

This significantly simplifies the workflow for common use cases.

**Example: Math Operations**


```
from google.genai import types # Ensure types is imported

def add(a: float, b: float):
    """returns a + b."""
    return a + b

def subtract(a: float, b: float):
    """returns a - b."""
    return a - b

def multiply(a: float, b: float):
    """returns a * b."""
    return a * b

def divide(a: float, b: float):
    """returns a / b."""
    if b == 0:
        return "Cannot divide by zero."
    return a / b

operation_tools = [add, subtract, multiply, divide]

chat = client.chats.create(
    model=MODEL_ID,
    config={
        "tools": operation_tools,
        "automatic_function_calling": {"disable": False} # Enabled by default
    }
)

response = chat.send_message(
    "I have 57 cats, each owns 44 mittens, how many mittens is that in total?"
)

print(response.text)
```

    That would be 2508 mittens in total.



```
print_history(chat)
```


###user:



I have 57 cats, each owns 44 mittens, how many mittens is that in total?


    --------------------------------------------------------------------------------



###model:


    Function call: { id=None args={'a': 57, 'b': 44} name='multiply' }
    --------------------------------------------------------------------------------



###user:


    Function response: { id=None name='multiply' response={'result': 2508} }
    --------------------------------------------------------------------------------



###model:



That would be 2508 mittens in total.


    --------------------------------------------------------------------------------


Automatic execution handled the `multiply` call seamlessly.

## Automatic Function Schema Declaration

A key convenience of the Python SDK is its ability to automatically generate the required `FunctionDeclaration` schema from your Python functions. It inspects:

- **Function Name**: (`func.__name__`)

- **Docstring**: Used for the function's description.

- **Parameters**: Names and type annotations (`int`, `str`, `float`, `bool`, `list`, `dict`). Docstrings for parameters (if using specific formats like Google style) can also enhance the description.

- **Return Type Annotation**: Although not strictly used by the model for deciding which function to call, it's good practice.

You generally don't need to create `FunctionDeclaration` objects manually when using Python functions directly as tools.

However, you can generate the schema explicitly using `genai.types.FunctionDeclaration.from_callable` if you need to inspect it, modify it, or use it in scenarios where you don't have the Python function object readily available.


```
import json

set_color_declaration = types.FunctionDeclaration.from_callable(
    callable = set_light_color,
    client = client
)

print(json.dumps(set_color_declaration.to_json_dict(), indent=4))
```

    {
        "description": "Set the light color. Lights must be enabled for this to work.",
        "name": "set_light_color",
        "parameters": {
            "properties": {
                "rgb_hex": {
                    "type": "STRING"
                }
            },
            "required": [
                "rgb_hex"
            ],
            "type": "OBJECT"
        }
    }


## Manual function calling

For more control, or if automatic function calling is not available,  you can process [`genai.types.FunctionCall`](https://googleapis.github.io/python-genai/genai.html#genai.types.FunctionCall) requests from the model yourself. This would be the case if:

- You use a `Chat` with the default `"automatic_function_calling": {"disable": True}`.
- You use [`Client.model.generate_content`](https://googleapis.github.io/python-genai/genai.html#genai.types.) (and manage the chat history yourself).

**Example: Movies**

The following example is a rough equivalent of the [function calling single-turn curl sample](https://ai.google.dev/docs/function_calling#function-calling-single-turn-curl-sample) in Python. It uses functions that return (mock) movie playtime information, possibly from a hypothetical API:


```
def find_movies(description: str, location: str):
    """find movie titles currently playing in theaters based on any description, genre, title words, etc.

    Args:
        description: Any kind of description including category or genre, title words, attributes, etc.
        location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
    """
    return ["Barbie", "Oppenheimer"]


def find_theaters(location: str, movie: str):
    """Find theaters based on location and optionally movie title which are currently playing in theaters.

    Args:
        location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
        movie: Any movie title
    """
    return ["Googleplex 16", "Android Theatre"]


def get_showtimes(location: str, movie: str, theater: str, date: str):
    """
    Find the start times for movies playing in a specific theater.

    Args:
      location: The city and state, e.g. San Francisco, CA or a zip code e.g. 95616
      movie: Any movie title
      thearer: Name of the theater
      date: Date for requested showtime
    """
    return ["10:00", "11:00"]

theater_functions = [find_movies, find_theaters, get_showtimes]
```

After using `generate_content()` to ask a question, the model requests a `function_call`:


```
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Which theaters in Mountain View, CA show the Barbie movie?",
    config = {
        "tools": theater_functions,
        "automatic_function_calling": {"disable": True}
    }
)

print(json.dumps(response.candidates[0].content.parts[0].to_json_dict(), indent=4))
```

    {
        "function_call": {
            "args": {
                "location": "Mountain View, CA",
                "movie": "Barbie"
            },
            "name": "find_theaters"
        }
    }


Since this is not using a `ChatSession` with automatic function calling, you have to call the function yourself.

A very simple way to do this would be with `if` statements:

```python
if function_call.name == 'find_theaters':
  find_theaters(**function_call.args)
elif ...
```

However, since you already made the `theater_functions` list, this can be simplified to:



```
def call_function(function_call, functions):
    function_name = function_call.name
    function_args = function_call.args
    # Find the function object from the list based on the function name
    for func in functions:
        if func.__name__ == function_name:
            return func(**function_args)

part = response.candidates[0].content.parts[0]

# Check if it's a function call; in real use you'd need to also handle text
# responses as you won't know what the model will respond with.
if part.function_call:
    result = call_function(part.function_call, theater_functions)

print(result)
```

    ['Googleplex 16', 'Android Theatre']


Finally, pass the response plus the message history to the next `generate_content()` call to get a final text response from the model. The next code cell is showing on purpose different ways to write down `Content` so you can choose the one that you prefer.


```
from google.genai import types
# Build the message history
messages = [
    types.Content(
        role="user",
        parts=[
            types.Part(
                text="Which theaters in Mountain View show the Barbie movie?."
            )
        ]
    ),
    types.Content(
        role="model",
        parts=[part]
    ),
    types.Content(
        role="tool",
        parts=[
            types.Part.from_function_response(
                name=part.function_call.name,
                response={"output":result},
            )
        ]
    )
]

# Generate the next response
response = client.models.generate_content(
    model=MODEL_ID,
    contents=messages,
    config = {
        "tools": theater_functions,
        "automatic_function_calling": {"disable": True}
    }
)
print(response.text)
```

    The Barbie movie is currently playing at the Googleplex 16 and Android Theatre in Mountain View.


This demonstrates the manual workflow: call, check, execute, respond, call again.

## Parallel function calls

The Gemini API can call multiple functions in a single turn. This caters for scenarios where there are multiple function calls that can take place independently to complete a task.

First set the tools up. Unlike the movie example above, these functions do not require input from each other to be called so they should be good candidates for parallel calling.


```
def power_disco_ball(power: bool) -> bool:
    """Powers the spinning disco ball."""
    print(f"Disco ball is {'spinning!' if power else 'stopped.'}")
    return True

def start_music(energetic: bool, loud: bool, bpm: int) -> str:
    """Play some music matching the specified parameters.

    Args:
      energetic: Whether the music is energetic or not.
      loud: Whether the music is loud or not.
      bpm: The beats per minute of the music.

    Returns: The name of the song being played.
    """
    print(f"Starting music! {energetic=} {loud=}, {bpm=}")
    return "Never gonna give you up."


def dim_lights(brightness: float) -> bool:
    """Dim the lights.

    Args:
      brightness: The brightness of the lights, 0.0 is off, 1.0 is full.
    """
    print(f"Lights are now set to {brightness:.0%}")
    return True

house_fns = [power_disco_ball, start_music, dim_lights]
```

Now call the model with an instruction that could use all of the specified tools.


```
# You generally set "mode": "any" to make sure Gemini actually *uses* the given tools.
party_chat = client.chats.create(
    model=MODEL_ID,
    config={
        "tools": house_fns,
        "tool_config" : {
            "function_calling_config": {
                "mode": "any"
            }
        }
    }
)

# Call the API
response = party_chat.send_message(
    "Turn this place into a party!"
)


print_history(party_chat)
```

    [Disco ball is spinning!, Starting music! energetic=True loud=True, bpm=130, Lights are now set to 30%, ..., Disco ball is spinning!, Starting music! energetic=True loud=True, bpm=130, Lights are now set to 30%]



###user:



Turn this place into a party!


    --------------------------------------------------------------------------------



###model:


    Function call: { id=None args={'power': True} name='power_disco_ball' }
    Function call: { id=None args={'energetic': True, 'bpm':