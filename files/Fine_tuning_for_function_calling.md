# Fine tuning with function-calling

This notebook covers how to fine-tune to increase function calling accuracy and reliability. You can find more information on function calling [here](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb), and on fine tuning [here](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_finetune_chat_models.ipynb)

For context, from the function calling notebook above:

> `tools` is an optional parameter in the Chat Completion API which can be used to provide function specifications. The purpose of this is to enable models to generate function arguments which adhere to the provided specifications. Note that the API will not actually execute any function calls. It is up to developers to execute function calls using model outputs.

Function calling is a very powerful tool when it functions as intended. However, we have seen that as the number of functions increases, and the complexity of the task at hand increases, function calling becomes less accurate (e.g.: more hallucinated invocations, and incorrect invocations).

Before fine tuning for function calling, it's best to begin with:

- Improvements to the function definitions. Make them more clear, and more distinct from one another.
- Experiment with prompt engineering: often a more detailed prompt can help the model call the correct function.

_If_ the steps above fail to improve function calling to a satisfactory level, then you can try fine tuning for function calling.

### Overview

This notebook contains three sections

- **Assessing baseline function calling performance:** Evaluating an out-of-the-box `gpt-3.5-turbo` model on our given function (let's assume that for latency + cost reasons we cannot use `gpt-4o` for a drone copilot)
- **Generating synthetic data:** Using `gpt-4o` to create 'golden' set of prompts and function invocations to use as training data
- **Fine-tuning**: Running the fine tuning job, and evaluating the fine-tuned model

Note: _This notebook provides an example of how to create synthetic training data for fine tuning for function calling given just a list of functions. While real-world production test evals are preferable, this method produces strong results and can be used in conjunction with real-world training data._

# Getting baseline function calling performance

```python
#!pip install tenacity -q
#!pip install openai -q
#!pip install typing -q
# !pip install python-dotenv
```

```python
import numpy as np
import json
import os
from IPython.display import display
import pandas as pd
from openai import OpenAI
import itertools
import time
import base64
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Any, Dict, List, Generator
import ast

%load_ext dotenv
%dotenv

client = OpenAI(api_key=os.environ.get("OPENAI_BUILD_HOUR_KEY"))
```

    The dotenv extension is already loaded. To reload it, use:
      %reload_ext dotenv

### Utilities

Let's define utility functions for making calls to the Chat Completions API, one to get the completion and one to get the function call.

```python
def get_chat_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-3.5-turbo",
    max_tokens=500,
    temperature=0.0,
    stop=None,
    tools=None,
    seed=42,
    functions=None,
    tool_choice=None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "tools": tools,
        "seed": seed,
        "tool_choice": tool_choice,
    }
    if functions:
        params["functions"] = functions

    completion = client.chat.completions.create(**params)
    return completion.choices[0].message, completion.usage


def eval(model: str, system_prompt: str, function_list, prompts_to_expected_tool_name):
    """
    Evaluate the performance of a model in selecting the correct function based on given prompts.

    Args:
        model (str): The name of the model to be evaluated.
        system_prompt (str): The system prompt to be used in the chat completion.
        function_list (list): A list of functions that the model can call.
        prompts_to_expected_tool_name (dict): A dictionary mapping prompts to their expected function names.

    Returns:
        None
    """

    prompts_to_actual = []
    latencies = []
    tokens_used = []

    for prompt, expected_function in prompts_to_expected_tool_name.items():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        start_time = time.time()
        completion, usage = get_chat_completion(
            model=model,
            messages=messages,
            seed=42,
            tools=function_list,
            temperature=0.0,
            tool_choice="required",
        )
        end_time = time.time()

        latency = (end_time - start_time) * 1000  # convert to milliseconds
        latencies.append(latency)

        prompts_to_actual.append(
            {prompt: completion.tool_calls[0].function.name})

        # Calculate tokens used
        tokens_used.append(usage.total_tokens)

    total_prompts = len(prompts_to_expected_tool_name)

    # Calculate the number of matches
    matches = sum(
        1
        for result in prompts_to_actual
        if list(result.values())[0]
        == prompts_to_expected_tool_name[list(result.keys())[0]]
    )
    match_percentage = (matches / total_prompts) * 100

    # Calculate average latency
    avg_latency = sum(latencies) / total_prompts
    # Calculate average tokens used
    avg_tokens_used = sum(tokens_used) / total_prompts

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(columns=["Prompt", "Expected", "Match"])

    results_list = []
    for result in prompts_to_actual:
        prompt = list(result.keys())[0]
        actual_function = list(result.values())[0]
        expected_function = prompts_to_expected_tool_name[prompt]
        match = actual_function == expected_function
        results_list.append(
            {
                "Prompt": prompt,
                "Actual": actual_function,
                "Expected": expected_function,
                "Match": "Yes" if match else "No",
            }
        )
    results_df = pd.DataFrame(results_list)

    def style_rows(row):
        match = row["Match"]
        background_color = "red" if match == "No" else "white"
        return ["background-color: {}; color: black".format(background_color)] * len(
            row
        )

    styled_results_df = results_df.style.apply(style_rows, axis=1)

    # Display the DataFrame as a table
    display(styled_results_df)

    print(
        f"Number of matches: {matches} out of {total_prompts} ({match_percentage:.2f}%)"
    )
    print(f"Average latency per request: {avg_latency:.2f} ms")
    print(f"Average tokens used per request: {avg_tokens_used:.2f}")
```

### Baseline testing

Let's build an intelligent drone co-pilot. We want to be able to give the co-pilot commands, and have it either call the function
for that command, or deny that request if the command is unfeasible.
We can first define a system prompt for the copilot.

```python
DRONE_SYSTEM_PROMPT = """You are an intelligent AI that controls a drone. Given a command or request from the user,
call one of your functions to complete the request. If the request cannot be completed by your available functions, call the reject_request function.
If the request is ambiguous or unclear, reject the request."""
```

Now let's define functions for all of the actions the copilot can take.

```python
function_list = [
    {
        "type": "function",
        "function": {
            "name": "takeoff_drone",
            "description": "Initiate the drone's takeoff sequence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "altitude": {
                        "type": "integer",
                        "description": "Specifies the altitude in meters to which the drone should ascend.",
                    }
                },
                "required": ["altitude"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "land_drone",
            "description": "Land the drone at its current location or a specified landing point.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "enum": ["current", "home_base", "custom"],
                        "description": "Specifies the landing location for the drone.",
                    },
                    "coordinates": {
                        "type": "object",
                        "description": "GPS coordinates for custom landing location. Required if location is 'custom'.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_drone_movement",
            "description": "Direct the drone's movement in a specific direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward", "left", "right", "up", "down"],
                        "description": "Direction in which the drone should move.",
                    },
                    "distance": {
                        "type": "integer",
                        "description": "Distance in meters the drone should travel in the specified direction.",
                    },
                },
                "required": ["direction", "distance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_drone_speed",
            "description": "Adjust the speed of the drone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "speed": {
                        "type": "integer",
                        "description": "Specifies the speed in km/h. Valid range is 0 to 100.",
                        "minimum": 0,
                    }
                },
                "required": ["speed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_camera",
            "description": "Control the drone's camera to capture images or videos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["photo", "video", "panorama"],
                        "description": "Camera mode to capture content.",
                    },
                    "duration": {
                        "type": "integer",
                        "description": "Duration in seconds for video capture. Required if mode is 'video'.",
                    },
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control_gimbal",
            "description": "Adjust the drone's gimbal for camera stabilization and direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tilt": {
                        "type": "integer",
                        "description": "Tilt angle for the gimbal in degrees.",
                    },
                    "pan": {
                        "type": "integer",
                        "description": "Pan angle for the gimbal in degrees.",
                    },
                },
                "required": ["tilt", "pan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_drone_lighting",
            "description": "Control the drone's lighting for visibility and signaling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["on", "off", "blink", "sos"],
                        "description": "Lighting mode for the drone.",
                    }
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "return_to_home",
            "description": "Command the drone to return to its home or launch location.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_battery_saver_mode",
            "description": "Toggle battery saver mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Toggle battery saver mode.",
                    }
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_obstacle_avoidance",
            "description": "Configure obstacle avoidance settings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Toggle obstacle avoidance.",
                    }
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_follow_me_mode",
            "description": "Enable or disable 'follow me' mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Toggle 'follow me' mode.",
                    }
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calibrate_sensors",
            "description": "Initiate calibration sequence for drone's sensors.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_autopilot",
            "description": "Enable or disable autopilot mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["on", "off"],
                        "description": "Toggle autopilot mode.",
                    }
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "configure_led_display",
            "description": "Configure the drone's LED display pattern and colors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "enum": ["solid", "blink", "pulse", "rainbow"],
                        "description": "Pattern for the LED display.",
                    },
                    "color": {
                        "type": "string",
                        "enum": ["red", "blue", "green", "yellow", "white"],
                        "description": "Color for the LED display. Not required if pattern is 'rainbow'.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_home_location",
            "description": "Set or change the home location for the drone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinates": {
                        "type": "object",
                        "description": "GPS coordinates for the home location.",
                    }
                },
                "required": ["coordinates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reject_request",
            "description": "Use this function if the request is not possible.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
```

For starters, let's see how function calling performs with some straight forward feasible prompts, and then couple of obviously impossible requests which call the 'reject_request' function.

```python
straightforward_prompts_to_expected = {
    "Land the drone at the home base": "land_drone",
    "Take off the drone to 50 meters": "takeoff_drone",
    "Change speed to 15 kilometers per hour": "set_drone_speed",
    "Turn into an elephant!": "reject_request",
    "Move the drone forward by 10 meters": "control_drone_movement",
    "I want the LED display to blink in red": "configure_led_display",
    "Can you take a photo?": "control_camera",
    "Can you detect obstacles?": "set_obstacle_avoidance",
    "Can you dance for me?": "reject_request",
    "Can you follow me?": "set_follow_me_mode",
}
```

```python
# Evaluate the model with the given prompts
eval(
    model="gpt-3.5-turbo",
    system_prompt=DRONE_SYSTEM_PROMPT,
    function_list=function_list,
    prompts_to_expected_tool_name=straightforward_prompts_to_expected,
)
```

Number of matches: 10 out of 10 (100.00%)
Average latency per request: 826.81 ms
Average tokens used per request: 796.20

Nice! The model performs quite well with these requests. Now let's try some more difficult requests: requests that are _almost_ feasible and are drone-related, but that the drone cannot actually do, and the pilot should reject.

```python
challenging_prompts_to_expected = {
    "Play pre-recorded audio message": "reject_request",
    "Initiate following on social media": "reject_request",
    "Scan environment for heat signatures": "reject_request",
    "Bump into obstacles": "reject_request",
    "Change drone's paint job color": "reject_request",
    "Coordinate with nearby drones": "reject_request",
    "Change speed to negative 120 km/h": "reject_request",
    "Detect a person": "reject_request",
    "Please enable night vision": "reject_request",
    "Report on humidity levels around you": "reject_request",
}
```

```python
# Evaluate the model with the challenging prompts
eval