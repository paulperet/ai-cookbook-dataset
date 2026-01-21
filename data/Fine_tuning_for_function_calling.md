# Fine-Tuning for Improved Function Calling

This guide demonstrates how to fine-tune a model to increase the accuracy and reliability of function calling. Function calling allows a model to generate structured arguments for predefined functions, but performance can degrade as the number of functions or task complexity increases. Before fine-tuning, you should first attempt to improve function definitions and use prompt engineering. If those steps are insufficient, fine-tuning can be a powerful solution.

## Overview

This tutorial is structured into three main parts:

1.  **Assessing Baseline Performance:** Evaluate the out-of-the-box `gpt-3.5-turbo` model on a set of drone copilot functions.
2.  **Generating Synthetic Data:** Use `gpt-4o` to create a high-quality training dataset of prompts and corresponding function calls.
3.  **Fine-Tuning:** Execute a fine-tuning job and evaluate the performance of the new model.

> **Note:** While real-world evaluation data is ideal, this method of generating synthetic data from function definitions produces strong results and can be combined with real data.

## Prerequisites

Ensure you have the required Python packages installed and your OpenAI API key configured.

```bash
pip install openai tenacity pandas python-dotenv
```

```python
import os
import json
import time
import pandas as pd
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Load your OpenAI API key from an environment variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

## Part 1: Assessing Baseline Function Calling Performance

First, we'll establish a performance baseline using `gpt-3.5-turbo`. We'll build an intelligent drone copilot that can interpret user commands and call the appropriate function—or reject infeasible requests.

### Step 1: Define the System Prompt and Utility Functions

Create a system prompt that defines the AI's role and a helper function to interact with the Chat Completions API.

```python
DRONE_SYSTEM_PROMPT = """You are an intelligent AI that controls a drone. Given a command or request from the user,
call one of your functions to complete the request. If the request cannot be completed by your available functions, call the reject_request function.
If the request is ambiguous or unclear, reject the request."""

def get_chat_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-3.5-turbo",
    max_tokens=500,
    temperature=0.0,
    stop=None,
    tools=None,
    seed=42,
    tool_choice=None,
) -> str:
    """Helper function to call the Chat Completions API."""
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
    completion = client.chat.completions.create(**params)
    return completion.choices[0].message, completion.usage
```

### Step 2: Define the Drone's Available Functions

The drone copilot can perform various actions, each represented by a function specification. A `reject_request` function handles impossible commands.

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
    # ... (Additional functions for movement, speed, camera, gimbal, lighting, etc.)
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

> **Note:** The full `function_list` includes 16 functions. For brevity, only a few are shown here. Ensure you include all functions (`control_drone_movement`, `set_drone_speed`, `control_camera`, etc.) in your implementation.

### Step 3: Create an Evaluation Function

We need a robust way to test the model's performance. The following function runs a series of prompts, compares the model's chosen function against the expected one, and calculates metrics like accuracy and latency.

```python
def eval(model: str, system_prompt: str, function_list, prompts_to_expected_tool_name):
    """
    Evaluate the performance of a model in selecting the correct function based on given prompts.
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

        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        latencies.append(latency)

        # Record the model's chosen function
        chosen_function = completion.tool_calls[0].function.name
        prompts_to_actual.append({prompt: chosen_function})
        tokens_used.append(usage.total_tokens)

    # Calculate performance metrics
    total_prompts = len(prompts_to_expected_tool_name)
    matches = sum(
        1
        for result in prompts_to_actual
        if list(result.values())[0] == prompts_to_expected_tool_name[list(result.keys())[0]]
    )
    match_percentage = (matches / total_prompts) * 100
    avg_latency = sum(latencies) / total_prompts
    avg_tokens_used = sum(tokens_used) / total_prompts

    # Display results in a formatted table
    results_list = []
    for result in prompts_to_actual:
        prompt = list(result.keys())[0]
        actual_function = list(result.values())[0]
        expected_function = prompts_to_expected_tool_name[prompt]
        match = actual_function == expected_function
        results_list.append({
            "Prompt": prompt,
            "Actual": actual_function,
            "Expected": expected_function,
            "Match": "Yes" if match else "No",
        })

    results_df = pd.DataFrame(results_list)

    # Apply simple styling to highlight incorrect matches
    def style_rows(row):
        background_color = "red" if row["Match"] == "No" else "white"
        return [f"background-color: {background_color}; color: black"] * len(row)

    styled_results_df = results_df.style.apply(style_rows, axis=1)
    display(styled_results_df)

    print(f"Number of matches: {matches} out of {total_prompts} ({match_percentage:.2f}%)")
    print(f"Average latency per request: {avg_latency:.2f} ms")
    print(f"Average tokens used per request: {avg_tokens_used:.2f}")
```

### Step 4: Run Baseline Evaluations

Let's test the model with two sets of prompts: straightforward commands and challenging, infeasible requests.

#### 4.1 Test with Straightforward Prompts

First, evaluate clear commands that map directly to a function or are obviously impossible.

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

eval(
    model="gpt-3.5-turbo",
    system_prompt=DRONE_SYSTEM_PROMPT,
    function_list=function_list,
    prompts_to_expected_tool_name=straightforward_prompts_to_expected,
)
```

**Expected Output:**
```
Number of matches: 10 out of 10 (100.00%)
Average latency per request: 826.81 ms
Average tokens used per request: 796.20
```

The model performs perfectly on these simple prompts.

#### 4.2 Test with Challenging Prompts

Now, test with more difficult prompts that are drone-related but not supported by the available functions. The model should reject all of them.

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

eval(
    model="gpt-3.5-turbo",
    system_prompt=DRONE_SYSTEM_PROMPT,
    function_list=function_list,
    prompts_to_expected_tool_name=challenging_prompts_to_expected,
)
```

This evaluation will reveal the model's baseline performance on ambiguous or unsupported requests, providing a benchmark before fine-tuning.

## Next Steps

After establishing the baseline, the next parts of this guide will cover:
1.  **Generating Synthetic Data:** Using a more powerful model (`gpt-4o`) to create a high-quality training dataset from your function definitions.
2.  **Fine-Tuning:** Preparing the data, running a fine-tuning job with the OpenAI API, and evaluating the new custom model.

Proceed to the next section to begin creating your training data.