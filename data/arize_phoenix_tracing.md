# Structured Data Extraction with Mistral AI and Phoenix Observability

In this guide, you will build a structured data extraction service using Mistral AI's tool calling feature. You'll instrument the service to emit OpenInference traces and visualize them in Phoenix for observability.

## Prerequisites

You will need:
- A Mistral AI API key
- Python 3.8+

## Setup

First, install the required packages.

```bash
pip install -q arize-phoenix jsonschema openinference-instrumentation-mistralai mistralai
```

Now, import the necessary libraries.

```python
import os
from getpass import getpass
from typing import Any, Dict

import pandas as pd
import phoenix as px
from phoenix.otel import register
from mistralai import Mistral
from mistralai.models import UserMessage, SystemMessage
from openinference.instrumentation.mistralai import MistralAIInstrumentor

pd.set_option("display.max_colwidth", None)
```

## Step 1: Configure Your Mistral API Key

Set up your Mistral client. If your API key isn't already in your environment, you'll be prompted to enter it.

```python
if not (api_key := os.getenv("MISTRAL_API_KEY")):
    api_key = getpass("🔑 Enter your Mistral AI API key: ")
client = Mistral(api_key=api_key)
```

## Step 2: Launch Phoenix for Observability

Phoenix will run in the background to collect and visualize trace data from your application.

```python
session = px.launch_app()
tracer_provider = register()
```

## Step 3: Instrument the Mistral Client

Instrument your Mistral client to emit telemetry in OpenInference format, which Phoenix can ingest.

```python
MistralAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

## Step 4: Define the Data Extraction Schema

You will extract structured attributes from natural language travel requests. Define the output schema using JSON Schema to specify the fields and their constraints.

```python
parameters_schema = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": 'The desired destination location. Use city, state, and country format when possible. If no destination is provided, return "unstated".',
        },
        "budget_level": {
            "type": "string",
            "enum": ["low", "medium", "high", "not_stated"],
            "description": 'The desired budget level. If no budget level is provided, return "not_stated".',
        },
        "purpose": {
            "type": "string",
            "enum": ["business", "pleasure", "other", "non_stated"],
            "description": 'The purpose of the trip. If no purpose is provided, return "not_stated".',
        },
    },
    "required": ["location", "budget_level", "purpose"],
}

function_schema = {
    "name": "record_travel_request_attributes",
    "description": "Records the attributes of a travel request",
    "parameters": parameters_schema,
}

tool_schema = {
    "type": "function",
    "function": function_schema,
}

system_message = "You are an assistant that parses and records the attributes of a user's travel request."
```

## Step 5: Create the Extraction Function

This function sends a travel request to the Mistral API, forcing the model to use the defined tool to return structured output.

```python
def extract_raw_travel_request_attributes_string(
    travel_request: str,
    tool_schema: Dict[str, Any],
    system_message: str,
    client: Mistral,
    model: str = "mistral-large-latest",
) -> str:
    chat_completion = client.chat.complete(
        model=model,
        messages=[
            SystemMessage(role="system", content=system_message),
            UserMessage(role="user", content=travel_request),
        ],
        tools=[tool_schema],
        # Force the model to use the tool to ensure schema-conformant output
        # tool_choice=ToolChoice.any,
    )
    return chat_completion.choices[0].message.tool_calls[0].function.arguments
```

## Step 6: Run Extraction on Sample Data

Define a list of sample travel requests and process each one through your extraction service.

```python
travel_requests = [
    "Can you recommend a luxury hotel in Tokyo with a view of Mount Fuji for a romantic honeymoon?",
    "I'm looking for a mid-range hotel in London with easy access to public transportation for a solo backpacking trip. Any suggestions?",
    "I need a budget-friendly hotel in San Francisco close to the Golden Gate Bridge for a family vacation. What do you recommend?",
    "Can you help me find a boutique hotel in New York City with a rooftop bar for a cultural exploration trip?",
    "I'm planning a business trip to Tokyo and I need a hotel near the financial district. What options are available?",
    "I'm traveling to London for a solo vacation and I want to stay in a trendy neighborhood with great shopping and dining options. Any recommendations for hotels?",
    "I'm searching for a luxury beachfront resort in San Francisco for a relaxing family vacation. Can you suggest any options?",
    "I need a mid-range hotel in New York City with a fitness center and conference facilities for a business trip. Any suggestions?",
    "I'm looking for a budget-friendly hotel in Tokyo with easy access to public transportation for a backpacking trip. What do you recommend?",
    "I'm planning a honeymoon in London and I want a luxurious hotel with a spa and romantic atmosphere. Can you suggest some options?",
]

raw_travel_attributes_column = []
for travel_request in travel_requests:
    print("Travel request:")
    print("==============")
    print(travel_request)
    print()
    raw_travel_attributes = extract_raw_travel_request_attributes_string(
        travel_request, tool_schema, system_message, client
    )
    raw_travel_attributes_column.append(raw_travel_attributes)
    print("Raw Travel Attributes:")
    print("=====================")
    print(raw_travel_attributes)
    print()
    print()
```

The output for each request will be a JSON string containing the extracted `location`, `budget_level`, and `purpose`.

## Step 7: Inspect Traces in Phoenix

All inference calls have been traced. You can now open the Phoenix UI to explore the spans and trace details.

```python
print(f"Open the Phoenix UI at: {session.url}")
```

## Step 8: Export Trace Data for Analysis

Export the collected trace data into a pandas DataFrame for further evaluation or custom analysis.

```python
trace_df = px.Client().get_spans_dataframe()
trace_df.head()
```

## Conclusion

You have successfully:

1. Built a structured data extraction service using Mistral AI's tool calling.
2. Instrumented the service with OpenInference tracing.
3. Collected and visualized telemetry data in Phoenix.
4. Exported trace data for offline analysis.

This pattern can be extended to other extraction tasks, and the observability setup allows you to monitor performance, debug issues, and improve your extraction schemas over time.