# Structured Data Extraction Service Observability with Mistral AI and Phoenix

In this tutorial, you will:

- Use Mistral's [tool calling feature](https://docs.mistral.ai/guides/function-calling/) to perform structured data extraction: the task of transforming unstructured input (e.g., user requests in natural language) into structured format (e.g., tabular format),
- Instrument your Mistral AI client to record trace data in [OpenInference tracing](https://github.com/Arize-ai/openinference) format,
- Inspect the traces and spans of your application to visualize your trace data,

## Background

One powerful feature of the Mistral AI chat completions API is tool calling, wherein a user describes the signature and arguments of one or more tools to the Mistral AI API via a JSON Schema and natural language descriptions, and the LLM decides when to use each tool and provides argument values depending on the context of the conversation. In addition to its primary purpose of integrating function inputs and outputs into a sequence of chat messages, tool calling is also useful for structured data extraction, since you can specify a "function" that describes the desired format of your structured output. Structured data extraction is useful for a variety of purposes, including ETL or as input to another machine learning model such as a recommender system.

While it's possible to produce structured output without using tool calling via careful prompting, tool calling is more reliable at producing output that conforms to a particular format. For more details on Mistral AI's tool calling API, see the [Mistral AI documentation](https://docs.mistral.ai/guides/function-calling/).

Let's get started!

ℹ️ This notebook requires a Mistral AI API key.

## 1. Install Dependencies and Import Libraries

Install dependencies.

```python
!pip install -q arize-phoenix jsonschema openinference-instrumentation-mistralai
!pip install -qU mistralai 
```

Import libraries.

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

## 2. Configure Your Mistral API Key

Set your Mistral API key if it is not already set as an environment variable.

```python
if not (api_key := os.getenv("MISTRAL_API_KEY")):
    api_key = getpass("🔑 Enter your Mistral AI API key: ")
client = Mistral(api_key=api_key)
```

## 3. Run Phoenix in the Background

Launch Phoenix as a background session to collect the trace data emitted by your instrumented Mistral client. For details on how to self-host Phoenix or connect to a remote Phoenix instance, see the [Phoenix documentation](https://docs.arize.com/phoenix/quickstart).

```python
session = px.launch_app()
tracer_provider = register()
```

## 4. Instrument Your Mistral Client

Instrument your Mistral client with a tracer that emits telemetry data in OpenInference format. [OpenInference](https://github.com/Arize-ai/openinference) is an open standard for capturing and storing LLM application traces that enables LLM applications to seamlessly integrate with LLM observability solutions such as Phoenix.

```python
MistralAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

## 5. Extract Your Structured Data

We'll extract structured data from the following list of ten travel requests.

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
```

The Mistral AI API uses [JSON Schema](https://json-schema.org/) and natural language descriptions to specify the signature of a function to call. In this case, we'll describe a function to record the following attributes of the unstructured text input:

- **location:** The desired destination,
- **budget_level:** A categorical budget preference,
- **purpose:** The purpose of the trip.

The use of JSON Schema enables us to define the type of each field in the output and even enumerate valid values in the case of categorical outputs. Mistral AI tool calling can thus be used for tasks that might previously have been performed by named-entity recognition (NER) and/ or classification models.

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
        # By default, the LLM will choose whether or not to use a tool given the conversation context.
        # The line below forces the LLM to use a tool so that the output conforms to the schema.
        # tool_choice=ToolChoice.any,
    )
    return chat_completion.choices[0].message.tool_calls[0].function.arguments
```

Run the extractions.

```python
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

[Travel request: ... Raw Travel Attributes: ..., ..., Travel request: ... Raw Travel Attributes: ...]

## 6. View traces in Phoenix

You should now be able to view traces in Phoenix.

```python
print(f"Current Phoenix URL: {session.url}")
```

## 7. Export Your Trace Data

Your OpenInference trace data is collected by Phoenix and can be exported to a pandas dataframe for further analysis and evaluation.

```python
trace_df = px.Client().get_spans_dataframe()
trace_df
```

## 8. Recap

Congrats! In this tutorial, you:

- Built a service to perform structured data extraction on unstructured text using Mistral AI tool calling
- Instrumented your service with an OpenInference tracer
- Examined your telemetry data in Phoenix