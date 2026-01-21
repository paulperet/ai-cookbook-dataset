# Generate a Company Report Using Gemini's Search Tool

This guide demonstrates how to use the Gemini API's new search tool to generate an up-to-date company research report. The search tool allows the model to retrieve current information from the web, making it ideal for tasks requiring recent data.

## Prerequisites

This tutorial requires a paid-tier Gemini API key, as the search tool is not available on the free tier. Ensure your key is stored securely.

## Setup

### 1. Install the SDK

Begin by installing the latest Google Gen AI SDK.

```bash
pip install -U -q google-genai
```

### 2. Import Libraries

Import the necessary modules for the client, configuration, and display.

```python
from google import genai
from google.genai.types import GenerateContentConfig, Tool
from IPython.display import display, HTML, Markdown
import io
import json
import re
```

### 3. Initialize the Client

Create a client instance using your API key. The model will be specified in the generation call.

```python
# Replace with your method of securely loading the API key
GOOGLE_API_KEY = "YOUR_API_KEY"
client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 4. Select a Model

Choose a Gemini 2.0 or later model that supports the search tool feature.

```python
MODEL_ID = "gemini-2.5-flash"  # Example model
```

## Generate the Company Report

### 5. Define the Target Company

Set the company you want to research. We'll use Alphabet as an example.

```python
COMPANY = 'Alphabet'
```

### 6. Configure the System Instruction

Define a system instruction that guides the model to act as an analyst, plan its research, use search, and format the final report.

```python
sys_instruction = """You are an analyst that conducts company research.
You are given a company name, and you will work on a company report. You have access
to Google Search to look up company news, updates and metrics to write research reports.

When given a company name, identify key aspects to research, look up that information
and then write a concise company report.

Feel free to plan your work and talk about it, but when you start writing the report,
put a line of dashes (---) to demarcate the report itself, and say nothing else after
the report has finished.
"""
```

### 7. Generate the Report with Streaming

Configure the generation to use the search tool and stream the response. The model will plan, search, and write in a single process.

**Important:** You must enable Google Search Suggestions to comply with the API's [display requirements](https://ai.google.dev/gemini-api/docs/grounding/search-suggestions#display-requirements).

```python
config = GenerateContentConfig(
    system_instruction=sys_instruction,
    tools=[Tool(google_search={})],
    temperature=0
)

response_stream = client.models.generate_content_stream(
    model=MODEL_ID,
    config=config,
    contents=[COMPANY]
)

report = io.StringIO()
for chunk in response_stream:
    candidate = chunk.candidates[0]

    for part in candidate.content.parts:
        if part.text:
            # Display the model's reasoning and report in real-time
            display(Markdown(part.text))

            # Extract the final report text after the '---' demarcation
            if m := re.search(r'(^|\n)-+\n(.*)$', part.text, re.M):
                report.write(m.group(2))
            elif report.tell():
                report.write(part.text)
        else:
            # Print any non-text parts (e.g., tool calls) for debugging
            print(json.dumps(part.model_dump(exclude_none=True), indent=2))

    # Display the required search suggestions
    if gm := candidate.grounding_metadata:
        if sep := gm.search_entry_point:
            display(HTML(sep.rendered_content))
```

When executed, the model will output its plan, perform searches, and stream the final report. The output will look similar to this:

> Okay, I will research Alphabet and create a company report. Here's my plan...
> (Model performs searches)
> ---
> **Company Report: Alphabet Inc.**
> **Overview:**
> Alphabet Inc. is a multinational technology conglomerate...
> (Complete report follows)

### 8. Display the Final Report

After the stream completes, you can render the cleaned final report.

```python
# Display the extracted report, escaping '$' for proper Markdown rendering
display(Markdown(report.getvalue().replace('$', r'\$')))
```

## Summary

You have successfully used Gemini's search tool to generate a detailed, current company report. The model autonomously planned the research, retrieved the latest information via web search, and structured its findings into a coherent document.

This tutorial highlights the power of the integrated search tool for tasks requiring fresh, factual data. For production use, consider adding more robust error handling, user input validation, and potentially breaking the process into distinct planning and writing phases for greater control.