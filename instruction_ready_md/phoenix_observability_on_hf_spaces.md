# Observability for LLM Applications: Deploying Phoenix on Hugging Face Spaces

_Authored by: [Andrew Reed](https://huggingface.co/andrewrreed)_

[Phoenix](https://docs.arize.com/phoenix) is an open-source observability library by [Arize AI](https://arize.com/) designed for experimentation, evaluation, and troubleshooting. It allows AI Engineers and Data Scientists to quickly visualize their data, evaluate performance, track down issues, and export data to improve.

This guide demonstrates how to deploy a Phoenix observability dashboard on [Hugging Face Spaces](https://huggingface.co/spaces) and configure it to automatically trace LLM calls, providing a comprehensive view into the inner workings of your LLM applications.

## Prerequisites

Before you begin, ensure you have:
- A Hugging Face account.
- An active Hugging Face API token with appropriate permissions.

## Step 1: Deploy Phoenix on Hugging Face Spaces

While Phoenix offers a notebook-first option for local development, a hosted dashboard provides a centralized, collaborative view. Hugging Face Spaces supports custom Docker images, making it an ideal platform for hosting Phoenix.

1.  **Duplicate the Demo Space:**
    Navigate to the [demo space](https://huggingface.co/spaces/andrewrreed/phoenix-arize-observability-demo?duplicate=true) and click "Duplicate this Space".

2.  **Configure the Space:**
    - Choose a name and set the visibility (Public or Private).
    - Select the default free-tier CPU.
    - **Crucially, attach a persistent disk.** This is a paid feature required for data to persist across Space restarts. Select the "Small - 20GB" option.

3.  **Build and Launch:**
    Click "Duplicate Space". The Docker image will build (this may take a few minutes). Once complete, you will see an empty Phoenix dashboard.

## Step 2: Configure Application Tracing

With the dashboard running, you can now instrument your application to automatically trace LLM calls using an OpenTelemetry TracerProvider. This example uses the OpenAI client library to call models via Hugging Face's Serverless Inference API.

### 2.1 Install Required Libraries

First, install the necessary Python packages.

```bash
pip install -q arize-phoenix arize-phoenix-otel openinference-instrumentation-openai openai huggingface-hub
```

### 2.2 Authenticate with Hugging Face

Log in to Hugging Face using your API token. Ensure the token has the necessary permissions for the organization hosting your Space.

```python
from huggingface_hub import interpreter_login

interpreter_login()
```

### 2.3 Configure the Phoenix Client

Configure the Phoenix client to connect to your running dashboard. You will need your Space's hostname, found in the "Settings" tab of your Space.

1.  Register the Phoenix tracer provider with your project name, endpoint, and authentication headers.
2.  Instrument your application code to use the OpenAI tracer provider.

```python
from phoenix.otel import register
from huggingface_hub.utils import build_hf_headers
from openinference.instrumentation.openai import OpenAIInstrumentor

# 1. Register the Phoenix tracer provider
tracer_provider = register(
    project_name="test",
    endpoint="https://YOUR-USERNAME-YOUR-SPACE-NAME.hf.space/v1/traces",  # Replace with your Space's hostname
    headers=build_hf_headers(),
)

# 2. Instrument the OpenAI client
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

## Step 3: Make LLM Calls and View Traces

Now you can make an LLM call. The trace will automatically be sent to your Phoenix dashboard.

```python
from openai import OpenAI
from huggingface_hub import get_token

# Initialize the OpenAI client for Hugging Face Inference API
client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=get_token(),
)

# Define the conversation
messages = [{"role": "user", "content": "What does a llama do for fun?"}]

# Make the API call
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=messages,
    max_tokens=500,
)

print(response.choices[0].message.content)
```

**Navigate back to your Phoenix dashboard.** You should see the trace from your LLM call captured and displayed. If you configured a persistent disk, this data will survive Space restarts.

## Step 4: Tracing a Multi-Agent Application with CrewAI

Observability becomes even more powerful for complex workflows. This example shows how to trace a multi-agent application built with [CrewAI](https://www.crewai.com/).

> **Note:** The `openinference-instrumentation-crewai` package currently requires Python 3.10 or higher. You may need to restart your kernel after installation.

### 4.1 Install CrewAI and Instrumentation

```bash
pip install -q openinference-instrumentation-crewai crewai crewai-tools
```

### 4.2 Reconfigure the Tracer Provider

You must uninstrument the previous OpenAI provider and set up a new one for CrewAI.

```python
from opentelemetry import trace
from openinference.instrumentation.crewai import CrewAIInstrumentor

# 0. Clean up the previous tracer provider
OpenAIInstrumentor().uninstrument()
if trace.get_tracer_provider():
    trace.get_tracer_provider().shutdown()
    trace._TRACER_PROVIDER = None  # Reset the global tracer provider

# 1. Register a new Phoenix tracer provider
tracer_provider = register(
    project_name="crewai",
    endpoint="https://YOUR-USERNAME-YOUR-SPACE-NAME.hf.space/v1/traces",  # Replace with your Space's hostname
    headers=build_hf_headers(),
)

# 2. Instrument the application for CrewAI
CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

### 4.3 Define and Run a Multi-Agent Crew

This example creates a simple crew with a researcher and a writer to produce a blog post.

```python
import os
from huggingface_hub import get_token
from crewai_tools import SerperDevTool
from crewai import LLM, Agent, Task, Crew, Process

# Define the LLM using Hugging Face's Inference API
llm = LLM(
    model="huggingface/meta-llama/Llama-3.1-8B-Instruct",
    api_key=get_token(),
    max_tokens=1024,
)

# Define a web search tool (requires a SERPER_API_KEY)
os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY"  # Set your API key
search_tool = SerperDevTool()

# Define the agents
researcher = Agent(
    role="Researcher",
    goal="Conduct thorough research on up to date trends around a given topic.",
    backstory="""You work at a leading tech think tank. You have a knack for dissecting complex data and presenting actionable insights.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm,
    max_iter=1,
)

writer = Agent(
    role="Technical Writer",
    goal="Craft compelling content on a given topic.",
    backstory="""You are a technical writer with a knack for crafting engaging and informative content.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=1,
)

# Create tasks for the agents
task1 = Task(
    description="""Conduct comprehensive research and analysis of the importance of observability and tracing in LLM applications.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
    expected_output="Full analysis report in bullet points",
    agent=researcher,
)

task2 = Task(
    description="""Using the insights provided, develop an engaging blog
  post that highlights the importance of observability and tracing in LLM applications.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.""",
    expected_output="Blog post of at least 3 paragraphs",
    agent=writer,
)

# Assemble the crew and run it
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=True,
    process=Process.sequential,
)

result = crew.kickoff()

print("------------ FINAL RESULT ------------")
print(result)
```

After execution, return to your Phoenix dashboard. You will now see detailed traces for the entire multi-agent workflow, including tool calls and LLM interactions, providing deep insight into the application's execution.