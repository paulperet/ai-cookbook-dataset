# Setup a Phoenix observability dashboard on Hugging Face Spaces for LLM application tracing

_Authored by: [Andrew Reed](https://huggingface.co/andrewrreed)_



[Phoenix](https://docs.arize.com/phoenix) is an open-source observability library by [Arize AI](https://arize.com/) designed for experimentation, evaluation, and troubleshooting. It allows AI Engineers and Data Scientists to quickly visualize their data, evaluate performance, track down issues, and export data to improve.

In this notebook, we'll see how to deploy a Phoenix observability dashboard on [Hugging Face Spaces](https://huggingface.co/spaces) and configure it to automatically trace LLM calls, providing a comprehensive view into the inner workings of your LLM applications.

## Step 1: Deploy Phoenix on Hugging Face Spaces

While Phoenix offers a [notebook-first option](https://docs.arize.com/phoenix/deployment/environments#notebooks) for local development, it can also be deployed as a [standalone dashboard via Docker](https://docs.arize.com/phoenix/deployment/environments#container). A long-running, hosted dashboard is a great way to provide a centralized view into your LLM application behavior, and to collaborate with your team. Hugging Face Spaces offers a simple way to host ML applications with optional, persistent storage, and it's support for custom Docker images makes it a great platform for hosting Phoenix - lets see how it works!

First, we'll [duplicate the demo space](https://huggingface.co/spaces/andrewrreed/phoenix-arize-observability-demo?duplicate=true)

We can configure the space to be private or public, and it can live in our user namespace, or in an organization namespace. We can use the default, free-tier CPU, and importantly, specify that we want to attach a persistent disk to the space.

> [!TIP] In order for the tracing data to persist across Space restarts, we _must_ configure a persistent disk, otherwise all data will be lost when the space is restarted. Configuring a persistent disk is a paid feature, and will incur a cost for the lifetime of the Space. In this case, we'll use the Small - 20GB disk option for $0.01 per hour.

After clicking "Duplicate Space", the Docker image will begin building. This will take a few minutes to complete, and then we'll see an empty Phoenix dashboard.

## Step 2: Configure application tracing

Now that we have a running Phoenix dashboard, we can configure our application to automatically trace LLM calls using an [OpenTelemetry TracerProvider](https://docs.arize.com/phoenix/quickstart#connect-your-app-to-phoenix). In this example, we'll instrument our application using the OpenAI client library, and trace LLM calls made from the `openai` Python package to open LLMs running on [Hugging Face's Serverless Inference API](https://huggingface.co/docs/api-inference/en/index).

> [!TIP] Phoenix supports tracing for [a wide variety of LLM frameworks](https://docs.arize.com/phoenix/tracing/integrations-tracing), including LangChain, LlamaIndex, AWS Bedrock, and more.


First, we need to install the necessary libraries:



```python
!pip install -q arize-phoenix arize-phoenix-otel openinference-instrumentation-openai openai huggingface-hub
```

Then, we'll login to Hugging Face using the `huggingface_hub` library. This will allow us to generate the necessary authentication for our Space and the Serverless Inference API. Be sure that the HF token used to authenticate has the correct permissions for the Organization where your Space is located.


```python
from huggingface_hub import interpreter_login

interpreter_login()
```

Now, we can [configure the Phoenix client](https://docs.arize.com/phoenix/deployment/configuration#client-configuration) to our running Phoenix dashboard:

1. Register the Phoenix tracer provider by
    - Specifying the `project_name` of our choice
    - Setting the `endpoint` value to the Hostname of our Space (found via the dashboard UI under the "Settings" tab - see below)
    - Setting the `headers` to the Hugging Face headers needed to access the Space
2. Instrument our application code to use the OpenAI tracer provider


```python
from phoenix.otel import register
from huggingface_hub.utils import build_hf_headers
from openinference.instrumentation.openai import OpenAIInstrumentor

# 1. Register the Phoenix tracer provider
tracer_provider = register(
    project_name="test",
    endpoint="https://andrewrreed-phoenix-arize-observability-demo.hf.space"
    + "/v1/traces",
    headers=build_hf_headers(),
)

# 2. Instrument our application code to use the OpenAI tracer provider
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

    🔭 OpenTelemetry Tracing Details 🔭
    |  Phoenix Project: test
    |  Span Processor: SimpleSpanProcessor
    |  Collector Endpoint: https://andrewrreed-phoenix-arize-observability-demo.hf.space/v1/traces
    |  Transport: HTTP
    |  Transport Headers: {'user-agent': '****', 'authorization': '****'}
    |  
    |  Using a default SpanProcessor. `add_span_processor` will overwrite this default.
    |  
    |  `register` has set this TracerProvider as the global OpenTelemetry default.
    |  To disable this behavior, call `register` with `set_global_tracer_provider=False`.
    


## Step 3: Make calls and view traces in the Phoenix dashboard

Now, we can make a call to an LLM and view the traces in the Phoenix dashboard. We're using the OpenAI client to make calls to the Hugging Face Serverless Inference API, which is instrumented to work with Phoenix. In this case, we're using the `meta-llama/Llama-3.1-8B-Instruct` model.


```python
from openai import OpenAI
from huggingface_hub import get_token

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=get_token(),
)

messages = [{"role": "user", "content": "What does a llama do for fun?"}]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=messages,
    max_tokens=500,
)

print(response.choices[0].message.content)
```

    Llamas are intelligent and social animals, and they do have ways to entertain themselves and have fun. While we can't directly ask a llama about its personal preferences, we can observe their natural behaviors and make some educated guesses. Here are some things that might bring a llama joy and excitement:
    
    1. **Socializing**: Llamas are herd animals and they love to interact with each other. They'll often engage in play-fighting, neck-wrestling, and other activities to establish dominance and strengthen social bonds. When llamas have a strong social network, it can make them feel happy and content.
    2. **Exploring new environments**: Llamas are naturally curious creatures, and they love to explore new surroundings. They'll often investigate their surroundings, sniffing and investigating new sights, sounds, and smells.
    3. **Playing with toys**: While llamas don't need expensive toys, they do enjoy playing with objects that stimulate their natural behaviors. For example, a ball or a toy that mimics a target can be an entertaining way to engage them.
    4. **Climbing and jumping**: Llamas are agile and athletic animals, and they enjoy using their limbs to climb and jump over obstacles. Providing a safe and stable area for them to exercise their physical abilities can be a fun and engaging experience.
    5. **Browsing and foraging**: Llamas have a natural instinct to graze and browse, and they enjoy searching for tasty plants and shrubs. Providing a variety of plants to munch on can keep them engaged and entertained.
    6. **Mentally stimulating activities**: Llamas are intelligent animals, and they can benefit from mentally stimulating activities like problem-solving puzzles or learning new behaviors (like agility training or obedience training).
    
    Some fun activities you can try with a llama include:
    
    * Setting up an obstacle course or agility challenge
    * Creating a "scavenger hunt" with treats and toys
    * Introducing new toys or objects to stimulate their curiosity
    * Providing a variety of plants and shrubs to browse and graze on
    * Engaging in interactive games like "follow the leader" or "find the treat"
    
    Remember to always prioritize the llama's safety and well-being, and to consult with a veterinarian or a trained llama handler before attempting any new activities or introducing new toys.


If we navigate back to the Phoenix dashboard, we can see the trace from our LLM call is captured and displayed! If you configured your space with a persistent disk, all of the trace information will be saved anytime you restart the space.

## Bonus: Tracing a multi-agent application with CrewAI

The real power of observability comes from being able to trace and inspect complex LLM workflows. In this example, we'll install and use [CrewAI](https://www.crewai.com/) to trace a multi-agent application.

> [!TIP] The `openinference-instrumentation-crewai` package currently requires Python 3.10 or higher. After installing the `crewai` library, you may need to restart the notebook kernel to avoid errors.


```python
!pip install -q openinference-instrumentation-crewai crewai crewai-tools
```

Like before, we'll register the Phoenix tracer provider and instrument the application code, but this time we'll also uninstrument the existing OpenAI tracer provider to avoid conflicts.


```python
from opentelemetry import trace
from openinference.instrumentation.crewai import CrewAIInstrumentor

# 0. Uninstrument existing tracer provider and clear the global tracer provider
OpenAIInstrumentor().uninstrument()
if trace.get_tracer_provider():
    trace.get_tracer_provider().shutdown()
    trace._TRACER_PROVIDER = None  # Reset the global tracer provider

# 1. Register the Phoenix tracer provider
tracer_provider = register(
    project_name="crewai",
    endpoint="https://andrewrreed-phoenix-arize-observability-demo.hf.space"
    + "/v1/traces",
    headers=build_hf_headers(),
)

# 2. Instrument our application code to use the OpenAI tracer provider
CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

    Overriding of current TracerProvider is not allowed


    🔭 OpenTelemetry Tracing Details 🔭
    |  Phoenix Project: crewai
    |  Span Processor: SimpleSpanProcessor
    |  Collector Endpoint: https://andrewrreed-phoenix-arize-observability-demo.hf.space/v1/traces
    |  Transport: HTTP
    |  Transport Headers: {'user-agent': '****', 'authorization': '****'}
    |  
    |  Using a default SpanProcessor. `add_span_processor` will overwrite this default.
    |  
    |  `register` has set this TracerProvider as the global OpenTelemetry default.
    |  To disable this behavior, call `register` with `set_global_tracer_provider=False`.
    


Now we'll define a multi-agent application using CrewAI to research and write a blog post about the importance of observability and tracing in LLM applications.

> [!TIP] This example is borrowed and modified from [here](https://docs.arize.com/phoenix/tracing/integrations-tracing/crewai).


```python
import os
from huggingface_hub import get_token
from crewai_tools import SerperDevTool
from crewai import LLM, Agent, Task, Crew, Process

# Define our LLM using HF's Serverless Inference API
llm = LLM(
    model="huggingface/meta-llama/Llama-3.1-8B-Instruct",
    api_key=get_token(),
    max_tokens=1024,
)

# Define a tool for searching the web
os.environ["SERPER_API_KEY"] = (
    "YOUR_SERPER_API_KEY"  # must set this value in your environment
)
search_tool = SerperDevTool()

# Define your agents with roles and goals
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

# Create tasks for your agents
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

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=True,
    process=Process.sequential,
)

# Get your crew to work!
result = crew.kickoff()

print("------------ FINAL RESULT ------------")
print(result)
```

    2024-10-31 09:10:49,986 - 8143739712 - __init__.py-__init__:538 - WARNING: Overriding of current TracerProvider is not allowed


    [Agent: Researcher, Task: Conduct comprehensive research and analysis of the importance of observability and tracing in LLM applications. Identify key trends, breakthrough technologies, and potential industry impacts., Agent: Researcher, Using tool: Search the internet, Tool Input: "{\"search_query\": \"importance of observability and tracing in LLM applications\"}", Tool Output: Search results: Title: LLM Observability: The 5 Key Pillars for Monitoring Large Language ..., Link: https://arize.com/blog-course/large-language-model-monitoring-observability/, Snippet: Why leveraging the five pillars of LLM observability is essential for ensuring performance, reliability, and seamless LLM applications. --- Title: Observability of LLM Applications: Exploration and Practice from the ..., Link: https://www.alibabacloud.com/blog/observability-of-llm-applications-exploration-and-practice-from-the-perspective-of-trace_601604, Snippet: This article clarifies the technical challenges of observability by analyzing LLM application patterns and different concerns. --- Title: What is LLM Observability? - The Ultimate LLM Monitoring Guide, Link: https://www.confident-ai.com/blog/what-is-llm-observability-the-ultimate-llm-monitoring-guide, Snippet: Observability tools collect and correlate logs, real-time evaluation metrics, and traces to understand the context of unexpected outputs or ... --- Title: An Introduction to Observability for LLM-based applications using ..., Link: https://opentelemetry.io/blog/2024/llm-observability/, Snippet: Why Observability Matters for LLM Applications · It's vital to keep track of how often LLMs are being used for usage and cost tracking. · Latency ... --- Title: Understanding LLM Observability - Key Insights, Best Practices ..., Link: https://signoz.io/blog/llm-observability/, Snippet: LLM Observability is essential for maintaining reliable, accurate, and efficient AI applications. Focus on the five pillars: evaluation, tracing ... --- Title: LLM Observability Tools: 2024 Comparison - lakeFS, Link: https://lakefs.io/blog/llm-observability-tools/, Snippet: LLM observability is the process that enables monitoring by providing full visibility and tracing in an LLM application system, as well as newer ... --- Title: From Concept to Production with Observability in LLM Applications, Link: https://hadijaveed.me/2024/03/05/tracing-and-observability-in-llm-applications/, Snippet: Traces are essential to understanding the full "path" a request takes in your application, e.g, prompt, query-expansion, RAG retrieved top-k ... --- Title: The Importance of LLM Observability: A Technical Deep Dive, Link: https://www.linkedin.com/pulse/importance-llm-observability-technical-deep-dive-patrick-carroll-trlqe, Snippet: LLM observability is crucial for any technical team that wants to maintain and improve the reliability, security, and performance of their AI- ... --- Title: Observability and Monitoring of LLMs - TheBlue.ai, Link: https://theblue.ai/blog/llm-observability-en/, Snippet: LLM-Observability is crucial to maximize the performance and reliability of Large Language Models (LLMs). By systematically capturing and ... ---, Final Answer: Comprehensive Research and Analysis Report: Importance of Observability and Tracing in LLM Applications Introduction Large Language Models (LLMs) have revolutionized the field of natural language processing, enabling applications such as language translation, text summarization, and conversational AI. However, as LLMs become increasingly complex and widespread, ensuring their performance, reliability, and security has become a significant challenge. Observability and tracing are crucial components in addressing this challenge, enabling developers to monitor, debug, and optimize LLM applications. This report provides an in-depth analysis of the importance of observability and tracing in LLM applications, highlighting key trends, breakthrough technologies, and potential industry impacts. Key Trends: * Increased Adoption of Cloud-Native Observability Tools: Cloud-native observability tools, such as OpenTelemetry and Signoz, are gaining popularity due to their ability to provide real-time insights into LLM application performance and behavior. * Growing Importance of Tracing: Tracing has become a critical aspect of LLM observability, enabling developers to understand the flow of requests through the application and