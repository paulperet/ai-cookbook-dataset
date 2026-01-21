# Monitoring Mistral AI with OpenTelemetry and OpenLIT

This guide provides a step-by-step tutorial for integrating OpenLIT with the Mistral AI SDK. You'll learn how to add comprehensive observability to your Mistral AI applications with just one line of code, enabling tracking of costs, tokens, prompts, responses, and all chat/completion activities using OpenTelemetry.

## Prerequisites

Before you begin, ensure you have:
- A Mistral AI API key (sign up at [Mistral AI Console](https://console.mistral.ai/))
- Python 3.8 or higher installed

## Setup

First, install the required packages:

```bash
pip install mistralai openlit langchain_community
```

Next, set your Mistral API key as an environment variable. If you don't have one yet:
1. [Sign up for a Mistral AI account](https://console.mistral.ai/)
2. [Subscribe to a free trial or billing plan](https://console.mistral.ai/billing/)
3. [Generate an API key](https://console.mistral.ai/api-keys/)

```python
import os

# Set your Mistral API key
os.environ["MISTRAL_API_KEY"] = "YOUR_MISTRAL_AI_API_KEY"
```

## Step 1: Initialize the Mistral Client

Create a synchronous client instance to interact with Mistral AI's API:

```python
from mistralai import Mistral
import os

# Initialize the Mistral client
client = Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
)
```

## Step 2: Enable OpenLIT Observability

Now, initialize OpenLIT to start collecting telemetry data. With just this single line, OpenLIT will automatically instrument all Mistral SDK calls:

```python
import openlit

# Initialize OpenLIT (disable_metrics=True disables metric collection if needed)
openlit.init(disable_metrics=True)
```

**Note:** By default, OpenLIT outputs traces to your console, which is ideal for development. For production deployment, you'll configure it to send data to an OpenTelemetry endpoint.

## Step 3: Monitor Chat Completions

Once OpenLIT is initialized, it automatically instruments all Mistral chat functions. Let's test this with a simple chat completion:

```python
# Make a chat completion request
response = client.chat.complete(
    model="mistral-small-latest",
    messages=[
        {
            "content": "Who is the best French painter? Answer in one short sentence.",
            "role": "user",
        },
    ],
)

# Print the response
print(response.choices[0].message.content)
```

OpenLIT will track:
- Input prompts and model parameters
- Output responses
- Token usage and costs
- Latency and performance metrics

## Step 4: Monitor Embeddings

OpenLIT also automatically instruments embedding functions. Let's create some embeddings:

```python
# Generate embeddings
embeddings_response = client.embeddings.create(
    inputs=[
        "Embed this sentence.",
        "As well as this one.",
    ],
    model="mistral-embed"
)

# Print the embeddings
print(f"Embedding dimensions: {len(embeddings_response.data[0].embedding)}")
```

OpenLIT tracks:
- Input text for embeddings
- Output embedding vectors
- Model parameters and costs
- Processing latency

## Step 5: Deploy OpenLIT for Production Monitoring

For production environments, you'll want to send telemetry data to a dedicated OpenLIT instance or any OpenTelemetry-compatible backend.

### Option A: Deploy OpenLIT Stack with Docker

1. Clone the OpenLIT repository:

```bash
git clone git@github.com:openlit/openlit.git
cd openlit
```

2. Start OpenLIT using Docker Compose:

```bash
docker compose up -d
```

### Option B: Configure OpenLIT for Your Backend

Configure where to send your telemetry data:

| Purpose | Parameter/Environment Variable | Example Value |
|---------|-------------------------------|---------------|
| Send to HTTP OTLP endpoint | `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `"http://127.0.0.1:4318"` |
| Add authentication headers | `otlp_headers` or `OTEL_EXPORTER_OTLP_HEADERS` | `"api-key=your-key"` |

Update your initialization to send data to OpenLIT:

```python
import openlit

# Send telemetry data to your OpenLIT instance
openlit.init(
    otlp_endpoint="http://127.0.0.1:4318",
    disable_metrics=False  # Enable metrics for production
)
```

## Step 6: Visualize and Analyze Your Data

With OpenLIT collecting data, access the dashboard to gain insights:

1. Open your browser and navigate to `http://127.0.0.1:3000`
2. Log in with the default credentials:
   - **Email**: `user@openlit.io`
   - **Password**: `openlituser`

In the OpenLIT dashboard, you can:
- Monitor LLM costs and usage patterns
- Analyze response times and performance
- Track token consumption across models
- Identify optimization opportunities
- Debug issues with detailed trace information

## Next Steps

- Explore OpenLIT's advanced features for prompt management and versioning
- Set up alerts for cost thresholds or performance degradation
- Integrate with other OpenTelemetry backends like Grafana, Jaeger, or DataDog
- Monitor additional AI components (vector databases, GPU usage, etc.)

By following this guide, you've successfully added comprehensive observability to your Mistral AI applications, enabling you to build, monitor, and optimize with confidence from development through production.