# Monitoring Mistral AI with OpenTelemetry

This cookbook will cover the process of integrating OpenLIT with the Mistral SDK. A straightforward guide demonstrates how adding a single line of code can seamlessly enable OpenLIT to track various metrics, including cost, tokens, prompts, responses, and all chat/completion activities from the Mistral SDK using OpenTelemetry.

## About OpenLIT

**OpenLIT** is an open-source AI Engineering tool that help you to simplify your AI development workflow, especially for Generative AI and LLMs. It streamlines essential tasks like experimenting with LLMs, organizing and versioning prompts, and securely handling API keys. With just one line of code, you can enable **OpenTelemetry-native** observability, offering full-stack monitoring that includes LLMs, vector databases, and GPUs. This enables developers to confidently build AI features and applications, transitioning smoothly from testing to production.

This project proudly follows and maintains the [Semantic Conventions](https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai) with the OpenTelemetry community, consistently updating to align with the latest standards in Observability.

```python
%pip install mistralai openlit langchain_community
```

[First Entry, ..., Last Entry]

Set your Mistral API key as an environment variable. If you haven't already, [sign up for a Mistral acccount](https://console.mistral.ai/). Then [subscribe](https://console.mistral.ai/billing/) to a free trial or billing plan, after which you'll be able to [generate an API key](https://console.mistral.ai/api-keys/).

```python
import os

# Your Mistral key
os.environ["MISTRAL_API_KEY"] = "YOUR_MISTRAL_AI_API_KEY"
```

```python
# Synchronous Example
from mistralai import Mistral
import os

client = Mistral(
    api_key=os.getenv("MISTRAL_API_KEY", ""),
)
```

```python
import openlit

openlit.init(disable_metrics=True)
```

## Chat Completions

Once OpenLIT is initialized in the application, It auto-instruments all Mistral Chat function usage from the SDK. This helps track LLM interactions, capturing inputs, outputs, model parameters along with cost.

```python
res = client.chat.complete(model="mistral-small-latest", messages=[
    {
        "content": "Who is the best French painter? Answer in one short sentence.",
        "role": "user",
    },
])
```

[First Entry, ..., Last Entry]

## Embeddings

Once OpenLIT is initialized in the application, It auto-instruments all Mistral embedding function usage from the SDK. This helps track embedding inputs, outputs, model parameters along with cost.

```python
res = client.embeddings.create(inputs=[
    "Embed this sentence.",
    "As well as this one.",
], model="mistral-embed")
```

[First Entry, ..., Last Entry]

# Sending Traces and metrics to OpenLIT

By default, OpenLIT generates OpenTelemetry traces and metrics that are logged to your console. To set up a detailed monitoring environment, this guide outlines how to deploy OpenLIT and direct all traces and metrics there. You also have the flexibility to send the telemetry data to any OpenTelemetry-compatible endpoint, such as Grafana, Jaeger, or DataDog.

## Deploy OpenLIT Stack

1. Clone the OpenLIT Repository

   Open your terminal or command line and execute:

   ```shell
   git clone git@github.com:openlit/openlit.git
   ```

2. Host it Yourself with Docker

   Deploy and start OpenLIT using the command:

   ```shell
   docker compose up -d
   ```

> For instructions on installing in Kubernetes using Helm, refer to the [Kubernetes Helm installation guide](https://docs.openlit.io/latest/installation#kubernetes).

Configure the telemetry data destination as follows:

| Purpose                                   | Parameter/Environment Variable                   | For Sending to OpenLIT         |
|-------------------------------------------|--------------------------------------------------|--------------------------------|
| Send data to an HTTP OTLP endpoint        | `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` | `"http://127.0.0.1:4318"`      |
| Authenticate telemetry backends           | `otlp_headers` or `OTEL_EXPORTER_OTLP_HEADERS`   | Not required by default        |

> 💡 Info: If the `otlp_endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` is not provided, the OpenLIT SDK will output traces directly to your console, which is recommended during the development phase.

## Visualize and Optimize!

With the Observability data now being collected and sent to OpenLIT, the next step is to visualize and analyze this data to get insights into your AI application's performance, behavior, and identify areas of improvement.

Just head over to OpenLIT at `127.0.0.1:3000` on your browser to start exploring. You can login using the default credentials
  - **Email**: `user@openlit.io`
  - **Password**: `openlituser`