# Monitoring LlamaIndex + Mistral Applications with PostHog and Langfuse

This guide walks you through building a Retrieval-Augmented Generation (RAG) application using LlamaIndex and Mistral, instrumenting it with Langfuse for observability, and analyzing the resulting data in PostHog for product analytics.

## Prerequisites

Before you begin, ensure you have:
1.  A [Mistral AI account](https://console.mistral.ai/) with an API key.
2.  A [Langfuse account](https://cloud.langfuse.com/auth/sign-up) with project API keys.
3.  (Optional) A [PostHog account](https://us.posthog.com/) for analytics.

## Step 1: Install Required Libraries

First, install the necessary Python packages.

```bash
pip install llama-index llama-index-llms-mistralai llama-index-embeddings-mistralai langfuse nest_asyncio --upgrade
```

## Step 2: Configure the Mistral AI LLM and Embedding Model

Set your Mistral API key and configure LlamaIndex to use Mistral's language and embedding models.

```python
import os
import nest_asyncio

# Apply nest_asyncio to allow sync/async code to work together seamlessly
nest_asyncio.apply()

# Set your Mistral API Key
os.environ["MISTRAL_API_KEY"] = "your-mistral-api-key-here"

# Import LlamaIndex components
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings

# Initialize the LLM and Embedding model
llm = MistralAI(model="open-mixtral-8x22b", temperature=0.1)
embed_model = MistralAIEmbedding(model_name="mistral-embed")

# Set these as the default models in the global Settings
Settings.llm = llm
Settings.embed_model = embed_model
```

## Step 3: Initialize Langfuse for Tracing

Initialize the Langfuse client and integrate its callback handler with LlamaIndex. This ensures all LLM calls and retrieval steps are automatically traced.

```python
from langfuse import Langfuse
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler

# Initialize the Langfuse client with your project credentials
langfuse = Langfuse(
    secret_key="sk-lf-...",      # Your Langfuse Secret Key
    public_key="pk-lf-...",      # Your Langfuse Public Key
    host="https://cloud.langfuse.com"  # Use "https://us.cloud.langfuse.com" for US region
)

# Create a Langfuse callback handler and register it with LlamaIndex
langfuse_callback_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([langfuse_callback_handler])
```

## Step 4: Load Your Document Data

For this example, we'll use a PDF guide on hedgehog care. Download the file and load it using LlamaIndex's document reader.

```python
# Download the example PDF
import requests

url = "https://www.pro-igel.de/downloads/merkblaetter_engl/wildtier_engl.pdf"
response = requests.get(url)
with open('./hedgehog.pdf', 'wb') as f:
    f.write(response.content)

# Load the document using LlamaIndex
from llama_index.core import SimpleDirectoryReader

hedgehog_docs = SimpleDirectoryReader(input_files=["./hedgehog.pdf"]).load_data()
```

## Step 5: Build the RAG Query Engine

Create a vector index from the loaded documents and instantiate a query engine for retrieval.

```python
from llama_index.core import VectorStoreIndex

# Create a vector index from the documents
hedgehog_index = VectorStoreIndex.from_documents(hedgehog_docs)

# Create a query engine from the index
hedgehog_query_engine = hedgehog_index.as_query_engine(similarity_top_k=5)
```

## Step 6: Run a Query and View the Trace

Now you can query the engine. The entire process—retrieval, LLM call, and generation—will be logged as a trace in your Langfuse project.

```python
# Query the engine
response = hedgehog_query_engine.query("Which hedgehogs require help?")
print(response)
```

**Expected Output:**
```
Hedgehogs that require help are those that are sick, injured, and helpless, such as orphaned hoglets. These hedgehogs in need may be temporarily taken into human care and must be released into the wild as soon as they can survive there independently.
```

After running this, navigate to your [Langfuse dashboard](https://cloud.langfuse.com/) to inspect the detailed trace of this query.

## Step 7: (Optional) Capture User Feedback with Langfuse Scores

To monitor application performance, you can capture explicit user feedback (e.g., thumbs up/down) as scores in Langfuse. These scores can later be analyzed alongside product metrics in PostHog.

Use the `@observe` decorator to automatically wrap your query function in a trace and then attach a score to it.

```python
from langfuse.decorators import langfuse_context, observe

@observe()
def hedgehog_helper(user_message):
    """A helper function to query the engine and return the trace ID."""
    response = hedgehog_query_engine.query(user_message)
    trace_id = langfuse_context.get_current_trace_id()
    print(response)
    return trace_id

# Run the helper function
trace_id = hedgehog_helper("Can I keep the hedgehog as a pet?")

# Attach a user feedback score to the generated trace
langfuse.score(
    trace_id=trace_id,
    name="user-explicit-feedback",
    value=0.9,  # A numeric score, e.g., 0.9 for positive feedback
    comment="Good to know!"
)
```

**Expected Output:**
```
Based on the provided context, it is not recommended to keep wild hedgehogs as pets...
```

## Step 8: Analyze Data in PostHog

To combine LLM observability with product analytics, connect Langfuse to PostHog.

1.  **Sign up** for a [free PostHog account](https://us.posthog.com/) if you don't have one.
2.  In your PostHog project settings, copy your **Project API Key** and **Host** (e.g., `https://us.posthog.com`).
3.  In your Langfuse project dashboard, go to **Settings > Integrations**.
4.  Find the PostHog integration, click **Configure**, and paste your PostHog Host and Project API Key.
5.  Enable the integration and click **Save**.

Langfuse will begin exporting your trace and score data to PostHog daily.

### Using the Pre-built Dashboard Template

PostHog provides a dashboard template specifically for Langfuse data, making it easy to visualize key LLM metrics.

1.  In PostHog, navigate to the **Dashboards** tab.
2.  Click **New dashboard** in the top right.
3.  Select the **LLM metrics – Langfuse** template from the list.

This dashboard will automatically populate with charts showing model costs, user feedback scores, latency, and other vital metrics for your application.

---

## Next Steps

*   Explore the [Langfuse documentation](https://langfuse.com/docs) to learn more about traces, evaluations, and prompt management.
*   Check the [PostHog docs](https://posthog.com/docs/ai-engineering/langfuse-posthog) for advanced analysis techniques.
*   Provide feedback or ask questions via the [Langfuse GitHub](https://langfuse.com/issue) or [Discord community](https://discord.langfuse.com/).