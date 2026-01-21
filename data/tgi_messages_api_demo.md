# Migrating from OpenAI to Open LLMs Using TGI's Messages API

_Authored by: [Andrew Reed](https://huggingface.co/andrewrreed)_

This guide demonstrates how to seamlessly transition from OpenAI models to Open LLMs without refactoring your existing code.

[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) now offers a [Messages API](https://huggingface.co/blog/tgi-messages-api) that is directly compatible with the OpenAI Chat Completion API. This means any existing scripts using OpenAI models (via the OpenAI client library or frameworks like LangChain and LlamaIndex) can be directly swapped to use any open LLM running on a TGI endpoint.

This migration allows you to leverage the advantages of open models:
* Complete control and transparency over models and data.
* No rate limits.
* Full customization for your specific needs.

In this tutorial, you will learn how to:
1. Create an Inference Endpoint to deploy a model with TGI.
2. Query the Inference Endpoint using OpenAI client libraries.
3. Integrate the endpoint with LangChain and LlamaIndex workflows.

## Prerequisites

First, install the required dependencies and set your Hugging Face API key.

```bash
pip install --upgrade huggingface_hub langchain langchain-community langchainhub langchain-openai llama-index chromadb bs4 sentence_transformers torch torchvision torchaudio llama-index-llms-openai-like llama-index-embeddings-huggingface
```

```python
import os
import getpass

# Enter your Hugging Face API key
os.environ["HF_TOKEN"] = HF_API_KEY = getpass.getpass()
```

## Step 1: Create an Inference Endpoint

You will deploy the [Nous-Hermes-2-Mixtral-8x7B-DPO](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO) model, a fine-tuned Mixtral model, to Hugging Face Inference Endpoints using TGI.

You can deploy via the UI or programmatically using the `huggingface_hub` Python library. This example uses the library, specifying the endpoint name, model repository, and task. The endpoint is set to `protected` type, requiring a valid Hugging Face token for access. You also configure hardware requirements like vendor, region, and instance type.

*Note: You may need to request a quota upgrade by emailing [api-enterprise@huggingface.co](mailto:api-enterprise@huggingface.co).*

```python
from huggingface_hub import create_inference_endpoint

endpoint = create_inference_endpoint(
    "nous-hermes-2-mixtral-8x7b-demo",
    repository="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    framework="pytorch",
    task="text-generation",
    accelerator="gpu",
    vendor="aws",
    region="us-east-1",
    type="protected",
    instance_type="p4de",
    instance_size="2xlarge",
    custom_image={
        "health_route": "/health",
        "env": {
            "MAX_INPUT_LENGTH": "4096",
            "MAX_BATCH_PREFILL_TOKENS": "4096",
            "MAX_TOTAL_TOKENS": "32000",
            "MAX_BATCH_TOTAL_TOKENS": "1024000",
            "MODEL_ID": "/repository",
        },
        "url": "ghcr.io/huggingface/text-generation-inference:sha-1734540",  # Must be >= 1.4.0
    },
)

endpoint.wait()
print(endpoint.status)
```

The deployment will take a few minutes. The `.wait()` method blocks until the endpoint reaches a final "running" state. Once running, you can test it via the UI Playground.

*Note: When deploying with `huggingface_hub`, your endpoint will scale-to-zero after 15 minutes of idle time by default to optimize costs. Check the [Hub Python Library documentation](https://huggingface.co/docs/huggingface_hub/guides/inference_endpoints) for full endpoint management functionality.*

## Step 2: Query the Inference Endpoint with OpenAI Client Libraries

Since your model is hosted with TGI and supports the Messages API, you can query it directly using the familiar OpenAI client libraries.

### Using the Python Client

The example below shows how to transition using the [OpenAI Python Library](https://github.com/openai/openai-python). Replace the base URL with your endpoint URL (including the `/v1/` suffix) and use your Hugging Face API key. The endpoint URL can be found in the Inference Endpoints UI or from the `endpoint.url` attribute.

You then use the client as usual, passing a list of messages to stream responses.

```python
from openai import OpenAI

BASE_URL = endpoint.url

# Initialize the client, pointing it to TGI
client = OpenAI(
    base_url=os.path.join(BASE_URL, "v1/"),
    api_key=HF_API_KEY,
)
chat_completion = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Why is open-source software important?"},
    ],
    stream=True,
    max_tokens=500,
)

# Iterate and print the stream
for message in chat_completion:
    print(message.choices[0].delta.content, end="")
```

Behind the scenes, TGI’s Messages API automatically converts the list of messages into the model’s required instruction format using its [chat template](https://huggingface.co/docs/transformers/chat_templating).

*Note: Certain OpenAI features, like function calling, are not compatible with TGI. Currently, the Messages API supports the following chat completion parameters: `stream`, `max_new_tokens`, `frequency_penalty`, `logprobs`, `seed`, `temperature`, and `top_p`.*

### Using the JavaScript Client

Here’s the same streaming example using the [OpenAI JavaScript/TypeScript Library](https://github.com/openai/openai-node).

```javascript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "<ENDPOINT_URL>" + "/v1/", // Replace with your endpoint URL
  apiKey: "<HF_API_TOKEN>", // Replace with your token
});

async function main() {
  const stream = await openai.chat.completions.create({
    model: "tgi",
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Why is open-source software important?" },
    ],
    stream: true,
    max_tokens: 500,
  });
  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || "");
  }
}

main();
```

## Step 3: Integrate with LangChain and LlamaIndex

Now, let's integrate your new endpoint with popular RAG frameworks like LangChain and LlamaIndex.

### Using LangChain

To use it in [LangChain](https://python.langchain.com/docs/get_started/introduction), create an instance of `ChatOpenAI` and pass your endpoint URL and Hugging Face API token.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="tgi",
    openai_api_key=HF_API_KEY,
    openai_api_base=os.path.join(BASE_URL, "v1/"),
)
llm.invoke("Why is open-source software important?")
```

You can directly leverage the same `ChatOpenAI` class used with OpenAI models. This allows all previous code to work with your endpoint by changing just one line.

Next, use your Mixtral model in a simple RAG pipeline to answer a question about a Hugging Face blog post.

```python
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load, chunk, and index the contents of the blog
loader = WebBaseLoader(
    web_paths=("https://huggingface.co/blog/open-source-llms-as-agents",),
)
docs = loader.load()

# Declare an HF embedding model
hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=hf_embeddings)

# Retrieve and generate using the relevant snippets of the blog
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

rag_chain_with_source.invoke(
    "According to this article which open-source model is the best for an agent behaviour?"
)
```

### Using LlamaIndex

Similarly, you can use a TGI endpoint in [LlamaIndex](https://www.llamaindex.ai/). Use the `OpenAILike` class and instantiate it by configuring additional arguments (`is_local`, `is_function_calling_model`, `is_chat_model`, `context_window`).

*Note: The `context_window` argument should match the value previously set for `MAX_TOTAL_TOKENS` in your endpoint configuration.*

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="tgi",
    api_key=HF_API_KEY,
    api_base=BASE_URL + "/v1/",
    is_chat_model=True,
    is_local=False,
    is_function_calling_model=False,
    context_window=4096,
)

llm.complete("Why is open-source software important?")
```

You have now successfully migrated from OpenAI to an open LLM using TGI's Messages API, integrating it with your existing workflows in LangChain and LlamaIndex.