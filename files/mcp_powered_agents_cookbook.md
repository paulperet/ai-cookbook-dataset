# MCP‑Powered Agentic Voice Framework

### Agents
Agents are becoming the de-facto framework in which we orchestrate various, often specialized, LLMs applications to work with one another. Many practical applications require the use of external tools to create a complex workflow for LLM-based agents.

Model Context Protocol (MCP) has quickly become the open standard for building Agentic systems. The protocol provides easy integration of common tool services and the interoperability between models across the AI ecosystem.

### What is MCP?
Model Context Protocol (MCP) is an open protocol designed to standardize how AI models - especially large language models (LLMs) - interface with external tools, data sources, and context providers in a secure, modular, and composable way. MCP provides a unified framework for sending structured requests from an agent or application to a set of “tool services,” such as databases, APIs, or custom logic modules. By adopting MCP, developers can,
* Decouple agent logic from tool implementations: Agents can call out to tools (like a database or search service) using a standard protocol, rather than relying on hardcoded integrations.
* Enforce consistent security and governance: MCP defines authentication, authorization, and data boundary controls between the model and external resources.
* Support modular, reusable agent architectures: Tools can be swapped, updated, or extended without changing the agent code, making it easy to evolve complex workflows.
* Run tools locally or remotely: The same protocol works whether a tool is running in the customer’s environment or in the cloud, supporting privacy and data residency requirements.

MCP acts as the “middleware” that bridges AI models and the external world, enabling secure, flexible, and maintainable integration of real-world context and capabilities into conversational or autonomous agents.

### Agents in the enterprise
In today’s enterprise landscape, conversational agents - especially voice-powered ones—are quickly becoming a standard for customer support, internal helpdesks, and task automation. Yet, building robust, scalable voice agents is challenging due to fragmented tooling, integration complexity, and the need for reliable orchestration of backend systems. A common pattern seen across the enterprise landscape is to develop agents that are backed by knowledge bases (both structured and unstructured). These bots are divided into several categories:
 - copilots for internal use, and 
 - customer-facing assistants. 
The latter of the two use cases, i.e. customer-facing assistants, tends to have a higher requirement for both accuracy, usability and design. Additionally, one common requirement for customer-facing chatbots is the need to add voice as a modality for user interface (i.e. for phone call automation).

These Q&A chatbots apply to a wide range of industries: healthcare, government, legal and other industries that requires a easy way for knowledge retrieval at a user's fingertips.

One such industry is the insurance industry, where we've seen tremendous value for customers we work with in the space. Insurance policies are complex and navigating the system can often be difficult for policy holders.

### What's in this Cookbook?
In this cookbook, we provide an end-to-end modular recipe leveraging MCP for building voice-enabled agents using the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/). In particular, we demonstrate how we can use it for dynamic context management and using agentic tool-calling. We demonstrate the capabilities of such a system for the aforementioned insurance use-case. In this example, we demonstrate the use of MCP for various tools that you may want for your application. Specifically, we showcase the use of custom MCP servers (for text retrieval and web search) as well as using predefined MCP servers (for SQLite). 

### End-to-end Flow

This section outlines a straightforward setup for deploying microservices for tools within the MCP framework, specifically focusing on RAG, database lookup, and web search functionalities. The MCP servers are responsible not only for hosting these services but also for performing RAG indexing to support backend operations.

We employ a "chained" approach for voice input and output throughout the system. During inference, the workflow begins by capturing a user's voice input, which is transcribed to text using a speech-to-text system. This transcribed text is then sent to the Planner agent, which determines which tools to invoke and makes requests to the appropriate microservices. After retrieving tool outputs, the Planner agent synthesizes a cohesive, contextually appropriate response. This textual response is subsequently converted to audio using a text-to-speech system, delivering the final voice response to the user.

The end-to-end workflow is summarized in the diagram below:

### Installing dependencies
First, we install the library dependencies for the project.

> Note: One specific dependency that may be needed on your machine, is to install `ffmpeg`. If you are using a mac, you will need to install this separately using `brew install ffmpeg`.

```python
#install dependencies
%pip install asyncio ffmpeg ffprobe mcp openai openai-agents pydub scipy sounddevice uv --quiet
%pip install "openai-agents[voice]" --quiet
```

[First Entry, ..., Last Entry]

### Setup

To execute this cookbook, you'll need to install the following packages providing access to OpenAI's API, the Agents SDK, MCP, and libraries for audio processing. Additionally, you can set your OpenAI API key for use by the agents via the `set_default_openai_key` function.

```python
import socket
import time
import warnings
from typing import List, Optional, AsyncGenerator

from numpy.typing import NDArray



warnings.filterwarnings("ignore", category=SyntaxWarning)


async def wait_for_server_ready(port: int = 8000, timeout: float = 10) -> None:
    """Wait for SSE server to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                print("✅ SSE server TCP port is accepting connections.")
                return
        except OSError as e:
            if time.time() - start > timeout - 1:  # Only print on last attempt
                print(f"Waiting for server... ({e})")
            time.sleep(0.5)
    raise RuntimeError("❌ SSE server did not become ready in time.")
```

### Defining Tool-use Agents through custom MCP services

First, we define a custom MCP service that host the RAG and web search tools using the `FastMCP` interface. Specifically, we add `@mcp.tool` functions for:

1.   Retrieving information from a RAG service
2.   Searching the broader internet for information using OpenAI's `web_search`


For the purpose in this cookbook, we'll run both tools under the same service.

The below code has been provided in `search_server.py` within the same directory. Run the code to start the server. As the server runs, your files will be indexed and stored in the vector store. 

You can run the `search_server.py` file by running the following command:

    ```bash
    uv run python search_server.py         
    ```

Once the server is running, you can access the vector store and files at https://platform.openai.com/storage/files and https://platform.openai.com/storage/vector_stores respectively, and continue with running the next cells in the notebook.

```python
# search_server.py
import os
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from agents import set_tracing_export_api_key

# Create server
mcp = FastMCP("Search Server")
_vector_store_id = ""

def _run_rag(query: str) -> str:
    """Do a search for answers within the knowledge base and internal documents of the user.
    Args:
        query: The user query
    """
    results = client.vector_stores.search(
        vector_store_id=_vector_store_id,
        query=query,
        rewrite_query=True,  # Query rewriting generally improves results
    )
    return results.data[0].content[0].text


def _summarize_rag_response(rag_output: str) -> str:
    """Summarize the RAG response using GPT-4
    Args:
        rag_output: The RAG response
    """
    response = client.responses.create(
        model="gpt-4.1-mini",
        tools=[{"type": "web_search_preview"}],
        input="Summarize the following text concisely: \n\n" + rag_output,
    )
    return response.output_text


@mcp.tool()
def generate_rag_output(query: str) -> str:
    """Generate a summarized RAG output for a given query.
    Args:
        query: The user query
    """
    print("[debug-server] generate_rag_output: ", query)
    rag_output = _run_rag(query)
    return _summarize_rag_response(rag_output)


@mcp.tool()
def run_web_search(query: str) -> str:
    """Run a web search for the given query.
    Args:
        query: The user query
    """
    print("[debug-server] run_web_search:", query)
    response = client.responses.create(
        model="gpt-4.1-mini",
        tools=[{"type": "web_search_preview"}],
        input=query,
    )
    return response.output_text


def index_documents(directory: str):
    """Index the documents in the given directory to the vector store
    Args:
        directory: The directory to index the documents from
    """
    # OpenAI supported file extensions for retrieval (see docs)
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx', '.pptx', '.csv', '.rtf', '.html', '.json', '.xml'}
    # Collect all files in the specified directory
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    # Filter files for supported extensions only
    supported_files = []
    for file_path in files:
        _, ext = os.path.splitext(file_path)
        if ext.lower() in SUPPORTED_EXTENSIONS:
            supported_files.append(file_path)
        else:
            print(f"[warning] Skipping unsupported file for retrieval: {file_path}")

    vector_store = client.vector_stores.create( # Create vector store
        name="Support FAQ",
    )
    global _vector_store_id
    _vector_store_id = vector_store.id

    for file_path in supported_files:
        # Upload each file to the vector store, ensuring the file handle is closed
        with open(file_path, "rb") as fp:
            client.vector_stores.files.upload_and_poll(
                vector_store_id=vector_store.id,
                file=fp
            )
        print(f"[debug-server] uploading file: {file_path}")


if __name__ == "__main__":
    oai_api_key = os.environ.get("OPENAI_API_KEY")
    if not oai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    set_tracing_export_api_key(oai_api_key)
    client = OpenAI(api_key=oai_api_key)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(current_dir, "sample_files")
    index_documents(samples_dir)

    mcp.run(transport="sse")
```

As seen above, we also include the RAG indexing as part of this workflow. In real-world applications, this will not be necessary for every run and if you have a large corpus of data, you may put this in a separate process.

In addition to simple RAG retrieval, we add an extra step to summarize the RAG output. This step is not always necessary, though we've found this to provide more succinct responses to the planner. Whether to do this depends on your system and your latency requirements.


### Using Pre-defined MCP Servers

While implementing custom MCPs servers is relatively straightforward, the power of MCP is the ability to use pre-defined servers that others have built and maintain. Using existing implementations enables more rapid development, has a consistent interface with other tools, and makes data integration more seamless. 

For our database lookup tool, we use the prebuilt [SQLite server](https://github.com/modelcontextprotocol/servers-archived/tree/main/src/sqlite) implementation. As you will see below, we can implement this simply with just a comand line prompt and providing it with a `*.db` file with the data.

### Defining the Planner Agent

Next, we can define how the MCP server will generate meaningful responses. The planner agent is a key component within MCP’s agent orchestration pipeline. Its primary function is to decompose user requests into actionable steps and decide which tools, APIs, or agents should be called at each stage. Given the input as text, the planner parses and analyzes the request, maintaining context across multiple turns. Based on the conversation state, it invokes MCP tool services by dispatching tool calls via the MCP server’s orchestration layer. The agent then collects intermediate results, synthesizes responses, and guides the conversation toward resolution.

A key design consideration is the model selection for the planner. While larger models like `4.1` offer superior reasoning, low end-to-end latency is critical in voice-driven applications. For this reason, we select the `4.1-mini` model, which achieves a strong balance between reasoning ability and response speed.

```python
from agents import Agent, trace
from agents.mcp import MCPServer, MCPServerSse, MCPServerStdio
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

voice_system_prompt = """[Voice Output Guidelines]
Your responses will be delivered via voice, so please:
1. Use conversational, natural language that sounds good when spoken
2. Keep responses concise - ideally 1-2 sentences per point
3. Avoid technical jargon unless necessary, and explain terms simply
4. Pause naturally between topics using brief sentences
5. Be warm and personable in tone
"""


async def create_insurance_agents(mcp_servers: list[MCPServer]) -> Agent:
    """Create the insurance agent workflow with voice optimization"""
    
    # Main insurance agent with MCP tools
    insurance_agent = Agent(
        name="InsuranceAssistant",
        instructions=voice_system_prompt + prompt_with_handoff_instructions("""
            #Identity
            You an a helpful chatbot that answers questions about our insurance plans. 
            #Task
            Use the tools provided to answer the questions. 
            #Instructions
            * Information about plans and policies are best answered with sqlite or rag_output tools.
            * web_search should be used for answering generic health questions that are not directly related to our insurance plans.
            * Evaluate the quality of the answer after the tool call. 
            * Assess whether you are confident in the answer generated.
            * If your confidence is low, try use another tool.
        """),
        mcp_servers=mcp_servers,
        model="gpt-4.1-mini",
    )
    
    return insurance_agent
```

[non-fatal] Tracing client error 400: {
  "error": {
    "message": "Invalid type for 'data[2].span_data.result': expected an array of strings, but got null instead.",
    "type": "invalid_request_error",
    "param": "data[2].span_data.result",
    "code": "invalid_type"
  }
}

In the agent definition, we clearly specify when each tool should be used. This ensures better control over responses and improves answer relevance. We also provide the Voice Agent with guidelines to set the desired tone and level of precision in its replies.

### Defining configurations for voice 

Next, we define the configurations for our voice module, both for speech-to-text (STT) and text-to-speech (TTS). We use the OpenAI Agent Voice library to handling both input and output of voice. As defaults, this API calls the `gpt-4o-transcribe` and `gpt-4o-mini-tts` for STT and TTS, respectively.

For more content on defining voice assistants, see [this Cookbook](https://cookbook.openai.com/examples/agents_sdk/app_assistant_voice_agents).

```python
import numpy as np
import sounddevice as sd


from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
    VoicePipelineConfig,
    TTSModelSettings
)

AudioBuffer = List[NDArray[np.int16]]

AUDIO_CONFIG = {
    "samplerate": 24000,
    "channels": 1,
    "dtype": "int16",
    "blocksize": 2400,
    "silence_threshold": 500,
    "silence_duration": 1.5,
    "min_speech_duration": 0.5,
}

insurance_tts_settings = TTSModelSettings(
    instructions=(
        "Personality: Professional, knowledgeable, and helpful insurance advisor"
        "Tone: Friendly, clear, and reassuring, making customers feel confident about their insurance choices"
        "Pronunciation: Clear and articulate, ensuring insurance terms are easily understood"
        "Tempo: Moderate pace with natural pauses, especially when explaining complex insurance concepts"
        "Emotion: Warm and supportive, conveying trust and expertise in insurance matters"
    )
)

class AudioStreamManager:
    """Context manager for handling audio streams"""
    def __init__(self, input_stream: sd.InputStream, output_stream: sd.OutputStream):
        self.input_stream = input_stream
        self.output_stream = output_stream

    async def __aenter__(self):
        try:
            self.input_stream.start()
            self.output_stream.start()
            return self
        except sd.PortAudioError as e:
            raise RuntimeError(f"Failed to start audio streams: {e}")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
            if self.output_stream:
                self.output_stream.stop()
                self.output_stream.close()
        except Exception as e:
            print(f"Warning: Error during audio stream cleanup: {e}")



```

In enterprise scenarios, the tone and style of audio responses are critical to system usability. Speech output should consistently reflect professionalism and align with the company's brand identity. For most applications, this means generating a realistic voice that mirrors the courteous, approachable demeanor typical of call-center representatives. With TTS, we can leverage prompt engineering to guide the model toward producing audio that better matches specific customer use cases and brand values.

### Processing Voice I/O

After configuring the voice settings, the next step is to implement functions for processing incoming audio and generating spoken responses. Pay particular attention to the `silence_threshold` parameter in your configuration—this plays a crucial role in accurately detecting when a user has finished speaking and helps with speech endpoint detection.

```python
import asyncio

async def continuous_voice_conversation(agent: Agent):
    """Run a continuous voice conversation with automatic speech detection"""
    
    voice_config = VoicePipeline