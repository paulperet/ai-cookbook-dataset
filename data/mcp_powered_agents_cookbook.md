# Building a Voice-Enabled Insurance Assistant with MCP

## Overview

This guide demonstrates how to build a voice-enabled insurance assistant using the Model Context Protocol (MCP) and OpenAI's Agents SDK. You'll create an agent that can answer insurance-related questions by retrieving information from knowledge bases, databases, and web searches, all delivered through natural voice interactions.

### What You'll Build
- A custom MCP server for RAG and web search functionality
- Integration with a pre-built SQLite MCP server for database queries
- A planner agent that orchestrates tool usage
- Voice input/output pipeline with speech-to-text and text-to-speech

### Prerequisites
- Python 3.8+
- OpenAI API key
- Basic understanding of async Python

## Setup

First, install the required dependencies:

```bash
pip install asyncio ffmpeg ffprobe mcp openai openai-agents pydub scipy sounddevice uv
pip install "openai-agents[voice]"
```

> **Note:** On macOS, you may need to install `ffmpeg` separately: `brew install ffmpeg`

Set up your environment by importing necessary modules:

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
            if time.time() - start > timeout - 1:
                print(f"Waiting for server... ({e})")
            time.sleep(0.5)
    raise RuntimeError("❌ SSE server did not become ready in time.")
```

## Step 1: Create a Custom MCP Server for Search and RAG

We'll create a custom MCP server that provides two key tools:
1. **RAG retrieval** from insurance documents
2. **Web search** for general information

Create a file called `search_server.py`:

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
    """Search for answers within the knowledge base and internal documents."""
    results = client.vector_stores.search(
        vector_store_id=_vector_store_id,
        query=query,
        rewrite_query=True,  # Query rewriting generally improves results
    )
    return results.data[0].content[0].text

def _summarize_rag_response(rag_output: str) -> str:
    """Summarize the RAG response using GPT-4."""
    response = client.responses.create(
        model="gpt-4.1-mini",
        tools=[{"type": "web_search_preview"}],
        input="Summarize the following text concisely: \n\n" + rag_output,
    )
    return response.output_text

@mcp.tool()
def generate_rag_output(query: str) -> str:
    """Generate a summarized RAG output for a given query."""
    print("[debug-server] generate_rag_output: ", query)
    rag_output = _run_rag(query)
    return _summarize_rag_response(rag_output)

@mcp.tool()
def run_web_search(query: str) -> str:
    """Run a web search for the given query."""
    print("[debug-server] run_web_search:", query)
    response = client.responses.create(
        model="gpt-4.1-mini",
        tools=[{"type": "web_search_preview"}],
        input=query,
    )
    return response.output_text

def index_documents(directory: str):
    """Index documents from the given directory to the vector store."""
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.docx', '.pptx', '.csv', '.rtf', '.html', '.json', '.xml'}
    
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    supported_files = []
    
    for file_path in files:
        _, ext = os.path.splitext(file_path)
        if ext.lower() in SUPPORTED_EXTENSIONS:
            supported_files.append(file_path)
        else:
            print(f"[warning] Skipping unsupported file for retrieval: {file_path}")

    vector_store = client.vector_stores.create(name="Support FAQ")
    global _vector_store_id
    _vector_store_id = vector_store.id

    for file_path in supported_files:
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

Start the server:

```bash
uv run python search_server.py
```

> **Note:** The server will index documents from a `sample_files` directory. In production, you might separate indexing from the server runtime for better performance.

## Step 2: Use a Pre-built SQLite MCP Server

MCP's power lies in its ecosystem of pre-built servers. For database operations, we'll use the official SQLite server:

```bash
# Clone and run the SQLite server
git clone https://github.com/modelcontextprotocol/servers-archived.git
cd servers-archived/src/sqlite
npx @modelcontextprotocol/server-sqlite insurance.db
```

This provides a database lookup tool without writing custom code, demonstrating MCP's interoperability benefits.

## Step 3: Define the Planner Agent

The planner agent orchestrates tool usage based on user queries. We'll use `gpt-4.1-mini` for optimal balance between reasoning and latency:

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
    
    insurance_agent = Agent(
        name="InsuranceAssistant",
        instructions=voice_system_prompt + prompt_with_handoff_instructions("""
            #Identity
            You are a helpful chatbot that answers questions about our insurance plans. 
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

The agent includes clear guidelines for tool selection and voice-optimized response formatting.

## Step 4: Configure Voice Settings

Define audio configurations and text-to-speech settings for natural voice interactions:

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

The `silence_threshold` parameter is crucial for accurate speech endpoint detection.

## Step 5: Process Voice Input/Output

Implement the continuous voice conversation handler:

```python
import asyncio

async def continuous_voice_conversation(agent: Agent):
    """Run a continuous voice conversation with automatic speech detection"""
    
    voice_config = VoicePipelineConfig(
        stt_model="gpt-4o-transcribe",
        tts_model="gpt-4o-mini-tts",
        tts_settings=insurance_tts_settings,
        **AUDIO_CONFIG
    )
    
    # Create voice pipeline
    voice_pipeline = VoicePipeline(config=voice_config)
    
    # Set up audio streams
    input_stream = sd.InputStream(
        samplerate=voice_config.samplerate,
        channels=voice_config.channels,
        dtype=voice_config.dtype,
        blocksize=voice_config.blocksize
    )
    
    output_stream = sd.OutputStream(
        samplerate=voice_config.samplerate,
        channels=voice_config.channels,
        dtype=voice_config.dtype
    )
    
    async with AudioStreamManager(input_stream, output_stream) as streams:
        print("🎤 Listening... Speak your insurance question.")
        
        while True:
            try:
                # Capture audio until silence is detected
                audio_data = await voice_pipeline.capture_audio(
                    input_stream,
                    silence_threshold=voice_config.silence_threshold,
                    silence_duration=voice_config.silence_duration
                )
                
                if len(audio_data) == 0:
                    continue
                
                # Transcribe speech to text
                transcription = await voice_pipeline.transcribe(audio_data)
                print(f"👤 User: {transcription}")
                
                # Get agent response
                response = await agent.run(transcription)
                print(f"🤖 Assistant: {response}")
                
                # Convert response to speech
                audio_output = await voice_pipeline.synthesize(response)
                
                # Play the audio response
                output_stream.write(audio_output)
                
            except KeyboardInterrupt:
                print("\n👋 Ending conversation.")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
```

## Step 6: Run the Complete System

Now, let's connect all components and start the voice assistant:

```python
async def main():
    # Connect to MCP servers
    search_server = MCPServerSse(url="http://localhost:8000/sse")
    
    # Wait for servers to be ready
    await wait_for_server_ready(port=8000)
    
    # Create agent with MCP tools
    insurance_agent = await create_insurance_agents([search_server])
    
    # Start voice conversation
    await continuous_voice_conversation(insurance_agent)

if __name__ == "__main__":
    asyncio.run(main())
```

## How It Works

### End-to-End Flow
1. **Voice Input**: User speaks an insurance question
2. **Speech-to-Text**: Audio is transcribed to text using `gpt-4o-transcribe`
3. **Planner Agent**: Analyzes the query and selects appropriate tools:
   - `generate_rag_output`: For insurance policy details from documents
   - SQLite tool: For structured data queries
   - `run_web_search`: For general health information
4. **Tool Execution**: MCP servers process the requests
5. **Response Synthesis**: Agent combines tool outputs into a coherent answer
6. **Text-to-Speech**: Response is converted to audio using `gpt-4o-mini-tts`
7. **Voice Output**: User hears the spoken response

### Tool Selection Logic
- **Insurance-specific questions**: Use RAG or SQLite tools
- **General health questions**: Use web search
- **Low confidence responses**: Try alternative tools

## Best Practices

### Performance Optimization
- Use `gpt-4.1-mini` for the planner agent to balance reasoning and latency
- Implement RAG summarization to reduce response length
- Adjust `silence_threshold` based on your environment's noise levels

### Voice Quality
- Provide detailed TTS instructions for brand-appropriate voice tone
- Keep responses concise for better audio delivery
- Test with real users to refine voice personality settings

### MCP Server Management
- Separate indexing from server runtime in production
- Use pre-built MCP servers when available to accelerate development
- Monitor server health and implement retry logic

## Next Steps

1. **Add more tools**: Integrate additional MCP servers for calendar, email, or CRM systems
2. **Implement fallback strategies**: Add logic for when tools fail or return low-confidence results
3. **Add multi-language support**: Configure the voice pipeline for different languages
4. **Implement session management**: Track conversation history across multiple turns
5. **Add authentication**: Secure access to sensitive insurance data

This framework provides a modular, extensible foundation for building voice-enabled agents that can securely interact with various data sources while maintaining natural, professional conversations.