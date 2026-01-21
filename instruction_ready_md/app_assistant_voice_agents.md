# Build a Voice Assistant with OpenAI's Agents SDK

This guide walks you through building a voice-enabled assistant for a fictitious consumer application. You will create a multi-agent system that can handle different types of user requests and then extend it to process voice input and output.

## Overview

You will build a **Triage Agent** that greets users and routes their queries to one of three specialized agents:
1.  **Search Agent**: Uses web search to provide real-time information.
2.  **Knowledge Agent**: Searches a company knowledge base to answer product questions.
3.  **Account Agent**: Retrieves user account information via a custom function.

Finally, you will convert this text-based workflow into a fully interactive voice assistant.

## Prerequisites

Ensure you have an OpenAI API key. Then, install the required packages.

```bash
pip install openai
pip install openai-agents 'openai-agents[voice]'
pip install numpy sounddevice
```

## Step 1: Import Libraries and Set API Key

Begin by importing the necessary modules and setting your OpenAI API key.

```python
from agents import Agent, function_tool, WebSearchTool, FileSearchTool, set_default_openai_key
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# Replace with your actual API key
set_default_openai_key("YOUR_API_KEY")
```

## Step 2: Define the Specialized Agents

You'll create three agents, each responsible for a specific domain.

### 2.1 Create the Search Agent

This agent uses the built-in `WebSearchTool` to find current information.

```python
search_agent = Agent(
    name="SearchAgent",
    instructions=(
        "You immediately provide an input to the WebSearchTool to find up-to-date information on the user's query."
    ),
    tools=[WebSearchTool()],
)
```

### 2.2 Create the Knowledge Agent

This agent answers questions about your product portfolio by searching a managed vector store.

First, you need to create a vector store and populate it with your documents. You can do this via the OpenAI Platform or the API. Here's an example using the API:

```python
from openai import OpenAI
import os

client = OpenAI(api_key='YOUR_API_KEY')

def create_vector_store(store_name: str) -> dict:
    """Creates a new vector store and returns its details."""
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
        }
        print("Vector store created:", details)
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}

def upload_file(file_path: str, vector_store_id: str):
    """Uploads a file to the specified vector store."""
    file_name = os.path.basename(file_path)
    try:
        file_response = client.files.create(file=open(file_path, 'rb'), purpose="assistants")
        client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )
        return {"file": file_name, "status": "success"}
    except Exception as e:
        print(f"Error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}

# Create a vector store and upload a sample document
vector_store_info = create_vector_store("ACME Shop Product Knowledge Base")
# Replace 'voice_agents_knowledge/acme_product_catalogue.pdf' with your file path
upload_file("voice_agents_knowledge/acme_product_catalogue.pdf", vector_store_info["id"])
```

Once your vector store is ready, create the Knowledge Agent. Replace `"VECTOR_STORE_ID"` with the actual ID from the previous step.

```python
knowledge_agent = Agent(
    name="KnowledgeAgent",
    instructions=(
        "You answer user questions on our product portfolio with concise, helpful responses using the FileSearchTool."
    ),
    tools=[FileSearchTool(
            max_num_results=3,
            vector_store_ids=["VECTOR_STORE_ID"], # Replace with your vector store ID
        ),],
)
```

### 2.3 Create the Account Agent

This agent uses a custom function tool to fetch user account data.

```python
# Define a dummy function tool to simulate fetching account info
@function_tool
def get_account_info(user_id: str) -> dict:
    """Return dummy account info for a given user."""
    return {
        "user_id": user_id,
        "name": "Bugs Bunny",
        "account_balance": "£72.50",
        "membership_status": "Gold Executive"
    }

# Create the Account Agent
account_agent = Agent(
    name="AccountAgent",
    instructions=(
        "You provide account information based on a user ID using the get_account_info tool."
    ),
    tools=[get_account_info],
)
```

## Step 3: Create the Triage Agent

The Triage Agent welcomes the user and routes their query to the appropriate specialized agent.

```python
triage_agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions("""
You are the virtual assistant for Acme Shop. Welcome the user and ask how you can help.
Based on the user's intent, route to:
- AccountAgent for account-related queries
- KnowledgeAgent for product FAQs
- SearchAgent for anything requiring real-time web search
"""),
    handoffs=[account_agent, knowledge_agent, search_agent],
)
```

## Step 4: Test the Text-Based Workflow

Let's verify the agent routing works correctly with some example queries.

```python
from agents import Runner, trace

async def test_queries():
    examples = [
        "What's my ACME account balance? My user ID is 1234567890", # Account Agent test
        "How big is the input and how fast is the output of the dynamite dispenser?", # Knowledge Agent test
        "What's trending in duck hunting gear right now?", # Search Agent test
    ]

    with trace("ACME App Assistant"):
        for query in examples:
            result = await Runner.run(triage_agent, query)
            print(f"User: {query}")
            print(f"Assistant: {result.final_output}")
            print("---")

# Run the tests
await test_queries()
```

You should see outputs similar to the following, confirming each query was routed correctly:
- The account query returns dummy account details.
- The product query searches the knowledge base.
- The trend query performs a web search.

## Step 5: Enable Voice Interaction

Now, convert your text-based workflow into a voice assistant. The `VoicePipeline` and `SingleAgentVoiceWorkflow` classes handle audio transcription, agent execution, and speech synthesis.

```python
import numpy as np
import sounddevice as sd
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline

async def voice_assistant():
    # Get the default microphone's sample rate
    samplerate = sd.query_devices(kind='input')['default_samplerate']

    while True:
        # Create a voice pipeline using our Triage Agent
        pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(triage_agent))

        # Check for user input to start recording or exit
        cmd = input("Press Enter to speak your query (or type 'esc' to exit): ")
        if cmd.lower() == "esc":
            print("Exiting...")
            break

        print("Listening...")
        recorded_chunks = []

        # Define a callback to capture audio data
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            recorded_chunks.append(indata.copy())

        # Start recording from the microphone
        with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', callback=audio_callback):
            input() # Recording stops when the user presses Enter again

        # Combine all recorded audio chunks
        recording = np.concatenate(recorded_chunks, axis=0)

        # Create an AudioInput object and run it through the pipeline
        audio_input = AudioInput(buffer=recording)

        with trace("ACME App Voice Assistant"):
            result = await pipeline.run(audio_input)

        # Collect the audio response chunks from the pipeline
        response_chunks = []
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                response_chunks.append(event.data)

        # Combine and play the assistant's audio response
        response_audio = np.concatenate(response_chunks, axis=0)
        print("Assistant is responding...")
        sd.play(response_audio, samplerate=samplerate)
        sd.wait()
        print("---")

# Run the voice assistant
await voice_assistant()
```

## Step 6: Optimize for Voice

The default agents are designed for text. To make voice interactions more natural, you should adapt the agent instructions.

### 6.1 Add a Voice-Optimized System Prompt

Create a common prompt that guides all agents to produce speech-friendly responses.

```python
voice_system_prompt = """
[Output Structure]
Your output will be delivered in an audio voice response, please ensure that every response meets these guidelines:
1. Use a friendly, human tone that will sound natural when spoken aloud.
2. Keep responses short and segmented—ideally one to two concise sentences per step.
3. Avoid technical jargon; use plain language so that instructions are easy to understand.
4. Provide only essential details so as not to overwhelm the listener.
"""
```

### 6.2 Create Voice-Optimized Agents

Update each agent's instructions by prepending the `voice_system_prompt`.

```python
# --- Voice-Optimized Search Agent ---
search_voice_agent = Agent(
    name="SearchVoiceAgent",
    instructions=voice_system_prompt + (
        "You immediately provide an input to the WebSearchTool to find up-to-date information on the user's query."
    ),
    tools=[WebSearchTool()],
)

# --- Voice-Optimized Knowledge Agent ---
knowledge_voice_agent = Agent(
    name="KnowledgeVoiceAgent",
    instructions=voice_system_prompt + (
        "You answer user questions on our product portfolio with concise, helpful responses using the FileSearchTool."
    ),
    tools=[FileSearchTool(
            max_num_results=3,
            vector_store_ids=["VECTOR_STORE_ID"], # Use your vector store ID
        ),],
)

# --- Voice-Optimized Account Agent ---
account_voice_agent = Agent(
    name="AccountVoiceAgent",
    instructions=voice_system_prompt + (
        "You provide account information based on a user ID using the get_account_info tool."
    ),
    tools=[get_account_info],
)

# --- Voice-Optimized Triage Agent ---
triage_voice_agent = Agent(
    name="VoiceAssistant",
    instructions=prompt_with_handoff_instructions("""
You are the virtual assistant for Acme Shop. Welcome the user and ask how you can help.
Based on the user's intent, route to:
- AccountVoiceAgent for account-related queries
- KnowledgeVoiceAgent for product FAQs
- SearchVoiceAgent for anything requiring real-time web search
"""),
    handoffs=[account_voice_agent, knowledge_voice_agent, search_voice_agent],
)
```

To use your new voice-optimized assistant, simply pass `triage_voice_agent` to the `SingleAgentVoiceWorkflow` in Step 5.

```python
# In the voice_assistant() function, update this line:
pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(triage_voice_agent))
```

## Next Steps

You have successfully built a modular, voice-enabled assistant. Here's how you can extend it further:

*   **Add More Agents**: Integrate new agents for other use cases (e.g., order status, customer support).
*   **Custom Tools**: Connect more `function_tool` decorators to your internal APIs for actions like placing orders.
*   **Evaluation**: Use the built-in [Traces dashboard](https://platform.openai.com/traces) to monitor, debug, and evaluate your agent's performance over time.
*   **Voice Customization**: Experiment with the `instructions` field in the `VoicePipeline` to adjust the personality, speed, and emotion of the synthesized voice.