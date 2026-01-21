# Guide: Building a Stateful Echo Agent with Google ADK and Gemini

This guide walks you through building a simple "Stateful Echo Agent" using the Google Agent Development Kit (ADK). You'll create an agent that echoes user input while managing its workflow state through distinct stages (START → PROCESSING → END). This demonstrates ADK's core concepts: agents, tools, session state, and orchestration.

## Prerequisites

Ensure you have the following installed:

```bash
pip install google-adk google-genai python-dotenv
```

You will also need a **Google API Key** with access to the Gemini API. Store it securely as an environment variable.

## Step 1: Configure Your Environment

Set up your API key. This example assumes you have it stored in an environment variable named `GOOGLE_API_KEY`.

```python
import os
from dotenv import load_dotenv

# Load your API key from the environment
load_dotenv()
api_key = os.environ.get('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
```

## Step 2: Import ADK Components

Import the necessary classes from the ADK library.

```python
from google.adk.agents import Agent
from google.adk.tools import FunctionTool, ToolContext
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
```

## Step 3: Define the State Management Tool

Create a tool that allows the agent to update the workflow status in the session state.

```python
async def set_workflow_state(state_name: str, tool_context: ToolContext) -> dict:
    """Sets the current workflow state in the session state.

    Use this tool to mark progress through the workflow stages:
    - Call with 'PROCESSING' before handling the user input.
    - Call with 'END' after handling the user input.

    Args:
        state_name: The state to set (e.g., 'PROCESSING', 'END').
        tool_context: Injected context to access session state.

    Returns:
        A dictionary confirming the status update.
    """
    try:
        tool_context.state['workflow_status'] = state_name
        return {'status': 'success', 'message': f'Workflow state set to {state_name}'}
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to set state: {e}'}

# Wrap the function as an ADK tool
state_tool = FunctionTool(func=set_workflow_state)
```

## Step 4: Create the Echo Agent

Define the agent, providing it with the Gemini model, instructions, and the state tool.

```python
# Choose your Gemini model
GEMINI_MODEL_ID = "gemini-2.5-flash-lite"  # You can change this to other supported models

echo_agent = Agent(
    name="EchoAgent",
    description="An agent that echoes input while tracking workflow state.",
    model=GEMINI_MODEL_ID,
    instruction="""
    You are a simple echo agent. You also manage a workflow status stored in the session state under the key 'workflow_status'.
    The workflow states are: START, PROCESSING, END.

    Your Workflow:
    1. The workflow starts in the 'START' state (this is set externally).
    2. When you receive user input:
        a. FIRST, use the 'set_workflow_state' tool to change the status to 'PROCESSING'.
        b. THEN, simply repeat the user's exact input back to them in your response text.
        c. AFTER preparing the echo response text, use the 'set_workflow_state' tool AGAIN to change the status to 'END'.
        d. FINALLY, provide only the echo response text to the user.
    """,
    tools=[state_tool],
)
```

## Step 5: Set Up Session and Runner Services

Initialize the session service (for state management) and the runner (for orchestrating the agent interaction). Create a new session with an initial state.

```python
# Initialize services
session_service = InMemorySessionService()
runner = Runner(
    agent=echo_agent,
    session_service=session_service,
    app_name="EchoAgentDemo"
)

# Define session identifiers
APP_NAME = "EchoAgentDemo"
USER_ID = "1"
SESSION_ID = "session_01"

# Create a new session with initial state
session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state={'workflow_status': 'START'}
)
```

## Step 6: Run the Agent Interaction

Now, send a message to the agent. The runner will process the interaction, and we'll capture the events to observe the workflow.

First, prepare the user message:

```python
from google.genai.types import Content, Part

user_input_text = "Hello ADK!"
user_message = Content(role='user', parts=[Part(text=user_input_text)])
```

Next, run the agent and process the event stream:

```python
final_agent_response_text = None

async def process_interaction_events():
    """Helper async function to process and log execution events."""
    global final_agent_response_text
    event_count = 0
    async for event in runner.run_async(session_id=SESSION_ID, new_message=user_message, user_id=USER_ID):
        event_count += 1
        print(f"\n[Event {event_count}] Type: {type(event).__name__}")

        if event.content:
            part = event.content.parts[0]
            role = event.content.role
            print(f"  Role: {role}")
            if part.text:
                print(f"  Text: '{part.text}'")
                if role == 'model':
                    final_agent_response_text = part.text
            elif part.function_call:
                print(f"  >>> Function Call <<<")
                print(f"      Name: {part.function_call.name}")
                print(f"      Args: {part.function_call.args}")
            elif part.function_response:
                print(f"  <<< Function Response >>>")
                print(f"      Name: {part.function_response.name}")
                print(f"      Data: {part.function_response.response}")

# Execute the interaction
import asyncio
asyncio.run(process_interaction_events())
```

**Expected Output:**
The event log will show the agent following its instructions:
1.  A function call to `set_workflow_state` with `PROCESSING`.
2.  The tool's success response.
3.  A second function call to `set_workflow_state` with `END`.
4.  The second tool's success response.
5.  The final model response containing the echoed text: "Hello ADK!".

## Step 7: Verify the Final State and Response

Retrieve the session to confirm the workflow status was updated and capture the final agent response.

```python
# Retrieve the updated session
final_session = session_service.get_session(
    session_id=session.id,
    user_id=USER_ID,
    app_name=APP_NAME
)

if final_session:
    final_state = final_session.state
    workflow_status = final_state.get('workflow_status')
    print(f"Final workflow status: {workflow_status}")

if final_agent_response_text:
    print(f"Agent response: {final_agent_response_text}")
```

**Output:**
```
Final workflow status: END
Agent response: Hello ADK!
```

## Summary and Next Steps

You've successfully built a stateful agent using Google ADK. This simple example illustrates the foundational pattern:
1.  **Agent:** Powered by Gemini, follows instructions.
2.  **Tool:** A custom function that modifies session state.
3.  **Session State:** Maintains context (`workflow_status`) across the interaction.
4.  **Runner:** Orchestrates the entire execution flow.

To build upon this:

*   **Explore the [Official ADK Documentation](https://google.github.io/adk-docs/)** for in-depth guides on all components.
*   **Try the [Getting Started Notebook](https://colab.sandbox.google.com/github/google/adk-docs/blob/main/examples/python/tutorial/agent_team/adk_tutorial.ipynb)** for a more comprehensive tutorial.
*   **Browse the [ADK GitHub Repository](https://github.com/google/adk-python)** for advanced examples like multi-agent systems, workflow agents, and integration with other Google Cloud services.