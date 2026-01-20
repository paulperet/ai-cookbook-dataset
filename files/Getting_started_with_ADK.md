# ADK Simple Demo: Stateful Echo Agent with Gemini


This notebook provides a basic, introductory example of using `Gemini` in the Google Agent Development Kit (ADK).

**Goal:** Demonstrate how ADK orchestrates a simple workflow involving state transitions (`START` -> `PROCESSING` -> `END`) around a core interaction with the Gemini API.

**Scenario:**
You will build a "Stateful Echo Agent". This agent's primary task is to echo the user's input. However, it will use ADK components to manage its internal state throughout the process:
1.  It starts in a `START` state.
2.  Upon receiving input, it uses an ADK Tool to transition to `PROCESSING`.
3.  It prepares the echo response (implicitly using the Gemini model configured in the Agent).
4.  It uses the ADK Tool again to transition to the `END` state.
5.  It delivers the final echo response.

This example highlights ADK's role in managing structured workflows and state, even for simple tasks.

This notebook was contributed by Anand Roy.

LinkedIn - See Anand other notebooks here.

Have a cool Gemini example? Feel free to share it too!

## Setup


```
%pip install -q google-adk google-genai python-dotenv
```

## 1. Configure Google API Key

To power the `Agent` with Gemini, access to the Google Generative AI API is required. The next code cell configures your API key.

**Important:** This example uses Colab Secrets (`userdata.get('GOOGLE_API_KEY')`). Make sure you have stored your key named `GOOGLE_API_KEY` in the Colab Secrets manager (View -> Secrets).


```
from google.colab import userdata
import os
from dotenv import load_dotenv

api_key = userdata.get('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = api_key
load_dotenv()
```




    False



## 2. Core ADK Components in this Demo

This example uses the following key ADK components:

*   **`Agent`**: The agent powered by the Gemini model. It understands instructions, decides when to use tools, and generates responses.
*   **`FunctionTool`**: A custom capability provided to the agent. In this case, it's a tool to update the workflow status.
*   **`ToolContext`**: An object automatically passed to our tool, allowing it to access and modify the `Session State`.
*   **`SessionService` (`InMemorySessionService`)**: Manages the conversation's state (`workflow_status`). `InMemory` means the state exists only while this script runs.
*   **`Runner`**: Orchestrates the entire interaction: passes user input to the agent, handles tool calls, manages the state via the `SessionService`, and delivers the final response.
*   **`Session State`**: A dictionary holding data for the current conversation (session). Here, you use it to store `{'workflow_status': '...'}`.


```
from google.adk.agents import Agent
from google.adk.tools import FunctionTool, ToolContext
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.sessions import Session
```

## 3. Define ADK Components (Tool, Agent, Services)

Now, let's define the core ADK components for our Stateful Echo Agent:

1.  **Tool (`set_workflow_state`):** A Python function wrapped as an ADK `FunctionTool`. This function will modify the `workflow_status` in the session state when called by the agent.
2.  **Agent (`echo_agent`):** An `LlmAgent` configured with the Gemini model, specific instructions on *when* to call the `state_tool`, and the tool itself.
3.  **Services (`session_service`, `runner`):** The `InMemorySessionService` to hold state and the `Runner` to execute the agent.
4.  **Session:** Used to create a specific session instance with an initial state `{'workflow_status': 'START'}`.


```
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

# Create the function tool
state_tool = FunctionTool(func=set_workflow_state)
```


```
GEMINI_MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}

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


```
session_service = InMemorySessionService()
runner = Runner(
    agent=echo_agent,
    session_service=session_service,
    app_name="EchoAgentDemo"
)

APP_NAME="EchoAgentDemo"
USER_ID="1"
ID="session_01"

session = session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=ID,
    state={'workflow_status': 'START'}
)
```

## 4. Run the Interaction

Now send a simple message ("Hello ADK!") to the `echo_agent` via the `Runner`. The `Runner` will manage the execution flow according to the agent's instructions.

**Expected Flow:**
1.  Agent receives "Hello ADK!".
2.  Agent calls `set_workflow_state` tool (state -> `PROCESSING`).
3.  Agent calls `set_workflow_state` tool (state -> `END`).
4.  Agent responds with the text "Hello ADK!".

The next cell initiates the `run_async` call and processes the stream of events generated during execution, logging the steps.


```
from google.genai.types import Content, Part

user_input_text = "Hello ADK!"
user_message = Content(role='user', parts=[Part(text=user_input_text)])
```


```
final_agent_response_text = None
async def process_interaction_events():
    """Helper async function to process events."""
    global final_agent_response_text
    event_count = 0
    async for event in runner.run_async(session_id=ID, new_message=user_message, user_id=USER_ID):
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

await process_interaction_events()
```

    [WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call'],returning concatenated text result from text parts,check out the non text parts for full response from model., ..., WARNING:google_genai.types:Warning: there are non-text parts in the response: ['function_call'],returning concatenated text result from text parts,check out the non text parts for full response from model.]
    
    [Event 1] Type: Event
      Role: model
      >>> Function Call <<<
          Name: set_workflow_state
          Args: {'state_name': 'PROCESSING'}
    
    [Event 2] Type: Event
      Role: user
      <<< Function Response >>>
          Name: set_workflow_state
          Data: {'status': 'success', 'message': 'Workflow state set to PROCESSING'}
    
    [Event 3] Type: Event
      Role: model
      >>> Function Call <<<
          Name: set_workflow_state
          Args: {'state_name': 'END'}
    
    [Event 4] Type: Event
      Role: user
      <<< Function Response >>>
          Name: set_workflow_state
          Data: {'status': 'success', 'message': 'Workflow state set to END'}
    
    [Event 5] Type: Event
      Role: model
      Text: 'Hello ADK!
    '


## 5. Analyze the Results

Examine the "Agent Event Log" printed above. You should clearly see the sequence reflecting the agent's instructions:

1.  `Function Call` event targeting `set_workflow_state` with `args={'state_name': 'PROCESSING'}`.
2.  `Function Response` event confirming the first tool execution.
3.  `Function Call` event targeting `set_workflow_state` with `args={'state_name': 'END'}`.
4.  `Function Response` event confirming the second tool execution.
5.  final event containing the agent's text response (the echoed message).

The next cell verifies this outcome by checking the final state stored in the session.


```
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

    Final workflow status: END
    Agent response: Hello ADK!
    


## Next Steps & Further Learning

This notebook demonstrated the basic structure of an ADK application, including:

*   Defining an `Agent` powered by Gemini.
*   Creating a simple `FunctionTool` to modify state.
*   Using `SessionService` and `ToolContext` for state management.
*   Orchestrating the flow with the `Runner`.

To dive deeper into the capabilities of the Google Agent Development Kit:

1.  **Explore the Official Documentation:** For detailed explanations of all components (Agents, Tools, Sessions, Callbacks, Multi-Agent systems, etc.), visit the [**Google ADK Documentation site**](https://google.github.io/adk-docs/).
2. **Try the Getting Started Notebook:** Explore the [**official ADK tutorial notebook on Colab**](https://colab.sandbox.google.com/github/google/adk-docs/blob/main/examples/python/tutorial/agent_team/adk_tutorial.ipynb) for a hands-on introduction to building your first agent.
3.  **Discover More Examples:** Check out the [**Google ADK GitHub repository**](https://github.com/google/adk-python)  for a wider range of examples, including more complex workflows, integrations, and advanced agent patterns.

Consider exploring concepts like:

*   **Workflow Agents** (`SequentialAgent`, `ParallelAgent`, `LoopAgent`) for structured process control.
*   **Multi-Agent Systems** for building collaborative agent teams.
*   Other **Tool Types** (OpenAPI, Google Cloud Tools, Built-in Tools) for broader integrations.