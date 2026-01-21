# Orchestrating AI Agents: A Guide to Routines and Handoffs

## Introduction

When building AI applications with language models, a well-crafted prompt and the right tools often deliver solid performance. However, as workflows become more complex with multiple unique processes, managing them effectively can become challenging. This guide introduces two powerful concepts—**routines** and **handoffs**—that enable you to orchestrate multiple AI agents in a simple, powerful, and controllable way.

We'll walk through implementing these concepts step-by-step, showing how to create specialized agents and enable them to transfer conversations seamlessly between each other. By the end, you'll have a framework for building sophisticated multi-agent systems.

## Prerequisites

First, ensure you have the necessary libraries installed and imported:

```bash
pip install openai pydantic
```

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import json
import inspect

# Initialize the OpenAI client
client = OpenAI()
```

## Part 1: Understanding and Implementing Routines

### What is a Routine?

A routine represents a set of steps an AI agent should follow, expressed in natural language (via a system prompt), along with the specific tools needed to complete those steps. Think of it as a specialized workflow for a particular task.

### Creating a Customer Service Routine

Let's create a customer service routine that triages user issues, suggests fixes, and processes refunds when necessary.

```python
# Define the system prompt for the customer service routine
system_message = (
    "You are a customer support agent for ACME Inc. "
    "Always answer in a sentence or less. "
    "Follow the following routine with the user:\n"
    "1. First, ask probing questions and understand the user's problem deeper.\n"
    "   - unless the user has already provided a reason.\n"
    "2. Propose a fix (make one up).\n"
    "3. ONLY if not satisfied, offer a refund.\n"
    "4. If accepted, search for the ID and then execute refund."
)

# Define the tools for this routine
def look_up_item(search_query):
    """Use to find item ID.
    Search query can be a description or keywords."""
    # In production, this would query a database
    return "item_132612938"

def execute_refund(item_id, reason="not provided"):
    """Process a refund for the given item."""
    print(f"Summary: {item_id}, {reason}")
    return "success"
```

The power of routines lies in their simplicity and robustness. Notice how the instructions contain conditional logic similar to a state machine. Language models handle these cases effectively for small to medium-sized routines, with the added benefit of "soft" adherence—they can naturally steer conversations without getting stuck.

### Building the Routine Execution Engine

To execute a routine, we need a loop that:
1. Takes user input
2. Calls the language model with the routine's instructions
3. Handles any function calls the model makes
4. Returns the results

First, let's create a helper function to convert Python functions into the JSON schema format that OpenAI's API expects:

```python
def function_to_schema(func) -> dict:
    """Convert a Python function to an OpenAI tool schema."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        param_type = type_map.get(param.annotation, "string")
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
```

Test the conversion function:

```python
def sample_function(param_1, param_2, the_third_one: int, some_optional="John Doe"):
    """This is my docstring. Call this function when you want."""
    print("Hello, world")

schema = function_to_schema(sample_function)
print(json.dumps(schema, indent=2))
```

Now, let's implement the complete routine execution engine:

```python
def execute_tool_call(tool_call, tools_map, agent_name=""):
    """Execute a tool call and return the result."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    print(f"{agent_name}: {name}({args})")
    
    # Call the corresponding function with provided arguments
    return tools_map[name](**args)

def run_full_turn(system_message, tools, messages):
    """Execute a complete turn of conversation with tool handling."""
    num_init_messages = len(messages)
    messages = messages.copy()
    
    while True:
        # Convert Python functions to tool schemas
        tool_schemas = [function_to_schema(tool) for tool in tools]
        tools_map = {tool.__name__: tool for tool in tools}
        
        # Get completion from the model
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_message}] + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)
        
        if message.content:
            print("Assistant:", message.content)
        
        if not message.tool_calls:
            break
        
        # Handle all tool calls
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools_map)
            
            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)
    
    # Return only the new messages
    return messages[num_init_messages:]

# Initialize the conversation
messages = []
tools = [execute_refund, look_up_item]

# Run the conversation loop
while True:
    user = input("User: ")
    messages.append({"role": "user", "content": user})
    
    new_messages = run_full_turn(system_message, tools, messages)
    messages.extend(new_messages)
```

This implementation handles the complete flow: user input, model response, tool execution, and result integration. The loop continues until the model doesn't make any more tool calls.

## Part 2: Introducing Handoffs Between Agents

### The Need for Multiple Routines

While a single routine can handle many tasks, complex workflows often require specialized agents. Instead of creating one massive routine, we can create multiple focused agents and enable them to hand off conversations to each other.

### Defining an Agent Class

First, let's create a structured Agent class using Pydantic:

```python
class Agent(BaseModel):
    """Represents an AI agent with specific instructions and tools."""
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: list = []
```

### Updating the Execution Engine for Agents

Now, let's update our execution engine to work with Agent objects:

```python
def run_full_turn(agent, messages):
    """Execute a complete turn with a specific agent."""
    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()
    
    while True:
        # Convert tools to schemas
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools_map = {tool.__name__: tool for tool in current_agent.tools}
        
        # Get completion from the model
        response = client.chat.completions.create(
            model=current_agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}] + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)
        
        if message.content:
            print(f"{current_agent.name}:", message.content)
        
        if not message.tool_calls:
            break
        
        # Handle tool calls
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools_map, current_agent.name)
            
            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)
    
    # Return the new messages
    return messages[num_init_messages:]

def execute_tool_call(tool_call, tools_map, agent_name=""):
    """Execute a tool call and return the result."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    print(f"{agent_name}: {name}({args})")
    
    return tools_map[name](**args)
```

### Creating Specialized Agents

Let's create two specialized agents: one for sales and one for refunds:

```python
def execute_refund(item_name):
    """Process a refund for an item."""
    return "success"

def place_order(item_name):
    """Place an order for an item."""
    return "success"

# Define the refund agent
refund_agent = Agent(
    name="Refund Agent",
    instructions="You are a refund agent. Help the user with refunds.",
    tools=[execute_refund],
)

# Define the sales agent
sales_assistant = Agent(
    name="Sales Assistant",
    instructions="You are a sales assistant. Sell the user a product.",
    tools=[place_order],
)

# Example conversation with manual handoff
messages = []

# User wants to place an order
user_query = "Place an order for a black boot."
print("User:", user_query)
messages.append({"role": "user", "content": user_query})

# Sales assistant handles the request
response = run_full_turn(sales_assistant, messages)
messages.extend(response)

# User changes their mind
user_query = "Actually, I want a refund."
print("User:", user_query)
messages.append({"role": "user", "content": user_query})

# Refund agent takes over
response = run_full_turn(refund_agent, messages)
messages.extend(response)
```

### Implementing Automatic Handoffs

While manual handoffs work, we want agents to decide when to transfer conversations automatically. A simple but effective approach is to give agents `transfer_to_XXX` functions that return the target Agent object.

First, let's define a Response class to handle agent transitions:

```python
class Response(BaseModel):
    """Container for agent responses that may include agent transfers."""
    agent: Optional[Agent]
    messages: list
```

Now, let's update our execution engine to handle automatic handoffs:

```python
def run_full_turn(agent, messages):
    """Execute a complete turn with automatic handoff support."""
    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()
    
    while True:
        # Convert tools to schemas
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}
        
        # Get completion from the model
        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}] + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)
        
        if message.content:
            print(f"{current_agent.name}:", message.content)
        
        if not message.tool_calls:
            break
        
        # Handle tool calls
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, current_agent.name)
            
            # Check if the result is an Agent (handoff request)
            if type(result) is Agent:
                current_agent = result
                result = f"Transferred to {current_agent.name}. Adopt persona immediately."
            
            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)
    
    # Return the current agent and new messages
    return Response(agent=current_agent, messages=messages[num_init_messages:])

def execute_tool_call(tool_call, tools, agent_name):
    """Execute a tool call and return the result."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    print(f"{agent_name}: {name}({args})")
    
    return tools[name](**args)
```

### Building a Complete Multi-Agent System

Let's create a comprehensive example with multiple specialized agents:

```python
# Define helper functions for the system
def escalate_to_human(summary):
    """Only call this if explicitly asked to."""
    print("Escalating to human agent...")
    print("\n=== Escalation Report ===")
    print(f"Summary: {summary}")
    print("=========================\n")
    exit()

def execute_order(product, price: int):
    """Price should be in USD."""
    print("\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=====================\n")
    
    confirm = input("Confirm order? y/n: ").strip().lower()
    if confirm == "y":
        print("Order execution successful!")
        return "Success"
    return "Order cancelled"

# Define the agents
def transfer_to_sales_agent():
    """Use for anything sales or buying related."""
    return sales_agent

def transfer_to_issues_and_repairs():
    """Use for issues, repairs, or refunds."""
    return issues_and_repairs_agent

def transfer_back_to_triage():
    """Call this if the user brings up a topic outside of your purview."""
    return triage_agent

# Create the triage agent
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are a customer service bot for ACME Inc. "
        "Introduce yourself. Always be very brief. "
        "Gather information to direct the customer to the right department. "
        "But make your questions subtle and natural."
    ),
    tools=[transfer_to_sales_agent, transfer_to_issues_and_repairs, escalate_to_human],
)

# Create the sales agent
sales_agent = Agent(
    name="Sales Agent",
    instructions=(
        "You are a sales agent for ACME Inc. "
        "Help customers find and purchase products. "
        "Be persuasive but not pushy. "
        "Use the execute_order function to finalize purchases."
    ),
    tools=[execute_order, transfer_back_to_triage],
)

# Create the issues and repairs agent
issues_and_repairs_agent = Agent(
    name="Issues & Repairs Agent",
    instructions=(
        "You handle product issues, repairs, and refunds for ACME Inc. "
        "Be empathetic and solution-oriented. "
        "For refunds, check if the product is within the 30-day return window."
    ),
    tools=[transfer_back_to_triage],
)

# Start a conversation with the triage agent
messages = []
current_agent = triage_agent

print("Starting conversation with Triage Agent...")
print("Type 'exit' to end the conversation.\n")

while True:
    user_input = input("User: ")
    
    if user_input.lower() == 'exit':
        break
    
    messages.append({"role": "user", "content": user_input})
    
    # Run a turn with the current agent
    response = run_full_turn(current_agent, messages)
    
    # Update the conversation history
    messages.extend(response.messages)
    
    # Update the current agent if there was a handoff
    if response.agent and response.agent != current_agent:
        current_agent = response.agent
        print(f"\n[System: Transferred to {current_agent.name}]\n")
```

## Conclusion

You've now built a complete multi-agent system with routines and handoffs. This architecture provides several key benefits:

1. **Modularity**: Each agent focuses on a specific domain, making them easier to develop and maintain.
2. **Scalability**: You can add new agents without modifying existing ones.
3. **Robustness**: The system gracefully handles complex conversation flows.
4. **Control**: You maintain oversight while allowing agents to make intelligent handoff decisions.

The concepts demonstrated here—routines as specialized workflows and handoffs as seamless transitions between agents—form the foundation for building sophisticated AI applications. You can extend this framework by adding more agents, refining their instructions, or implementing more sophisticated handoff logic.

For a production-ready implementation with additional features and examples, check out the [Swarm repository](https://github.com/openai/swarm).