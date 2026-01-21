# Build a Bank Support Agent with Pydantic AI and Mistral AI

In this guide, you'll build a bank support agent using Mistral AI and Pydantic AI. This agent will demonstrate several key features of modern AI application development:
*   **Structured Responses:** Using Pydantic to ensure outputs conform to a predefined, validated schema.
*   **External Dependencies:** Integrating external data sources (like a database) via a type-safe dependency injection system.
*   **Dynamic Context:** Personalizing interactions by injecting runtime information (e.g., a customer's name) into the agent's system prompt.
*   **Tool Integration:** Empowering the agent to call functions for real-time information retrieval.

> **Note:** This example is adapted from the official [Pydantic AI documentation](https://ai.pydantic.dev/).

## Prerequisites & Setup

First, install the required libraries and configure your environment.

1.  **Install the packages:**
    ```bash
    pip install pydantic-ai==0.0.14 nest_asyncio
    ```

2.  **Apply `nest_asyncio` (for Jupyter/Colab only):**
    If you are running this code in a Jupyter notebook or Google Colab, you need to apply `nest_asyncio` to manage event loop conflicts.
    ```python
    import nest_asyncio
    nest_asyncio.apply()
    ```

3.  **Set your Mistral AI API Key:**
    ```python
    import os
    from getpass import getpass

    os.environ["MISTRAL_API_KEY"] = getpass("Enter your Mistral AI API Key: ")
    ```

4.  **Import the core modules:**
    ```python
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.mistral import MistralModel
    from pydantic import BaseModel, Field
    from dataclasses import dataclass
    ```

## Part 1: Basic Q&A with Mistral

Let's start with a simple example to verify our setup. We'll create a basic agent that uses Mistral and is instructed to be concise.

```python
# Initialize the Mistral model
model = MistralModel('mistral-small-latest')

# Create an agent with a simple system prompt
agent = Agent(
    model,
    system_prompt='Be concise, reply with one sentence.',
)

# Run the agent synchronously
result = agent.run_sync('Where does "hello world" come from?')
print(result.data)
```

This agent will return a brief, one-sentence answer about the origin of the "hello world" phrase.

## Part 2: Building the Bank Support Agent

Now, let's build the more complex bank support agent. This involves defining a mock database, the agent's dependencies, its expected output format, and the tools it can use.

### Step 1: Define a Mock Database

In a real application, your agent would fetch data from an external source. Here, we create a simple class to simulate a database connection.

```python
class DatabaseConn:
    """
    A simulated database for example purposes.
    In reality, this would connect to a system like PostgreSQL.
    """
    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        # Simulate fetching a customer name by ID
        if id == 123:
            return 'John'
        return None

    @classmethod
    async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
        # Simulate fetching a customer's balance
        if id == 123:
            return 123.45
        else:
            raise ValueError('Customer not found')
```

### Step 2: Define the Agent's Dependencies and Output Schema

We use Pydantic to strictly define what data the agent needs (dependencies) and what format its response must take (result type). This ensures type safety and predictable outputs.

1.  **Define the dependencies:** The agent needs a customer ID and a database connection to operate.
    ```python
    @dataclass
    class SupportDependencies:
        customer_id: int
        db: DatabaseConn
    ```

2.  **Define the result structure:** Every agent response will be a validated Pydantic model with these fields.
    ```python
    class SupportResult(BaseModel):
        support_advice: str = Field(description='Advice returned to the customer')
        block_card: bool = Field(description='Whether to block their card')
        risk: int = Field(description='Risk level of query', ge=0, le=10)
    ```

3.  **Initialize the support agent:** Create the agent, specifying its model, dependencies, result type, and a base system prompt.
    ```python
    support_agent = Agent(
        model, # The Mistral model from Part 1
        deps_type=SupportDependencies,
        result_type=SupportResult,
        system_prompt=(
            'You are a support agent for our bank. '
            'Provide the customer with support and assess the risk level of their query. '
            "Use the customer's name in your reply."
        ),
    )
    ```

### Step 3: Add a Dynamic System Prompt

To personalize interactions, we can attach a function that dynamically modifies the system prompt at runtime, injecting the customer's name.

```python
@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    """Dynamically adds the customer's name to the system prompt."""
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"
```

### Step 4: Define Tools for the Agent

Tools allow the agent to actively retrieve information. By decorating a function with `@support_agent.tool`, we expose it to the model for use during its reasoning process.

```python
@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> str:
    """Returns the customer's current account balance. Use this when asked about balance."""
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return f'${balance:.2f}'
```

### Step 5: Run the Agent

Now you can test the agent with different customer queries. You must provide the dependencies (customer ID and database) when running it.

1.  **Create the dependency object:**
    ```python
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    ```

2.  **Query 1: Check Balance**
    The agent will use the `customer_balance` tool you defined.
    ```python
    result = support_agent.run_sync('What is my balance?', deps=deps)
    print(result.data)
    # Example output:
    # support_advice='Hello John, your current balance is $123.45.' block_card=False risk=1
    ```

3.  **Query 2: Report a Lost Card**
    The agent will assess this as a higher-risk scenario.
    ```python
    result = support_agent.run_sync('I just lost my card!', deps=deps)
    print(result.data)
    # Example output:
    # support_advice="I'm sorry to hear that, John. I've blocked your card immediately to prevent fraud." block_card=True risk=8
    ```

### Step 6: Inspect the Agent's Reasoning (Optional)

You can examine the full interaction history, including the dynamic system prompt and tool calls, which is useful for debugging and understanding the agent's process.

```python
# This shows the detailed message history and metadata
print(result.messages)
# or inspect the full result object
print(result.__dict__)
```

## Summary

You have successfully built a bank support agent that:
*   **Personalizes responses** by dynamically fetching and using the customer's name.
*   **Interacts with external data** via a simulated database connection and a custom tool.
*   **Returns structured, validated outputs** using Pydantic models, making its responses easy to integrate into other systems.
*   **Assesses risk and takes action** (like blocking a card) based on the context of the customer's query.

This pattern of dependencies, dynamic prompts, and tools provides a robust foundation for building complex, reliable AI agents for various support and operational tasks.