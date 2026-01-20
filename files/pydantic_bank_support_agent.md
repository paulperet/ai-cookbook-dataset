# Build a bank support agent with Pydantic AI and Mistral AI

In this cookbook, we'll demonstrate how to build a bank support agent using Mistral AI and PydanticAI, which offers the following features:
- Structured Responses: Pydantic ensures that outputs conform to a predefined schema, providing consistent and validated responses.
- External Dependencies: Enhance AI interactions by integrating external dependencies, such as databases, through a type-safe dependency injection system.
- Dynamic Context: System prompt functions allow the injection of runtime information, like a customer's name, into the agent's context, enabling personalized interactions.
- Tool Integration: The agent can invoke tools for real-time information retrieval, enhancing its capabilities beyond static responses.

Example in this cookbook is adapted from https://ai.pydantic.dev/.

```python
!pip install pydantic-ai==0.0.14 nest_asyncio
```

[First Entry, ..., Last Entry]

If you're running pydantic-ai in a jupyter notebook or Colab, you will need nest-asyncio to manage conflicts between event loops that occur between jupyter's event loops and pydantic-ai's:

```python
import nest_asyncio
nest_asyncio.apply()
```

```python
import os
from getpass import getpass

os.environ["MISTRAL_API_KEY"] = getpass("Type your API Key")
```

## Example 1: Basic Q&A with Mistral

Let's start with a straightforward example using Pydantic AI for a basic Q&A with Mistral.

We'll define an agent powered by Mistral with a system prompt designed to ensure concise responses. When we ask about the origin of “hello world,” the model will provide a brief, one-sentence answer.

```python
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel

model = MistralModel('mistral-small-latest')

agent = Agent(
    model,
    system_prompt='Be concise, reply with one sentence.',
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.data)
```

# Example 2: Bank support agent

In this more complex example, we build a bank support agent.

## Step 1: Define a database

In more advanced AI workflows, your model may require external data, such as information from a database. Here, we define a fictional database class that retrieves a customer's name and balance. In a real-world scenario, this class could connect to a live database. The agent can use these methods to respond to customer inquiries effectively.

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


class DatabaseConn:
    """This is a fake database for example purposes.

    In reality, you'd be connecting to an external database
    (e.g. PostgreSQL) to get information about customers.
    """

    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        if id == 123:
            return 'John'

    @classmethod
    async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
        if id == 123:
            return 123.45
        else:
            raise ValueError('Customer not found')
```

## Step 2: Define the bank support agent

In this step, we define how the support agent works by setting up its input dependencies and expected output format.

Code Breakdown:

	1.	Input Dependencies:
	•	SupportDependencies specifies what the agent needs to function:
	•	customer_id (the ID of the customer being helped).
	•	db (a connection to the database).

	2.	Expected Response Format:
	•	SupportResult defines the response structure, ensuring consistency:
	•	support_advice: A string containing advice given to the customer.
	•	block_card: A boolean indicating whether the customer’s card should be blocked.
	•	risk: An integer from 0 to 10, representing the assessed risk level.

	3.	Agent Initialization:
	•	We create the support_agent using the Agent class:
	•	model: The underlying AI model.
	•	deps_type: Specifies the required input dependencies (SupportDependencies).
	•	result_type: Defines the expected output structure (SupportResult).
	•	system_prompt: A prompt guiding the agent to act as a bank support representative. The prompt ensures customer-specific responses by including the customer’s name.

Why This Design Matters:

- By defining input dependencies and output formats, we guarantee that the agent always receives the correct data and produces predictable results. This makes integration into larger systems easier and supports clear, actionable responses.

```python
@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


class SupportResult(BaseModel):
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description='Whether to block their')
    risk: int = Field(description='Risk level of query', ge=0, le=10)


support_agent = Agent(
    model,
    deps_type=SupportDependencies,
    result_type=SupportResult,
    system_prompt=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query. '
        "Reply using the customer's name."
    ),
)
```

## Step 3: Add a dynamic system prompt

This code attaches a dynamic system prompt function. Before the model sees the user's query, it gets a special system prompt enriched with the customer's name.

```python
@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"
```

## Step 4: Defining tools that the agent can use

By decorating customer_balance with @support_agent.tool, we're telling the model it can call this function to retrieve the customer's balance. This transforms the model from a passive text generator into an active problem solver that can interact with external resources.

```python
@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> str:
    """Returns the customer's current account balance."""
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return f'${balance:.2f}'
```

## Step 5: Run agent

When asked about the customer’s balance, the agent uses the injected dependencies and tools to return a structured response.

```python
deps = SupportDependencies(customer_id=123, db=DatabaseConn())
result = support_agent.run_sync('What is my balance?', deps=deps)
print(result.data)
```

```python
result = support_agent.run_sync('I just lost my card!', deps=deps)
print(result.data)
```

You can check the results and message history including the system prompt and the tool usage:

```python
result.__dict__
```

```python

```