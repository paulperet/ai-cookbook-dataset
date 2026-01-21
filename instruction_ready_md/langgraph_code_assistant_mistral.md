# Building a Self-Correcting Code Generator with Codestral and LangGraph

This guide demonstrates how to combine the powerful code generation capabilities of Mistral's Codestral model with a self-correction loop using LangGraph. Inspired by the AlphaCodium paper, we'll create a system that iteratively generates, tests, and refines code solutions.

## Prerequisites

First, install the required packages:

```bash
pip install -U langchain_community langchain-mistralai langchain langgraph
```

## Setup and Configuration

### 1. Configure API Keys

Set up your Mistral API key. Optionally, configure LangSmith for tracing:

```python
import os
import getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Set Mistral API key
_set_env("MISTRAL_API_KEY")
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Optional: Set LangSmith for tracing
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "mistral-cookbook"
```

### 2. Initialize the LLM

We'll use the Codestral instruct model, which supports structured output:

```python
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Initialize the Codestral model
mistral_model = "codestral-latest"
llm = ChatMistralAI(model=mistral_model, temperature=0)
```

### 3. Define the Code Generation Chain

Create a structured output chain that generates code with a specific format:

```python
# Define the prompt template
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """You are a coding assistant. Ensure any code you provide can be executed with all required imports and variables defined. Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the functioning code block.
            \n Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)

# Define the output schema
class CodeSolution(BaseModel):
    """Schema for code solutions to questions about LCEL."""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

# Create the structured output chain
code_gen_chain = llm.with_structured_output(CodeSolution, include_raw=False)
```

### 4. Test the Code Generator

Let's test the basic code generation with a simple example:

```python
question = "Write a function for fibonacci."
messages = [("user", question)]

# Generate a solution
result = code_gen_chain.invoke(messages)
print(result)
```

Output:
```
prefix='A function to calculate the nth Fibonacci number.'
imports=''
code='def fibonacci(n):
    if n <= 0:
        return "Input should be positive integer"
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n):
            a, b = b, a + b
        return b'
```

## Building the Self-Correction Graph

Now we'll implement the self-correction workflow using LangGraph.

### 1. Define the Graph State

The state tracks the progress of our code generation and testing:

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import AnyMessage, add_messages

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """
    error: str
    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    iterations: int
```

### 2. Implement the Graph Nodes

We need three main nodes: one for generation, one for testing, and a conditional router.

#### Generation Node

```python
def generate(state: GraphState):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """
    print("---GENERATING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]

    # Generate solution
    code_solution = code_gen_chain.invoke(messages)
    messages += [
        (
            "assistant",
            f"Here is my attempt to solve the problem: {code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
        )
    ]

    # Increment iteration counter
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}
```

#### Code Checking Node

```python
def code_check(state: GraphState):
    """
    Check code for import and execution errors

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """
    print("---CHECKING CODE---")

    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code

    # Check imports
    try:
        exec(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the import test. Here is the error: {e}. Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # Check execution
    try:
        combined_code = f"{imports}\n{code}"
        # Use a shared scope for exec
        global_scope = {}
        exec(combined_code, global_scope)
    except Exception as e:
        print("---CODE BLOCK CHECK: FAILED---")
        error_message = [("user", f"Your solution failed the code execution test: {e}) Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # No errors
    print("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }
```

#### Conditional Routing

```python
def decide_to_finish(state: GraphState):
    """
    Determines whether to finish or retry.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state["error"]
    iterations = state["iterations"]
    max_iterations = 3

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"
```

### 3. Assemble the Graph

Now let's connect all the nodes into a complete workflow:

```python
from langgraph.graph import END, StateGraph

# Define the graph
builder = StateGraph(GraphState)

# Add nodes
builder.add_node("generate", generate)  # Generate solution
builder.add_node("check_code", code_check)  # Check code

# Build graph structure
builder.set_entry_point("generate")
builder.add_edge("generate", "check_code")
builder.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "generate": "generate",
    },
)

# Compile the graph
graph = builder.compile()
```

## Running the Self-Correcting Code Generator

### Example 1: Simple "Hello, World!" Program

```python
from langchain_core.messages import HumanMessage

question = "Write a Python program that prints 'Hello, World!' to the console."

# Run the graph
for event in graph.stream({"messages": [HumanMessage(content=question)], "iterations": 0}, stream_mode="values"):
    print(event)
```

The system will generate, test, and output a working "Hello, World!" program.

### Example 2: Complex Tic-Tac-Toe Game

Let's test with a more complex problem:

```python
question = """Create a Python program that allows two players to play a game of Tic-Tac-Toe. The game should be played on a 3x3 grid. The program should:

- Allow players to take turns to input their moves.
- Check for invalid moves (e.g., placing a marker on an already occupied space).
- Determine and announce the winner or if the game ends in a draw.

Requirements:
- Use a 2D list to represent the Tic-Tac-Toe board.
- Use functions to modularize the code.
- Validate player input.
- Check for win conditions and draw conditions after each move."""

# Run the graph
for event in graph.stream({"messages": [HumanMessage(content=question)], "iterations": 0}, stream_mode="values"):
    print(event)
```

The system will generate a complete Tic-Tac-Toe game implementation with proper validation, win condition checking, and modular functions.

## How It Works

1. **Generation Phase**: The Codestral model generates a structured code solution with imports and implementation.
2. **Testing Phase**: The system attempts to execute the imports and code to catch any syntax or runtime errors.
3. **Feedback Loop**: If errors are found, they're fed back to the model along with instructions to reflect and correct.
4. **Termination**: The loop continues until either:
   - The code passes all tests (error = "no")
   - The maximum iteration limit is reached (default: 3 attempts)

## Key Features

- **Structured Output**: Ensures consistent formatting of code solutions
- **Automatic Testing**: Validates imports and code execution automatically
- **Self-Correction**: Uses error feedback to improve subsequent generations
- **Iteration Control**: Prevents infinite loops with a maximum attempt limit
- **Modular Design**: Easy to extend with additional test types or validation rules

This approach combines the power of modern code generation models with automated testing to create robust, self-correcting coding assistants.