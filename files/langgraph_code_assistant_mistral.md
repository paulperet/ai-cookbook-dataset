# Codestral with self-correction

> Codestral is a cutting-edge generative model that has been specifically designed and optimized
for code generation tasks, including fill-in-the-middle and code completion. Codestral was trained
on 80+ programming languages, enabling it to perform well on both common and less common
languages

We can combine the code generation capabilities of Codestral the self-correction approach presented in the [AlphaCodium](https://github.com/Codium-ai/AlphaCodium) paper, [constructing an answer to a coding question iteratively](https://x.com/karpathy/status/1748043513156272416?s=20).  

We will implement some of these ideas from scratch using [LangGraph](https://python.langchain.com/docs/langgraph) to 1) produce structured code generation output from Codestral-instruct, 2) perform inline unit tests to confirm imports and code execution work, 3) feed back any errors for Codestral for self-correction.

```python
! pip install -U langchain_community langchain-mistralai langchain langgraph
```

### LLM

We'll use the Mistral API and `Codestral` instruct model, which support tool use!


```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("MISTRAL_API_KEY")
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
```

Optionally, you can use [LangSmith](https://docs.smith.langchain.com/) for tracing. 


```python
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "mistral-cookbook"
```

## Code Generation

Test with structured output.


```python
# Select LLM
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Mistral model
mistral_model = "codestral-latest"
llm = ChatMistralAI(model=mistral_model, temperature=0)

# Prompt 
code_gen_prompt_claude = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """You are a coding assistant. Ensure any code you provide can be executed with all required imports and variables \n
            defined. Structure your answer: 1) a prefix describing the code solution, 2) the imports, 3) the functioning code block.
            \n Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)

# Data model
class code(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

# LLM
code_gen_chain = llm.with_structured_output(code, include_raw=False)
```


```python
question = "Write a function for fibonacci."
messages = [("user", question)]
```


```python
# Test
result = code_gen_chain.invoke(messages)
result
```




    code(prefix='A function to calculate the nth Fibonacci number.', imports='', code='def fibonacci(n):\n    if n <= 0:\n        return "Input should be positive integer"\n    elif n == 1:\n        return 0\n    elif n == 2:\n        return 1\n    else:\n        a, b = 0, 1\n        for _ in range(2, n):\n            a, b = b, a + b\n        return b')



# Graph 

We build the above workflow as a graph using [LangGraph](https://langchain-ai.github.io/langgraph/).

### Graph state

The graph `state` schema contains keys that we want to:

* Pass to each node in our graph
* Optionally, modify in each node of our graph 

See conceptual docs [here](https://langchain-ai.github.io/langgraph/concepts/low_level/#state).


```python
from typing import Annotated
from typing import Dict, TypedDict, List
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

## Graph


```python
from operator import itemgetter
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

### Parameters
max_iterations = 3

### Nodes
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
    error = state.get("error", "")

    # Solution
    code_solution = code_gen_chain.invoke(messages)
    messages += [
        (
            "assistant",
            f"Here is my attempt to solve the problem: {code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
        )
    ]

    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}

def code_check(state: GraphState):
    """
    Check code

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
    prefix = code_solution.prefix
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

### Conditional edges

def decide_to_finish(state: GraphState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"
```

We'll add persistence to the graph using [a checkpointer](https://langchain-ai.github.io/langgraph/concepts/persistence/).


```python
from IPython.display import Image, display
from langgraph.graph import END, StateGraph

# Define the graph
builder = StateGraph(GraphState)

# Define the nodes
builder.add_node("generate", generate)  # generation solution
builder.add_node("check_code", code_check)  # check code

# Build graph
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

graph = builder.compile()
display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
```


```python
from langchain_core.messages import HumanMessage
question = "Write a Python program that prints 'Hello, World!' to the console."
for event in graph.stream({"messages": [HumanMessage(content=question)], "iterations": 0}, stream_mode="values"):
    print(event)
```

    [{'messages': [HumanMessage(content="Write a Python program that prints 'Hello, World!' to the console.", additional_kwargs={}, response_metadata={}, id='9dd8d0c4-50f7-4b1d-8377-82e16ff0261d')], 'iterations': 0}, ..., {'error': 'no', 'messages': [HumanMessage(content="Write a Python program that prints 'Hello, World!' to the console.", additional_kwargs={}, response_metadata={}, id='9dd8d0c4-50f7-4b1d-8377-82e16ff0261d'), AIMessage(content="Here is my attempt to solve the problem: The task is to write a simple Python program that prints 'Hello, World!' to the console. There are no specific imports required for this task. \n Imports:  \n Code: print('Hello, World!')", additional_kwargs={}, response_metadata={}, id='4f9368ca-af4e-44c6-9473-330d8c9000ee'), AIMessage(content="Here is my attempt to solve the problem: The task is to write a simple Python program that prints 'Hello, World!' to the console. There are no specific imports required for this task. \n Imports:  \n Code: print('Hello, World!')", additional_kwargs={}, response_metadata={}, id='7a9bcea4-e55b-4ceb-afa4-5b5fb9a2149d')], 'generation': code(prefix="The task is to write a simple Python program that prints 'Hello, World!' to the console. There are no specific imports required for this task.", imports='', code="print('Hello, World!')"), 'iterations': 1}]


`Trace:`

https://smith.langchain.com/public/a59ec940-f618-411d-adc9-1781816e7627/r


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

for event in graph.stream({"messages": [HumanMessage(content=question)], "iterations": 0}, stream_mode="values"):
    print(event)
```

    [{'messages': [HumanMessage(content='Create a Python program that allows two players to play a game of Tic-Tac-Toe. The game should be played on a 3x3 grid. The program should:\n\n- Allow players to take turns to input their moves.\n- Check for invalid moves (e.g., placing a marker on an already occupied space).\n- Determine and announce the winner or if the game ends in a draw.\n\nRequirements:\n- Use a 2D list to represent the Tic-Tac-Toe board.\n- Use functions to modularize the code.\n- Validate player input.\n- Check for win conditions and draw conditions after each move.', additional_kwargs={}, response_metadata={}, id='8950badb-d5f4-4ccf-b55f-f52a74ad3e68')], 'iterations': 0}, ..., {'error': 'no', 'messages': [HumanMessage(content='Create a Python program that allows two players to play a game of Tic-Tac-Toe. The game should be played on a 3x3 grid. The program should:\n\n- Allow players to take turns to input their moves.\n- Check for invalid moves (e.g., placing a marker on an already occupied space).\n- Determine and announce the winner or if the game ends in a draw.\n\nRequirements:\n- Use a 2D list to represent the Tic-Tac-Toe board.\n- Use functions to modularize the code.\n- Validate player input.\n- Check for win conditions and draw conditions after each move.', additional_kwargs={}, response_metadata={}, id='8950badb-d5f4-4ccf-b55f-f52a74ad3e68'), AIMessage(content="Here is my attempt to solve the problem: The program will use a 2D list to represent the Tic-Tac-Toe board. It will have functions to display the board, check for win conditions, check for draw conditions, and validate player input. The game will be played in a loop until a win condition is met or the board is full. \n Imports:  \n Code: board = [[' ' for _ in range(3)] for _ in range(3)]\n\ndef display_board():\n    for row in board:\n        print('|'.join(row))\n        print('-' * 5)\n\ndef check_win(player):\n    for row in board:\n        if all(cell == player for cell in row):\n            return True\n    for col in range(3):\n        if all(board[row][col] == player for row in range(3)):\n            return True\n    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):\n        return True\n    return False\n\ndef check_draw():\n    return all(cell != ' ' for row in board for cell in row)\n\ndef validate_input(move):\n    try:\n        row, col = map(int, move.split(','))\n        if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == ' ':\n            return row, col\n    except:\n        pass\n    return None\n\nplayers = ['X', 'O']\ncurrent_player = 0\n\nwhile True:\n    display_board()\n    move = input(f'Player {players[current_player]}, enter your move (row,col): ')\n    move = validate_input(move)\n    if move is None:\n        print('Invalid move. Try again.')\n        continue\n    row, col = move\n    board[row][col] = players[current_player]\n    if check_win(players[current_player]):\n        display_board()\n        print(f'Player {players[current_player]} wins!')\n        break\n    if check_draw():\n        display_board()\n        print('The game is a draw.')\n        break\n    current_player = (current_player + 1) % 2", additional_kwargs={}, response_metadata={}, id='d82552ee-3ecd-4174-8c19-adb7895d7226'), AIMessage(content="Here is my attempt to solve the problem: The program will use a 2D list to represent the Tic-Tac-Toe board. It will have functions to display the board, check for win conditions, check for draw conditions, and validate player input. The game will be played in a loop until a win condition is met or the board is full. \n Imports:  \n Code: board = [[' ' for _ in range(3)] for _ in range(3)]\n\ndef display_board():\n    for row in board:\n        print('|'.join(row))\n        print('-' * 5)\n\ndef check_win(player):\n    for row in board:\n        if all(cell == player for cell in row):\n            return True\n    for col in range(3):\n        if all(board[row][col] == player for row in range(3)):\n            return True\n    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):\n        return True\n    return False\n\ndef check_draw():\n    return all(cell != ' ' for row in board for cell in row)\n\ndef validate_input(move):\n    try:\n        row, col = map(int, move.split(','))\n        if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == ' ':\n            return row, col