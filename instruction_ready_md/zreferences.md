# Setup and Prerequisites

Before we begin, ensure you have the necessary libraries installed. This tutorial uses `langchain` for building the agent and `openai` for the language model. We'll also use `langchain_experimental` for the `PlanAndExecute` agent.

```bash
pip install langchain openai langchain_experimental
```

Now, import the required modules.

```python
import os
from langchain_openai import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents.tools import Tool
from langchain.chains import LLMMathChain
```

## Step 1: Configure the Language Model

First, set up the language model that will power the agent. We'll use OpenAI's GPT model. Ensure your OpenAI API key is set as an environment variable.

```python
# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize the language model
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
```

## Step 2: Define the Tools

The agent needs tools to perform tasks. We'll create a simple math tool using `LLMMathChain` and a custom tool for general knowledge questions.

### 2.1 Create the Math Tool

The `LLMMathChain` allows the agent to perform mathematical calculations.

```python
# Initialize the math chain
llm_math = LLMMathChain.from_llm(llm=llm)

# Define the math tool
math_tool = Tool(
    name="Calculator",
    func=llm_math.run,
    description="Useful for answering math questions. Input should be a mathematical expression."
)
```

### 2.2 Create a General Knowledge Tool

For questions outside of math, we'll create a tool that uses the language model directly.

```python
def general_qa(input_text: str) -> str:
    """A tool for answering general knowledge questions."""
    return llm.invoke(input_text).content

qa_tool = Tool(
    name="General QA",
    func=general_qa,
    description="Useful for answering general knowledge questions. Input should be a question."
)
```

### 2.3 Combine the Tools

Create a list of all available tools for the agent.

```python
tools = [math_tool, qa_tool]
```

## Step 3: Build the Plan-and-Execute Agent

The `PlanAndExecute` agent first creates a plan to solve a problem, then executes the steps using the available tools.

### 3.1 Load the Planner and Executor

The planner generates the step-by-step plan, and the executor carries it out.

```python
# Load the planner
planner = load_chat_planner(llm)

# Load the executor with the tools
executor = load_agent_executor(llm, tools, verbose=True)
```

### 3.2 Initialize the Agent

Combine the planner and executor into the final agent.

```python
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
```

## Step 4: Run the Agent

Now, let's test the agent with a complex query that requires both planning and tool use.

```python
# Define the user's question
question = "What is the square root of 25? Also, who was the first president of the United States?"

# Run the agent
result = agent.run(question)
```

### Expected Output

The agent will first generate a plan, then execute it. The final output should be:

```
The square root of 25 is 5. The first president of the United States was George Washington.
```

## Conclusion

You've successfully built a Plan-and-Execute agent using LangChain. This agent can handle multi-step queries by first planning the approach and then using specialized tools to find answers. You can extend this by adding more tools for different domains, such as web search or database queries.

## Next Steps

- Experiment with different language models (e.g., `gpt-4`).
- Add more tools, like a web search tool using `SerpAPI`.
- Customize the planner's prompt for more complex planning tasks.