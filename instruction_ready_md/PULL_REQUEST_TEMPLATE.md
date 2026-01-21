# AI Engineering Cookbook: Building a Custom Agent with LangChain

## Overview
This guide walks you through creating a custom AI agent using LangChain that can execute Python code to solve mathematical problems. You'll learn how to set up the agent, define its tools, and implement a structured workflow for code execution and result validation.

## Prerequisites

Before starting, ensure you have the necessary packages installed:

```bash
pip install langchain langchain-openai
```

## Step 1: Import Required Libraries

Begin by importing the essential components from LangChain and setting up your environment:

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

## Step 2: Define the Code Execution Tool

The core functionality of our agent is executing Python code. We'll create a custom tool that safely runs code and returns the result:

```python
def execute_python_code(code: str) -> str:
    """
    Executes Python code and returns the output.
    
    Args:
        code: Python code as a string
        
    Returns:
        Execution result or error message
    """
    try:
        # Create a local namespace for execution
        local_vars = {}
        
        # Execute the code
        exec(code, {}, local_vars)
        
        # Return the result if available
        if 'result' in local_vars:
            return str(local_vars['result'])
        else:
            return "Code executed successfully. No result variable found."
            
    except Exception as e:
        return f"Error executing code: {str(e)}"

# Create the tool instance
code_tool = Tool(
    name="PythonCodeExecutor",
    func=execute_python_code,
    description="Executes Python code and returns the result. Use this for mathematical calculations and data processing."
)
```

## Step 3: Configure the Language Model

Initialize the language model that will power the agent's reasoning capabilities:

```python
# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    max_tokens=1000
)
```

## Step 4: Create the Agent Prompt Template

Define a structured prompt that guides the agent's thought process and actions:

```python
agent_prompt = PromptTemplate.from_template("""
You are a helpful AI assistant that can execute Python code to solve problems.

When presented with a task:
1. Think through the problem step by step
2. Write Python code to solve it
3. Execute the code using the PythonCodeExecutor tool
4. Return the final answer

Available tools:
{tools}

Task: {input}

{agent_scratchpad}
""")
```

## Step 5: Assemble and Initialize the Agent

Combine all components to create the complete agent:

```python
# Create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=[code_tool],
    prompt=agent_prompt
)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[code_tool],
    verbose=True,
    max_iterations=3,
    handle_parsing_errors=True
)
```

## Step 6: Test the Agent with Mathematical Problems

Now let's test our agent with various mathematical challenges:

### Example 1: Basic Arithmetic

```python
result = agent_executor.invoke({
    "input": "Calculate 15% of 80, then add 25 to the result"
})

print(f"Result: {result['output']}")
```

**Expected Output:**
```
> Entering new AgentExecutor chain...
I need to calculate 15% of 80 first, then add 25 to that result.

Action: PythonCodeExecutor
Action Input: result = (15/100) * 80 + 25

Observation: 37.0

Thought: I have the result. The calculation is complete.

Action: Final Answer
Action Input: 37.0

Result: 37.0
```

### Example 2: Complex Mathematical Operation

```python
result = agent_executor.invoke({
    "input": "What is the square root of 144 multiplied by 3, then divided by 2?"
})

print(f"Result: {result['output']}")
```

**Expected Output:**
```
> Entering new AgentExecutor chain...
I need to calculate sqrt(144) * 3 / 2.

Action: PythonCodeExecutor
Action Input: import math
result = math.sqrt(144) * 3 / 2

Observation: 18.0

Thought: I have the result. The calculation is complete.

Action: Final Answer
Action Input: 18.0

Result: 18.0
```

## Step 7: Handle Edge Cases and Errors

Let's test how the agent handles problematic inputs:

```python
# Test with invalid code
result = agent_executor.invoke({
    "input": "Calculate 10 divided by 0"
})

print(f"Result: {result['output']}")
```

**Expected Output:**
```
> Entering new AgentExecutor chain...
I need to calculate 10 divided by 0.

Action: PythonCodeExecutor
Action Input: result = 10 / 0

Observation: Error executing code: division by zero

Thought: I encountered an error. Division by zero is mathematically undefined.

Action: Final Answer
Action Input: Division by zero is undefined in mathematics.

Result: Division by zero is undefined in mathematics.
```

## Best Practices and Tips

1. **Error Handling**: Always wrap code execution in try-except blocks to prevent crashes
2. **Security**: Consider adding sandboxing for production use to prevent malicious code execution
3. **Resource Limits**: Implement timeouts and memory limits for long-running calculations
4. **Validation**: Add input validation to ensure code is safe and relevant to the task
5. **Caching**: Implement result caching for repeated calculations to improve performance

## Conclusion

You've successfully built a custom AI agent that can execute Python code to solve mathematical problems. This foundation can be extended to handle more complex tasks by:

1. Adding more specialized tools (data visualization, API calls, etc.)
2. Implementing memory for multi-turn conversations
3. Adding validation layers for code safety
4. Integrating with external data sources

The agent follows the ReAct (Reasoning + Acting) pattern, making it transparent in its decision-making process and reliable in its execution.