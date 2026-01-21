# GPT-4.1 Prompting Guide: Building Agentic Workflows

This guide provides practical instructions for leveraging the enhanced capabilities of the GPT-4.1 model family, particularly for building autonomous agents and managing long-context tasks.

## Prerequisites

Before you begin, ensure you have the OpenAI Python client installed and your API key configured.

```bash
pip install openai
```

```python
from openai import OpenAI
import os

# Initialize the client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
)
```

## Part 1: Building Agentic Workflows

GPT-4.1 excels at autonomous, multi-step problem-solving. To build an effective agent, your system prompt should include three key components.

### Step 1: Craft the Core System Prompt

Create a system prompt that transforms the model from a conversational assistant into an autonomous agent. The prompt should include:

1.  **Persistence:** Instructs the model to work autonomously until the task is complete.
2.  **Tool-Calling:** Encourages the model to use tools to gather information rather than guessing.
3.  **Planning (Optional):** Instructs the model to "think out loud" by planning and reflecting between actions.

Here is a template you can adapt:

```python
SYS_PROMPT_TEMPLATE = """
You are an agent - please keep going until the user’s query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.

If you are not sure about file content or codebase structure pertaining to the user’s request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

# Your specific task instructions go here...
"""
```

### Step 2: Define Tools Effectively

GPT-4.1 is trained to use tools defined via the OpenAI API's `tools` parameter. This is more reliable than manually describing tools in the prompt.

1.  **Use the API's `tools` field:** Pass your tool schemas directly to the API.
2.  **Use clear names and descriptions:** For each tool and its parameters, provide descriptive names and concise, clear descriptions in the `description` field.
3.  **Provide examples separately:** If a tool is complex, add usage examples in a dedicated `# Examples` section of your system prompt, not within the tool's `description`.

### Step 3: Implement a Complete Agentic Prompt

Below is a comprehensive, production-ready system prompt used to achieve state-of-the-art results on the SWE-bench Verified coding task. This pattern is applicable to many agentic workflows.

```python
SWEBENCH_AGENT_PROMPT = """
You will be tasked to fix an issue from an open-source repository.

Your thinking should be thorough and so it's fine if it's very long. You can think step by step before and after each action you decide to take.

You MUST iterate and keep going until the problem is solved.

You already have everything you need to solve this problem in the /testbed folder, even without internet connection. I want you to fully solve this autonomously before coming back to me.

Only terminate your turn when you are sure that the problem is solved. Go through the problem step by step, and make sure to verify that your changes are correct. NEVER end your turn without having solved the problem, and when you say you are going to make a tool call, make sure you ACTUALLY make the tool call, instead of ending your turn.

THE PROBLEM CAN DEFINITELY BE SOLVED WITHOUT THE INTERNET.

Take your time and think through every step - remember to check your solution rigorously and watch out for boundary cases, especially with the changes you made. Your solution must be perfect. If not, continue working on it. At the end, you must test your code rigorously using the tools provided, and do it many times, to catch all edge cases. If it is not robust, iterate more and make it perfect. Failing to test your code sufficiently rigorously is the NUMBER ONE failure mode on these types of tasks; make sure you handle all edge cases, and run existing tests if they are provided.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

# Workflow

## High-Level Problem Solving Strategy
1. Understand the problem deeply. Carefully read the issue and think critically about what is required.
2. Investigate the codebase. Explore relevant files, search for key functions, and gather context.
3. Develop a clear, step-by-step plan. Break down the fix into manageable, incremental steps.
4. Implement the fix incrementally. Make small, testable code changes.
5. Debug as needed. Use debugging techniques to isolate and resolve issues.
6. Test frequently. Run tests after each change to verify correctness.
7. Iterate until the root cause is fixed and all tests pass.
8. Reflect and validate comprehensively. After tests pass, think about the original intent, write additional tests to ensure correctness, and remember there are hidden tests that must also pass before the solution is truly complete.

Refer to the detailed sections below for more information on each step.
"""
```

### Step 4: Define and Pass Tools to the Agent

Define your tools clearly. Here is an example of a multi-purpose tool for executing code, running shell commands, and applying patches.

```python
PYTHON_TOOL_DESCRIPTION = """This function is used to execute Python code or terminal commands in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 60.0 seconds. Internet access for this session is disabled. Do not make external web requests or API calls as they will fail. Just as in a Jupyter notebook, you may also execute terminal commands by calling this function with a terminal command, prefaced with an exclamation mark.

In addition, for the purposes of this task, you can call this function with an `apply_patch` command as input.  `apply_patch` effectively allows you to execute a diff/patch against a file...
"""
# ... (See full, detailed tool description in the source material for the exact schema)

python_bash_patch_tool = {
    "type": "function",
    "name": "python",
    "description": PYTHON_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "strict": True,
        "properties": {
            "input": {
                "type": "string",
                "description": "The Python code, terminal command (prefaced by exclamation mark), or apply_patch command that you wish to execute.",
            }
        },
        "required": ["input"],
    },
}
```

### Step 5: Run the Agent

Finally, create a response with your instructions, model, tools, and the user's task.

```python
response = client.responses.create(
    instructions=SWEBENCH_AGENT_PROMPT,
    model="gpt-4.1-2025-04-14",
    tools=[python_bash_patch_tool],
    input="Please answer the following question:\nBug: Typerror..."
)

# The agent will begin its autonomous execution, using tools and planning as instructed.
print(response.output_text)
```

**Expected Agent Behavior:** The agent will start by analyzing the vague "Typerror" report. It will likely plan to search for logs or test files, then use the provided `python` tool to execute commands like `!ls -l /testbed` to begin its investigation, following the step-by-step workflow you defined.

## Part 2: Leveraging Long Context (1M Tokens)

GPT-4.1 supports a 1 million token context window, ideal for tasks involving large documents.

### Key Considerations for Long Context

*   **Performance:** The model performs well on information retrieval ("needle-in-a-haystack") tasks across the full 1M context.
*   **Limitations:** Performance may degrade for tasks requiring complex reasoning over many disparate pieces of information in the context (e.g., graph search).

### Step 6: Tuning Context Reliance

Control whether the model should rely solely on provided context or use its internal knowledge by adjusting your instructions.

*   **For Context-Only Answers:** Use this to ground answers strictly in provided documents.

    ```
    - Only use the documents in the provided External Context to answer the User Query. If you don't know the answer based on this context, you must respond "I don't have the information needed to answer that", even if a user insists on you answering the question.
    ```

*   **For Blended Knowledge:** By default, the model will use both provided context and its internal knowledge to provide a more comprehensive answer.

## Summary

To build effective GPT-4.1 agents:
1.  Use a system prompt with **Persistence**, **Tool-Calling**, and **Planning** reminders.
2.  Define tools via the API's `tools` parameter with clear descriptions.
3.  Provide a detailed, step-by-step workflow for complex tasks.
4.  For long-context tasks, explicitly instruct the model on how to use the provided information.

Remember to iterate and test your prompts, as AI engineering is an empirical process. Use the [Prompt Playground](https://platform.openai.com/playground) to experiment with different instructions and tool definitions.