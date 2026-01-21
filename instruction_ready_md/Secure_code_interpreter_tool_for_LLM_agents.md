# Build Your Own Code Interpreter: Dynamic Tool Generation and Execution With o3-mini

## Overview

At the core of providing an LLM Agent with the ability to interact with the outside world is "tool calling," where an LLM can invoke a function with specific arguments. Traditionally, these functions are predefined by the developer. This cookbook explores a more flexible paradigm: **dynamically generating tools** at runtime using an LLM (specifically **o3-mini**) and executing them via a custom code interpreter.

This approach enables open-ended, adaptive problem-solving for tasks like data analysis, visualization, and process automation without requiring every possible function to be pre-coded.

### Why Build Your Own Code Interpreter?

While many API providers offer built-in code interpreters, there are compelling reasons to build your own:

1.  **Language or Library Support**: Your task may require an unsupported language (e.g., C++, Java) or specific libraries.
2.  **Task Compatibility**: Your use case might not align with the provider's built-in solution.
3.  **Model Constraints**: You may need to use a model not supported by the provider's interpreter.
4.  **Cost & File Size**: Provider costs or file size limits may be prohibitive.
5.  **Internal Integration**: The provider's interpreter may not integrate with your internal systems.

### What You'll Learn

By following this guide, you will learn how to:
*   Set up an isolated Python code execution environment using Docker.
*   Configure a custom code interpreter tool for LLM agents.
*   Establish a clear separation of "Agentic" concerns for security.
*   Use the **o3-mini** model to dynamically generate code for data analysis.
*   Orchestrate agents to accomplish a given task efficiently.

### Example Scenario

We will use a sample dataset, "[Key Factors Traffic Accidents](https://www.kaggle.com/datasets/willianoliveiragibin/key-factors-traffic-accidents)", to answer analytical questions. The power of our dynamic approach is that we don't need to predefine functions for questions like:
*   "What factors contribute the most to accident frequency?"
*   "Which areas are at the highest risk of accidents?"
*   "How does traffic fine amount influence the number of accidents?"

The LLM will generate the necessary code to answer these questions on the fly.

---

## Prerequisites

Before you begin, ensure you have the following installed and configured:

1.  **Docker**: Installed and running. [Install Docker](https://www.docker.com/).
2.  **Python**: Installed on your local machine. [Install Python](https://www.python.org/downloads/).
3.  **OpenAI API Key**: Set up as an environment variable or in a `.env` file. [Get an API Key](https://platform.openai.com/docs/api-reference/introduction).

---

## Step 1: Set Up an Isolated Code Execution Environment

We need a secure, isolated environment to execute LLM-generated code. We'll use a Docker container with restricted resource access (e.g., no network).

> **⚠️ A Word of Caution**: Always implement strong guardrails. LLMs can generate harmful code. Isolate the execution environment and avoid running generated code directly on your host machine.

### 1.1 Build the Docker Image

We'll use a custom Dockerfile to create a sandboxed Python environment. The image is based on Python 3.10, uses a non-root user, and pre-installs packages from a `requirements.txt` file. Since the container will have no network access, all required packages must be pre-installed.

Navigate to the directory containing your `Dockerfile` and `requirements.txt` (e.g., `./resources/docker`) and run the build command.

```bash
docker build -t python_sandbox:latest ./resources/docker
```

### 1.2 Run the Container in Restricted Mode

Once the image is built, run the container with strict security policies:
*   `--network none`: Disables all network access.
*   `--cap-drop all`: Drops all Linux capabilities.
*   `--pids-limit 64`: Limits the number of processes.
*   `--tmpfs /tmp:rw,size=64M`: Uses a temporary, in-memory filesystem for `/tmp`.

```bash
docker run -d \
  --name sandbox \
  --network none \
  --cap-drop all \
  --pids-limit 64 \
  --tmpfs /tmp:rw,size=64M \
  python_sandbox:latest \
  sleep infinity
```

### 1.3 Verify the Container is Running

Check that your container is active.

```bash
docker ps
```

You should see your `sandbox` container listed with status `Up`.

---

## Step 2: Define and Test the Agents

We will define two agents with distinct responsibilities to enforce security through separation of concerns.

| Agent | Role | Tool Calling Type | Host File System Access | Docker File System Access | Code Interpreter Access |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Agent 1: File Access Agent** | Reads files from the host and provides context. | Pre-defined Tools | **Yes** | Yes | No |
| **Agent 2: Python Code Exec Agent** | Generates & executes Python code to answer questions. | Dynamically Generated Tools | No | **Yes** | **Yes** |

This separation prevents the code-executing agent from directly accessing or modifying the host machine.

### 2.1 Core Architecture

The application uses a set of core classes for consistency (defined in `resources/object_oriented_agents/core_classes`):
*   **`BaseAgent`**: An abstract base class enforcing common methods like `task()`.
*   **`ChatMessages`**: Manages conversation history for the stateless Chat Completions API.
*   **`ToolManager`**: Manages the tools an agent can call.
*   **`ToolInterface`**: An abstract class ensuring all tools have a consistent interface.

### 2.2 Define Agent 1: FileAccessAgent

The `FileAccessAgent` is a concrete implementation of `BaseAgent`. Its purpose is to safely read files from the host system and make them available inside the Docker container.

**Key Properties:**
*   **Model**: Uses `gpt-4o` for reliable tool calling and reasoning.
*   **Tool**: Binds the `FileAccessTool`, which implements the `ToolInterface`.
*   **Method**: Its `task()` method calls the `FileAccessTool` to read a file and copy it into the container.

### 2.3 Define Agent 2: PythonExecAgent

The `PythonExecAgent` is the core of our dynamic tool generation. It receives context from Agent 1, generates Python code to answer the user's question, and executes it within the Docker sandbox.

**Key Properties:**
*   **Model**: Uses `o3-mini`, which excels at STEM tasks and code generation.
*   **Reasoning Effort**: Set to `'high'` for more complete reasoning on complex tasks.
*   **Tool**: Binds the `PythonExecTool`, which handles code execution inside the container.
*   **Method**: Its `task()` method calls the OpenAI API, which can dynamically generate a function call to the `PythonExecTool`.

---

## Step 3: Set Up Agentic Orchestration

With the agents defined, we create an orchestration loop. This loop:
1.  Takes a user's question.
2.  Calls the `FileAccessAgent` to read the relevant data file and provide context.
3.  Passes the context and question to the `PythonExecAgent`.
4.  The `PythonExecAgent` uses `o3-mini` to generate and execute the necessary Python code.
5.  Returns the result to the user.

### 3.1 Import and Instantiate the Agents

Let's set up the agents with the necessary prompts. We'll analyze the `traffic_accidents.csv` file.

```python
# Import the agents
from resources.registry.agents.file_access_agent import FileAccessAgent
from resources.registry.agents.python_code_exec_agent import PythonExecAgent

# Define the context prompt describing the dataset
prompt = """Use the file traffic_accidents.csv for your analysis. The column names are:
Variable	Description
accidents	Number of recorded accidents, as a positive integer.
traffic_fine_amount	Traffic fine amount, expressed in thousands of USD.
traffic_density	Traffic density index, scale from 0 (low) to 10 (high).
traffic_lights	Proportion of traffic lights in the area (0 to 1).
pavement_quality	Pavement quality, scale from 0 (very poor) to 5 (excellent).
urban_area	Urban area (1) or rural area (0), as an integer.
average_speed	Average speed of vehicles in km/h.
rain_intensity	Rain intensity, scale from 0 (no rain) to 3 (heavy rain).
vehicle_count	Estimated number of vehicles, in thousands, as an integer.
time_of_day	Time of day in 24-hour format (0 to 24).
"""

print("Setup: ")
print(prompt)
print("Setting up the agents... ")

# Instantiate the agents
file_agent = FileAccessAgent()
code_agent = PythonExecAgent()
```

### 3.2 Run the Orchestration Loop

Now, we can create a simple loop to process user questions. The user can type `exit` to quit.

```python
def main():
    print("\n--- Dynamic Code Interpreter Agent ---")
    print("Type your question about the traffic accident data.")
    print("Type 'exit' to quit.\n")

    while True:
        # Get user input
        user_question = input("Your question: ").strip()

        if user_question.lower() == 'exit':
            print("Exiting. Goodbye!")
            break

        if not user_question:
            continue

        print("\nProcessing...")

        # Step 1: Use FileAccessAgent to get file context
        print("[Step 1/2] File Access Agent reading data...")
        file_context = file_agent.task(instruction=f"Provide the context for the file 'traffic_accidents.csv' to answer this question: {user_question}")

        # Step 2: Use PythonExecAgent to generate and execute code
        print("[Step 2/2] Python Exec Agent generating and running code...")
        final_answer = code_agent.task(
            context=file_context,
            user_question=user_question
        )

        # Display the result
        print("\n" + "="*50)
        print("RESULT:")
        print(final_answer)
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
```

### 3.3 Example Execution

When you run the script and ask a question like **"What factors contribute the most to accident frequency?"**, the orchestration works as follows:

1.  The `FileAccessAgent` reads `traffic_accidents.csv` and prepares a context summary.
2.  This context and your question are sent to the `PythonExecAgent`.
3.  The `o3-mini` model reasons about the task, dynamically generates a Python script (e.g., to perform a feature importance analysis using a Random Forest), and calls the `PythonExecTool`.
4.  The `PythonExecTool` runs this generated script inside the isolated Docker container.
5.  The output (e.g., a list of top contributing factors) is returned and displayed.

You have now built an agentic application capable of dynamically generating and safely executing code to solve open-ended data analysis tasks.