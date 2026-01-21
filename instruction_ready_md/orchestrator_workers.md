# Guide: Implementing an Orchestrator-Workers Workflow for Dynamic Task Delegation

## Introduction

When tackling complex tasks with large language models (LLMs), you often need multiple perspectives or specialized outputs. Pre-defining all possible subtasks can be rigid and inefficient. The orchestrator-workers pattern introduces flexibility: a central "orchestrator" LLM analyzes each unique task at runtime and dynamically determines the best subtasks to delegate to specialized "worker" LLMs.

This guide walks you through building a system that generates multiple styles of marketing copy for a product. You'll learn how to implement a two-phase workflow where the orchestrator intelligently breaks down a task and coordinates parallel execution.

### What You'll Build
A Python system that:
1.  Receives a product description request.
2.  Uses an orchestrator LLM to analyze the task and determine valuable content variations.
3.  Delegates specialized writing tasks to worker LLMs.
4.  Returns coordinated results from all workers.

### Prerequisites
-   Python 3.9+
-   An Anthropic API key (set as the environment variable `ANTHROPIC_API_KEY`)
-   Basic knowledge of Python and prompt engineering.

## Understanding the Workflow

This pattern is ideal for tasks where the optimal subtasks cannot be predicted in advance. It operates in two distinct phases:

1.  **Analysis & Planning**: The orchestrator LLM receives the main task, analyzes it, and generates a structured plan (in XML) detailing which specialized subtasks would be most valuable.
2.  **Parallel Execution**: Each defined subtask is sent to a worker LLM, which receives the original context plus its specific instructions. Results are collected and presented.

**Use this pattern when:**
-   Tasks require multiple distinct approaches or perspectives.
-   The optimal subtasks depend on the specific input.
-   You need to compare different strategies or styles.

**Avoid this pattern for:**
-   Simple, single-output tasks (unnecessary complexity).
-   Latency-critical applications (multiple LLM calls add overhead).
-   Predictable, pre-definable subtasks (use simpler parallelization instead).

## Step 1: Project Setup

Begin by installing the required library and setting up your environment.

### 1.1 Install Dependencies
```bash
pip install anthropic
```

### 1.2 Create a Utilities Module (`util.py`)

This implementation relies on helper functions for LLM calls and XML parsing. Create a file named `util.py` with the following content:

```python
import os
import re
from anthropic import Anthropic

def llm_call(prompt: str, system_prompt: str = "", model: str = "claude-3-5-sonnet-20241022") -> str:
    """
    Send a prompt to the Claude API and return the text response.
    Reads the ANTHROPIC_API_KEY from the environment.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text

def extract_xml(text: str, tag: str) -> str:
    """
    Extract content from XML tags using regex.
    Returns the content between <tag> and </tag>, or an empty string if not found.
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""
```

These utilities handle API authentication and provide a clean interface for interacting with Claude and parsing its structured responses.

## Step 2: Implement the Core Orchestrator Class

Now, create the main `FlexibleOrchestrator` class that coordinates the entire workflow.

### 2.1 Define Helper Functions and Class

Create a new Python script (e.g., `orchestrator.py`) and start with the imports and a helper function to parse the orchestrator's XML output.

```python
from util import extract_xml, llm_call

# Model configuration
MODEL = "claude-3-5-sonnet-20241022"

def parse_tasks(tasks_xml: str) -> list[dict]:
    """Parse XML tasks into a list of task dictionaries."""
    tasks = []
    current_task = {}

    for line in tasks_xml.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("<task>"):
            current_task = {}
        elif line.startswith("<type>"):
            current_task["type"] = line[6:-7].strip()
        elif line.startswith("<description>"):
            current_task["description"] = line[12:-13].strip()
        elif line.startswith("</task>"):
            if "description" in current_task:
                if "type" not in current_task:
                    current_task["type"] = "default"
                tasks.append(current_task)

    return tasks
```

### 2.2 Build the `FlexibleOrchestrator` Class

Below the helper function, define the main class. Its `process` method encapsulates the two-phase workflow.

```python
class FlexibleOrchestrator:
    """Break down tasks and run them in parallel using worker LLMs."""

    def __init__(
        self,
        orchestrator_prompt: str,
        worker_prompt: str,
        model: str = MODEL,
    ):
        """Initialize with prompt templates and model selection."""
        self.orchestrator_prompt = orchestrator_prompt
        self.worker_prompt = worker_prompt
        self.model = model

    def _format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with variables."""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required prompt variable: {e}") from e

    def process(self, task: str, context: dict | None = None) -> dict:
        """Process task by breaking it down and running subtasks in parallel."""
        context = context or {}

        # Phase 1: Orchestrator Analysis & Planning
        print("\n" + "=" * 80)
        print("ORCHESTRATOR ANALYSIS")
        print("=" * 80)

        orchestrator_input = self._format_prompt(self.orchestrator_prompt, task=task, **context)
        orchestrator_response = llm_call(orchestrator_input, model=self.model)

        # Parse the orchestrator's response
        analysis = extract_xml(orchestrator_response, "analysis")
        tasks_xml = extract_xml(orchestrator_response, "tasks")
        tasks = parse_tasks(tasks_xml)

        print(f"\n{analysis}\n")

        # Display the identified approaches
        print("\n" + "=" * 80)
        print(f"IDENTIFIED {len(tasks)} APPROACHES")
        print("=" * 80)
        for i, task_info in enumerate(tasks, 1):
            print(f"\n{i}. {task_info['type'].upper()}")
            print(f"   {task_info['description']}")

        # Phase 2: Parallel Worker Execution
        print("\n" + "=" * 80)
        print("GENERATING CONTENT")
        print("=" * 80 + "\n")

        worker_results = []
        for i, task_info in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] Processing: {task_info['type']}...")

            # Prepare the specific prompt for this worker
            worker_input = self._format_prompt(
                self.worker_prompt,
                original_task=task,
                task_type=task_info["type"],
                task_description=task_info["description"],
                **context,
            )

            worker_response = llm_call(worker_input, model=self.model)
            worker_content = extract_xml(worker_response, "response")

            # Validate the worker's response
            if not worker_content or not worker_content.strip():
                print(f"⚠️  Warning: Worker '{task_info['type']}' returned no content")
                worker_content = f"[Error: Worker '{task_info['type']}' failed to generate content]"

            worker_results.append(
                {
                    "type": task_info["type"],
                    "description": task_info["description"],
                    "result": worker_content,
                }
            )

        # Display all results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        for i, result in enumerate(worker_results, 1):
            print(f"\n{'-' * 80}")
            print(f"Approach {i}: {result['type'].upper()}")
            print(f"{'-' * 80}")
            print(f"\n{result['result']}\n")

        return {
            "analysis": analysis,
            "worker_results": worker_results,
        }
```

**Key Design Notes:**
*   **Prompt Templates:** The class uses templates that accept runtime variables (`task`, `context`) for maximum flexibility.
*   **Structured XML:** XML provides a reliable, model-friendly format for parsing complex responses.
*   **Context for Workers:** Each worker receives the original task *and* its specific instructions, ensuring it has full context.
*   **Error Handling:** The validation step catches empty worker outputs, making the system more robust.

## Step 3: Define the Prompts

The success of this pattern hinges on well-crafted prompts. You need two: one for the orchestrator and one for the workers.

### 3.1 Orchestrator Prompt
The orchestrator's job is to analyze and plan. Its prompt instructs it to break down the task and return a structured XML response.

```python
ORCHESTRATOR_PROMPT = """
Analyze this task and break it down into 2-3 distinct approaches:

Task: {task}

Return your response in this format:

<analysis>
Explain your understanding of the task and which variations would be valuable.
Focus on how each approach serves different aspects of the task.
</analysis>

<tasks>
    <task>
    <type>formal</type>
    <description>Write a precise, technical version that emphasizes specifications</description>
    </task>
    <task>
    <type>conversational</type>
    <description>Write an engaging, friendly version that connects with readers</description>
    </task>
</tasks>
"""
```

### 3.2 Worker Prompt
Each worker receives a tailored prompt that includes its specific mission.

```python
WORKER_PROMPT = """
Generate content based on:
Task: {original_task}
Style: {task_type}
Guidelines: {task_description}

Return your response in this format:

<response>
Your content here, maintaining the specified style and fully addressing requirements.
</response>
"""
```

## Step 4: Execute the Workflow

With the class and prompts defined, you can now run the complete orchestrator-workers workflow. This example generates marketing copy for an eco-friendly water bottle.

### 4.1 Instantiate and Run the Orchestrator

Add the following code to your script to create an instance and process a task.

```python
# Instantiate the orchestrator with our prompts
orchestrator = FlexibleOrchestrator(
    orchestrator_prompt=ORCHESTRATOR_PROMPT,
    worker_prompt=WORKER_PROMPT,
)

# Define the task and any additional context
task_description = "Write a product description for a new eco-friendly water bottle"
context = {
    "target_audience": "environmentally conscious millennials",
    "key_features": ["plastic-free", "insulated", "lifetime warranty"],
}

# Execute the workflow
results = orchestrator.process(task=task_description, context=context)
```

### 4.2 Understanding the Output

When you run the script, it will print a detailed log to the console, showing the orchestrator's analysis, the subtasks it created, and the final outputs from each worker. The function also returns a structured dictionary containing all results.

**Example Orchestrator Analysis Output:**
```
================================================================================
ORCHESTRATOR ANALYSIS
================================================================================

This task requires creating marketing copy for an eco-friendly water bottle. The core challenge is balancing product information with persuasive messaging...
1. A **feature-focused technical approach** would appeal to detail-oriented consumers...
2. A **lifestyle-oriented emotional approach** would connect with values-driven consumers...
3. A **benefit-driven practical approach** would focus on solving everyday problems...
```

**Example Identified Approaches:**
```
================================================================================
IDENTIFIED 3 APPROACHES
================================================================================

1. TECHNICAL-SPECIFICATIONS
   Write a detailed, feature-focused description emphasizing materials, construction, environmental certifications...

2. LIFESTYLE-EMOTIONAL
   Write an inspirational, story-driven description that connects the product to environmental values...

3. BENEFIT-PRACTICAL
   Write a problem-solution focused description highlighting everyday usability benefits...
```

The workers will then generate content, and you will see each unique result printed under the **RESULTS** header.

## Conclusion

You have successfully built a dynamic orchestrator-workers system. The key advantage of this pattern is its adaptability: the orchestrator decides *at runtime* what subtasks to create based on the specific input, making it more powerful than pre-defined parallel workflows.

**Next Steps & Customization:**
*   **Adjust Prompts:** Modify `ORCHESTRATOR_PROMPT` and `WORKER_PROMPT` for different tasks (e.g., code review, research summarization, creative brainstorming).
*   **Enhance Error Handling:** Implement retry logic for failed worker calls or more sophisticated XML parsing.
*   **Concurrency:** Use Python's `asyncio` or `concurrent.futures` to make the worker calls truly parallel and reduce total execution time.
*   **Different LLMs:** Experiment with using different models for the orchestrator (e.g., a more powerful, expensive model) and the workers (e.g., faster, cheaper models).

This framework provides a solid foundation for building complex, adaptive AI workflows that can tackle a wide variety of sophisticated tasks.