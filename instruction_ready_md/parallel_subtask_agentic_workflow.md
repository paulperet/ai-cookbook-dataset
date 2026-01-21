# Parallel Subtask Agent Workflow: A Step-by-Step Guide

## Introduction

This guide demonstrates how to create a generic agent workflow that automatically breaks complex tasks into multiple subtasks. These subtasks are completed using parallel MistralAI LLM calls, enhanced with real-time information from the Tavily API. The results are then synthesized into a comprehensive response.

### Workflow Overview
1.  An orchestrator LLM analyzes the main task and breaks it into distinct, parallel subtasks.
2.  Each subtask is assigned to a worker LLM with specialized instructions.
3.  Workers execute in parallel, using the Tavily API for up-to-date information as needed.
4.  Results are synthesized into a unified response.

**Note:** We will use MistralAI's LLM for subtask handling and response synthesis, and the Tavily API to retrieve up-to-date, real-time information.

---

## Prerequisites and Setup

### Step 1: Install Required Libraries
First, install the necessary Python packages.

```bash
pip install -U mistralai requests pydantic nest_asyncio
```

### Step 2: Import Dependencies
Import the required modules for the workflow.

```python
import os
import json
import asyncio
import requests
from typing import Any, Optional, Dict, List, Union
from pydantic import Field, BaseModel, ValidationError
from mistralai import Mistral
from IPython.display import display, Markdown

import nest_asyncio
nest_asyncio.apply()
```

### Step 3: Set Your API Keys
Set your API keys for MistralAI and Tavily. You can obtain them from the following links:
1.  **MistralAI:** https://console.mistral.ai/api-keys
2.  **Tavily:** https://app.tavily.com/home

```python
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "<YOUR MISTRAL API KEY>")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "<YOUR TAVILY API KEY>")
```

### Step 4: Initialize the Mistral Client
Initialize the Mistral client with your API key and specify the model to use.

```python
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
MISTRAL_MODEL = "mistral-small-latest"  # Can be configured based on needs
```

### Step 5: Configure the Tavily API
Set up the Tavily API endpoint and headers for making search requests.

```python
TAVILY_API_URL = "https://api.tavily.com/search"
TAVILY_HEADERS = {
    "Authorization": f"Bearer {TAVILY_API_KEY}",
    "Content-Type": "application/json"
}
```

---

## Core Components

### Step 6: Define Pydantic Models for Structured Data
Pydantic models provide data validation and serialization, ensuring the data we receive from LLMs matches our expected structure. This helps maintain consistency between the orchestrator and worker components.

```python
class SubTask(BaseModel):
    """Individual subtask definition"""
    task_id: str
    type: str
    description: str
    search_query: Optional[str]  # Query for Tavily search for the subtask

class TaskList(BaseModel):
    """Structure for orchestrator output"""
    analysis: str
    subtasks: List[SubTask]
```

### Step 7: Create API Utility Functions
These functions handle communication with external APIs and process the responses, providing clean interfaces for the rest of the workflow.

```python
def fetch_information(query: str, max_results: int = 3):
    """Retrieve information from Tavily API"""
    payload = {
        "query": query,
        "search_depth": "advanced",
        "include_answer": True,
        "max_results": max_results
    }

    try:
        response = requests.post(TAVILY_API_URL, json=payload, headers=TAVILY_HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Tavily: {e}")
        return {"error": str(e), "results": []}

def run_mistral_llm(prompt: str, system_prompt: Optional[str] = None):
    """Run Mistral LLM with given prompts"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    response = mistral_client.chat.complete(
        model=MISTRAL_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=4000
    )

    return response.choices[0].message.content

def parse_structured_output(prompt: str, response_format: BaseModel, system_prompt: Optional[str] = None):
    """Get structured output from Mistral LLM based on a Pydantic model"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    response = mistral_client.chat.parse(
        model=MISTRAL_MODEL,
        messages=messages,
        response_format=response_format,
        temperature=0.2
    )

    return json.loads(response.choices[0].message.content)
```

### Step 8: Implement Async Worker Functions
These functions enable parallel execution of subtasks, allowing the workflow to process multiple components simultaneously for greater efficiency.

```python
async def run_task_async(task: SubTask, original_task: str):
    """Execute a single subtask asynchronously with Tavily enhancement"""

    # Prepare context with Tavily information if a search query is provided
    context = ""
    if task.search_query:
        print(f"Fetching information for: {task.search_query}")
        search_results = fetch_information(task.search_query)

        # Format search results into context
        if "results" in search_results and search_results["results"]:
            context = "### Relevant Information:\n"
            for result in search_results["results"]:
                context += f"- {result.get('content', '')}\n"

        if "answer" in search_results and search_results["answer"]:
            context += f"\n### Summary: {search_results['answer']}\n"

    # Worker prompt with task information and context
    worker_prompt = f"""
    Complete the following subtask based on the given information:

    Original Task: {original_task}

    Subtask Type: {task.type}
    Subtask Description: {task.description}

    {context}

    Please provide a detailed response for this specific subtask only.
    """

    # Use asyncio to run in a thread pool to prevent blocking
    return await asyncio.to_thread(
        run_mistral_llm,
        prompt=worker_prompt,
        system_prompt="You are a specialized agent focused on solving a specific aspect of a larger task."
    )

async def execute_tasks_in_parallel(subtasks: List[SubTask], original_task: str):
    """Execute all subtasks in parallel"""
    tasks = []
    for subtask in subtasks:
        tasks.append(run_task_async(subtask, original_task))

    return await asyncio.gather(*tasks)
```

---

## Main Workflow Function

### Step 9: Define the Primary Orchestration Function
This is the primary orchestration function that coordinates the entire parallel subtask process from initial request to final synthesized response.

```python
async def workflow(user_task: str):
    """Main workflow function to process a task through the parallel subtask agent workflow"""

    print("=== USER TASK ===\n")
    print(user_task)

    # Step 1: Orchestrator breaks down the task into subtasks
    orchestrator_prompt = f"""
    Analyze this task and break it down into 3-5 distinct, specialized subtasks that could be executed in parallel:

    Task: {user_task}

    For each subtask:
    1. Assign a unique task_id
    2. Define a specific type that describes the subtask's focus
    3. Write a clear description explaining what needs to be done
    4. Provide a search query if the subtask requires additional information

    First, provide a brief analysis of your understanding of the task.
    Then, define the subtasks that would collectively solve this problem effectively.

    Remember to make the subtasks complementary, not redundant, focusing on different aspects of the problem.
    """

    orchestrator_system_prompt = """
    You are a task orchestrator that specializes in breaking down complex problems into smaller,
    well-defined subtasks that can be solved independently and in parallel. Think carefully about
    the most logical way to decompose the given task.
    """

    print("\nOrchestrating task decomposition...")
    # Get structured output from orchestrator
    task_breakdown = parse_structured_output(
        prompt=orchestrator_prompt,
        response_format=TaskList,
        system_prompt=orchestrator_system_prompt
    )

    # Display orchestrator output
    print("\n=== ORCHESTRATOR OUTPUT ===")
    print(f"\nANALYSIS:\n{task_breakdown['analysis']}")
    print("\nSUBTASKS:")
    for task in task_breakdown["subtasks"]:
        print(f"- {task['task_id']}: {task['type']} - {task['description'][:100]}...")

    # Step 2: Execute subtasks in parallel
    print("\nExecuting subtasks in parallel...")
    subtask_results = await execute_tasks_in_parallel(
        [SubTask(**task) for task in task_breakdown["subtasks"]],
        user_task
    )

    # Display worker results
    for i, (task, result) in enumerate(zip(task_breakdown["subtasks"], subtask_results)):
        print(f"\n=== WORKER RESULT ({task['type']}) ===")
        print(f"{result[:200]}...\n")

    # Step 3: Synthesize final response
    print("\nSynthesizing final response...")

    # Format worker responses for synthesizer
    worker_responses = ""
    for task, response in zip(task_breakdown["subtasks"], subtask_results):
        worker_responses += f"\n=== SUBTASK: {task['type']} ===\n{response}\n"

    synthesizer_prompt = f"""
    Given the following task: {user_task}

    And these responses from different specialized agents focusing on different aspects of the task:

    {worker_responses}

    Please synthesize a comprehensive, coherent response that addresses the original task.
    Integrate insights from all specialized agents while avoiding redundancy.
    Ensure your response is balanced, considering all the different perspectives provided.
    """

    final_response = run_mistral_llm(
        prompt=synthesizer_prompt,
        system_prompt="You are a synthesis agent that combines specialized analyses into comprehensive responses."
    )

    return {
        "orchestrator_analysis": task_breakdown["analysis"],
        "subtasks": task_breakdown["subtasks"],
        "subtask_results": subtask_results,
        "final_response": final_response
    }
```

---

## Running the Workflow

### Step 10: Execute the Workflow with an Example Task
Now, let's run the workflow with a sample task comparing mobile phone recommendations.

```python
task = "Compare the iPhone 16 Pro, iPhone 15 Pro, and Google Pixel 9 Pro, and recommend which one I should purchase."
result = asyncio.run(workflow(task))
```

### Step 11: Display the Final Synthesized Response
After the workflow completes, you can view the final, synthesized recommendation.

```python
print("\n=== FINAL SYNTHESIZED RESPONSE ===")
display(Markdown(result["final_response"]))
```

**Example Output:**
The workflow will generate a comprehensive comparison and recommendation. Here is a condensed example of the final output structure:

> ### Comprehensive Comparison and Recommendation: iPhone 16 Pro, iPhone 15 Pro, and Google Pixel 9 Pro
>
> **Recommendation:** For Performance and Overall User Experience, the **iPhone 16 Pro** is the clear winner. It offers superior CPU and graphics performance, better battery management, and excellent camera capabilities.
>
> **For Budget-Conscious Users:** The **iPhone 15 Pro** is a solid option if you don't need the latest features.
>
> **For Camera and Display Enthusiasts:** The **Google Pixel 9 Pro** is a strong contender, but be aware of its underperforming processor.

---

## Examining the Workflow Internals

### Step 12: Review the Orchestrator's Analysis
You can inspect the orchestrator's initial analysis of the task.

```python
print("\n=== Orchestrator Analysis ===\n")
display(Markdown(result['orchestrator_analysis']))
```

### Step 13: Inspect the Created Subtasks
Examine the subtasks that were generated, including their types, descriptions, and search queries.

```python
print("\n=== SUBTASKS CREATED ===\n")
for subtask in result['subtasks']:
    display(Markdown(f"- **{subtask['task_id']}:** \n - **Type:** {subtask['type']} \n - **Description:** {subtask['description'][:100]}... \n - **Search Query:** {subtask['search_query']}"))
```

## Summary
You have successfully built a parallel subtask agent workflow. This system intelligently decomposes complex queries, executes research and analysis in parallel, and synthesizes the results into a coherent, actionable answer. You can adapt this framework for various analytical and comparative tasks by modifying the prompts and Pydantic models.