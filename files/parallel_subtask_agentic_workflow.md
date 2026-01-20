# Parallel Subtask Agent Workflow

## Introduction

This notebook demonstrates how to create a generic agent workflow that automatically breaks complex tasks into multiple subtasks.

These subtasks are completed using parallel MistralAI LLM calls, enhanced with real-time information from Tavily API.

The results are then synthesized into a comprehensive response.

## Workflow Overview

1. An orchestrator LLM analyzes the main task and breaks it into distinct, parallel subtasks
2. Each subtask is assigned to a worker LLM with specialized instructions
3. Workers execute in parallel, using Tavily API for up-to-date information as needed
4. Results are synthesized into a unified response

**NOTE**: We will use MistralAI’s LLM for subtask handling and response synthesis, and the Tavily API to retrieve up-to-date real.

## Solution Architecture

### Installation


```python
!pip install -U mistralai
```

[First Entry, ..., Last Entry]

### Imports


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

### Set your API keys

Here we set the API keys for `MistralAI` and `Tavily`. You can obtain the keys from the following links:

1. MistralAI: https://console.mistral.ai/api-keys
2. Tavily: https://app.tavily.com/home


```python
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "<YOUR MISTRAL API KEY>") # Get it from https://console.mistral.ai/api-keys
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "<YOUR TAVILY API KEY>") # Get it from https://app.tavily.com/home
```

### Initialize Mistral client


```python
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
MISTRAL_MODEL = "mistral-small-latest"  # Can be configured based on needs
```

### Tavily API configuration


```python
TAVILY_API_URL = "https://api.tavily.com/search"
TAVILY_HEADERS = {
    "Authorization": f"Bearer {TAVILY_API_KEY}",
    "Content-Type": "application/json"
}
```

### Pydantic Models for Structured Data

Pydantic models provide data validation and serialization, ensuring the data we receive from LLMs matches our expected structure. This helps maintain consistency between the orchestrator and worker components.

**SubTask:** Individual subtask definition - defines a discrete unit of work with its type, description, and optional search query.

**TaskList:** Output structure from the orchestrator - contains analysis and a list of defined subtasks to be executed in parallel.


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

### API Utility Functions

API Utility functions handle communication with external APIs and process the responses, providing clean interfaces for the rest of the workflow.

**fetch_information:** Retrieves relevant information from Tavily API based on a query and returns structured results.

**run_mistral_llm:** Executes a standard call to Mistral AI with given prompts, returning the generated content.

**parse_structured_output:** Uses Mistral's structured output capability to generate and parse responses according to Pydantic models.


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

### Async Worker Functions

These functions enable parallel execution of subtasks, allowing the workflow to process multiple components simultaneously for greater efficiency.

**run_task_async:** Executes a single subtask asynchronously, enhancing it with relevant information from Tavily when needed.

**execute_tasks_in_parallel:** Manages the parallel execution of all subtasks, ensuring they run concurrently and their results are properly collected.


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

## Main Workflow Function

The primary orchestration function that coordinates the entire parallel subtask process from initial request to final synthesized response.

**parallel_subtask_workflow:** Manages the complete workflow by orchestrating task decomposition, parallel execution of subtasks, and final synthesis of results into a comprehensive response.

Steps:

1. **Task Analysis:** The orchestrator analyzes the user's query and breaks it into distinct subtasks

2. **Subtask Definition:** Each subtask is defined with a unique ID, type, description, and search query

3. **Parallel Execution:** Subtasks are executed concurrently by worker agents
Information Enhancement: Workers retrieve relevant information from Tavily when needed

4. **Result Collection:** Outputs from all workers are gathered

5. **Synthesis:** Individual results are combined into a comprehensive final response

6. **Final Response:** Complete workflow results are returned, including both individual analyses and the synthesized answer


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

### Run workflow with an example task

Here we run the worklow with a sample example task comparing mobile phones recommendation


```python
task = "Compare the iPhone 16 Pro, iPhone 15 Pro, and Google Pixel 9 Pro, and recommend which one I should purchase."
```


```python
result = asyncio.run(workflow(task))
```

[First Entry, ..., Last Entry]

### Final Response


```python
print("\n=== FINAL SYNTHESIZED RESPONSE ===")

display(Markdown(result["final_response"]))
```

### Comprehensive Comparison and Recommendation: iPhone 16 Pro, iPhone 15 Pro, and Google Pixel 9 Pro

When comparing the iPhone 16 Pro, iPhone 15 Pro, and Google Pixel 9 Pro, several key factors stand out, including performance, camera quality, user reviews, and pricing. Here's a synthesized analysis to help you make an informed decision:

#### Performance
- **Processor**: The iPhone 16 Pro, powered by the Apple A18 Pro chipset, offers the best performance with up to 15% faster CPU and 20% faster graphics compared to the iPhone 15 Pro's A17 Pro chipset. The Google Pixel 9 Pro's Tensor G4 processor underperforms in benchmarks, making it the least powerful option.
- **RAM**: Both iPhone models come with 8GB of RAM, while the Pixel 9 Pro is expected to have around 8GB or 12GB. This ensures smooth multitasking across all devices.
- **Battery Life**: The iPhone 16 Pro benefits from improved battery efficiency and heat management, likely providing better battery life compared to the iPhone 15 Pro and potentially the Pixel 9 Pro, despite the Pixel's larger battery capacity.

#### Camera Quality
- **iPhone 16 Pro**: Excels in low-light performance, telephoto clarity, and offers advanced video capabilities like ProRes and Cinematic modes. It supports RAW photography and extensive manual controls.
- **iPhone 15 Pro**: Good camera performance but lacks the advanced telephoto capabilities and low-light prowess of the iPhone 16 Pro.
- **Google Pixel 9 Pro**: Competitive camera specs but generally doesn't match the iPhone 16 Pro in practical image quality, particularly in low-light conditions and telephoto performance.

#### User Reviews and Ratings
- **iPhone 16 Pro**: Highly rated for performance, camera quality, and battery life. Users appreciate the seamless integration of hardware and software.
- **iPhone 15 Pro**: Well-received for performance and camera improvements, but some users find the incremental upgrades less compelling.
- **Google Pixel 9 Pro**: Praised for camera and display quality but criticized for processor performance. Users appreciate the software experience but note it lags behind the iPhone 16 Pro in performance.

#### Pricing and Value
- **Base Model (128GB)**: All three models are priced at $999.
- **256GB Model**: The iPhone models are priced at $1,099, while the Pixel 9 Pro is $1,199.
- **512GB and 1TB Models**: The iPhone 16 Pro offers more storage options at competitive prices.
- **Display**: The Pixel 9 Pro has a slightly better display with higher brightness, but the iPhone 16 Pro offers better color accuracy and minimum brightness.
- **Software and Features**: The iPhone 16 Pro offers free Apple Intelligence features, while the Pixel 9 Pro comes with one year of Gemini Advanced (free for the first year, then $20/month).

#### Recommendation
- **For Performance and Overall User Experience**: The **iPhone 16 Pro** is the clear winner. It offers superior CPU and graphics performance, better battery management, and excellent camera capabilities. User reviews and ratings consistently praise its performance, camera quality, and battery life.
- **For Budget-Conscious Users**: The **iPhone 15 Pro** is a solid option if you don't need the latest features of the iPhone 16 Pro. It offers good performance and camera quality at a potentially more affordable price point.
- **For Camera and Display Enthusiasts**: The **Google Pixel 9 Pro** is a strong contender if you prioritize display brightness and camera specs. However, be aware of its underperforming processor and potential long-term costs for software features.

In conclusion, if you prioritize performance, camera quality, and overall user experience, the **iPhone 16 Pro** is the best choice. If you are looking for a more budget-friendly option with still excellent performance, consider the **iPhone 15 Pro**. For those who value display and camera specs above all else, the **Google Pixel 9 Pro** might be the way to go, but be prepared for potential performance trade-offs.

### Examining Orchestrator Analysis, Subtask information and responses

We can examine the Orchestrator Analysis, subtasks created, the corresponding search queries, and the individual responses.


```python
print("\n=== Orchestrator Analysis ===\n")

display(Markdown(result['orchestrator_analysis']))
```

To effectively compare the iPhone 16 Pro, iPhone 15 Pro, and Google Pixel 9 Pro, and recommend the best purchase, we need to break down the task into specialized subtasks that cover different aspects of the comparison. These aspects include performance, camera quality, user reviews, and pricing. Each subtask will focus on gathering and analyzing specific information to provide a comprehensive comparison.



```python
print("\n=== SUBTASKS CREATED ===\n")

for subtask in result['subtasks']:
    display(Markdown(f"- {subtask['task_id']}: \n - Task type: {subtask['type']} \n - Task Description: - {subtask['description'][:100]} \n - search_query - {subtask['search_query']}"))
```

- TASK001: 
 - Task type: Performance Analysis 
 - Task Description: - Compare the performance specifications of the iPhone 16 Pro, iPhone 15 Pro, and Google Pixel 9 Pro,  
 - search_query - iPhone 16 Pro vs iPhone 15 Pro vs Google Pixel 9 Pro performance comparison

- TASK002: 
 - Task type: Camera Quality Assessment 
 - Task Description: - Evaluate the camera specifications and capabilities of the iPhone 16 Pro, iPhone 15 Pro, and Google  
 - search_query - iPhone 16 Pro vs iPhone 15 Pro vs Google Pixel 9 Pro camera comparison

- TASK003: 
 - Task type: User Reviews and Ratings 
 - Task Description: - Gather and analyze user reviews and ratings for the iPhone 16 Pro, iPhone 15 Pro, and Google Pixel 9 
 - search_query - iPhone 16 Pro vs iPhone 15 Pro vs Google Pixel 9 Pro user reviews

- TASK004: 
 - Task type: Pricing and Value Analysis 
 - Task Description: - Compare the pricing of the iPhone 16 Pro, iPhone 15 Pro