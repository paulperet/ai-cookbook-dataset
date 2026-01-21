# Evaluator-Optimizer Workflow: A Step-by-Step Guide

This guide demonstrates an Evaluator-Optimizer workflow, where one LLM generates a response and another evaluates it, creating an iterative refinement loop. This pattern is particularly useful when you have clear evaluation criteria and benefit from incremental improvements.

## When to Use This Workflow

Consider this workflow when:
*   You have well-defined evaluation criteria.
*   Iterative refinement adds significant value.
*   An LLM can generate meaningful feedback on its own (or another model's) outputs.
*   Responses can be demonstrably improved with targeted feedback.

## Prerequisites

Ensure you have the necessary utility functions. This guide assumes you have a module `util` with `llm_call` and `extract_xml` functions.

```python
from util import extract_xml, llm_call
```

## Step 1: Define the Core Functions

The workflow is built on three key functions: `generate`, `evaluate`, and the main `loop`.

### 1.1 The Generator Function

The `generate` function creates an initial or revised solution based on a prompt, task, and optional context from previous attempts.

```python
def generate(prompt: str, task: str, context: str = "") -> tuple[str, str]:
    """Generate and improve a solution based on feedback."""
    full_prompt = f"{prompt}\n{context}\nTask: {task}" if context else f"{prompt}\nTask: {task}"
    response = llm_call(full_prompt)
    thoughts = extract_xml(response, "thoughts")
    result = extract_xml(response, "response")

    print("\n=== GENERATION START ===")
    print(f"Thoughts:\n{thoughts}\n")
    print(f"Generated:\n{result}")
    print("=== GENERATION END ===\n")

    return thoughts, result
```

### 1.2 The Evaluator Function

The `evaluate` function assesses a generated solution against the original task and criteria, providing a pass/fail status and constructive feedback.

```python
def evaluate(prompt: str, content: str, task: str) -> tuple[str, str]:
    """Evaluate if a solution meets requirements."""
    full_prompt = f"{prompt}\nOriginal task: {task}\nContent to evaluate: {content}"
    response = llm_call(full_prompt)
    evaluation = extract_xml(response, "evaluation")
    feedback = extract_xml(response, "feedback")

    print("=== EVALUATION START ===")
    print(f"Status: {evaluation}")
    print(f"Feedback: {feedback}")
    print("=== EVALUATION END ===\n")

    return evaluation, feedback
```

### 1.3 The Orchestration Loop

The `loop` function ties everything together. It repeatedly generates a solution, evaluates it, and uses the feedback to generate an improved version until the evaluation passes.

```python
def loop(task: str, evaluator_prompt: str, generator_prompt: str) -> tuple[str, list[dict]]:
    """Keep generating and evaluating until requirements are met."""
    memory = []
    chain_of_thought = []

    thoughts, result = generate(generator_prompt, task)
    memory.append(result)
    chain_of_thought.append({"thoughts": thoughts, "result": result})

    while True:
        evaluation, feedback = evaluate(evaluator_prompt, result, task)
        if evaluation == "PASS":
            return result, chain_of_thought

        context = "\n".join(
            ["Previous attempts:", *[f"- {m}" for m in memory], f"\nFeedback: {feedback}"]
        )

        thoughts, result = generate(generator_prompt, task, context)
        memory.append(result)
        chain_of_thought.append({"thoughts": thoughts, "result": result})
```

## Step 2: Apply the Workflow to a Coding Task

Let's see the workflow in action by implementing a `MinStack` class with O(1) operations.

### 2.1 Define the Prompts and Task

First, we craft the prompts for the evaluator and generator, and define the user's task.

```python
evaluator_prompt = """
Evaluate this following code implementation for:
1. code correctness
2. time complexity
3. style and best practices

You should be evaluating only and not attemping to solve the task.
Only output "PASS" if all criteria are met and you have no further suggestions for improvements.
Output your evaluation concisely in the following format.

<evaluation>PASS, NEEDS_IMPROVEMENT, or FAIL</evaluation>
<feedback>
What needs improvement and why.
</feedback>
"""

generator_prompt = """
Your goal is to complete the task based on <user input>. If there are feedback
from your previous generations, you should reflect on them to improve your solution

Output your answer concisely in the following format:

<thoughts>
[Your understanding of the task and feedback and how you plan to improve]
</thoughts>

<response>
[Your code implementation here]
</response>
"""

task = """
<user input>
Implement a Stack with:
1. push(x)
2. pop()
3. getMin()
All operations should be O(1).
</user input>
"""
```

### 2.2 Execute the Loop

Now, run the main loop with the defined components.

```python
final_result, chain_of_thought = loop(task, evaluator_prompt, generator_prompt)
```

## Step 3: Observe the Iterative Refinement

The loop will run, printing the generation and evaluation steps. Here is a summary of the first two cycles:

**Cycle 1: Initial Generation & Evaluation**
*   **Generation:** The LLM produces a correct `MinStack` algorithm using a secondary stack to track minimums.
*   **Evaluation:** The evaluator returns `NEEDS_IMPROVEMENT`, citing missing error handling, type hints, documentation, and input validation, though it confirms the core algorithm and time complexity are correct.

**Cycle 2: Improved Generation**
*   **Generation:** Reflecting on the feedback, the LLM produces a second version. This version adds:
    *   Proper exceptions (`IndexError`, `TypeError`).
    *   Full type hints and descriptive docstrings.
    *   Input validation in the `push` method.
*   The loop would continue if this version also failed evaluation, but in this case, let's assume it passes on the next cycle.

This workflow automates a rigorous review and refinement process, ensuring the final output meets a high standard of correctness, efficiency, and code quality.