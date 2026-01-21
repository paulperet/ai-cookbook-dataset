# Optimize Prompts: A Multi-Agent System for AI Prompt Refinement

Crafting effective prompts is a critical skill when working with AI models. Even experienced users can inadvertently introduce contradictions, ambiguities, or inconsistencies that lead to suboptimal results. This guide demonstrates how to build a system that identifies and fixes common issues, resulting in more reliable and effective prompts.

The optimization process uses a collaborative multi-agent approach where specialized AI agents analyze and rewrite prompts. You can use this code to optimize your own prompts or explore the feature in the [OpenAI Playground](https://platform.openai.com/playground/prompts).

## Prerequisites

Before you begin, ensure you have the following:

1.  **Python Environment**: A working Python installation.
2.  **Required Packages**: Install the necessary Python libraries.
3.  **API Key**: An OpenAI API key set in your environment variables.

Run the following setup commands:

```bash
pip install openai openai-agents pydantic
```

```python
import os
# Set your OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

## Step 1: System Overview

The prompt optimization system uses five specialized agents that work together:

1.  **Dev-Contradiction-Checker**: Identifies logical contradictions within the prompt instructions.
2.  **Format-Checker**: Ensures structured output requirements (JSON, CSV, etc.) are clearly specified.
3.  **Few-Shot-Consistency-Checker**: Verifies that example responses actually follow the rules in the prompt.
4.  **Dev-Rewriter**: Rewrites the prompt to resolve contradictions and clarify format specifications.
5.  **Few-Shot-Rewriter**: Updates inconsistent example responses to align with the optimized prompt.

## Step 2: Define Data Models

We'll use Pydantic models to structure data and ensure consistency throughout the workflow. Create these models in your Python script:

```python
from enum import Enum
from typing import Any, List, Dict
from pydantic import BaseModel, Field

class Role(str, Enum):
    """Role enum for chat messages."""
    user = "user"
    assistant = "assistant"

class ChatMessage(BaseModel):
    """Single chat message used in few-shot examples."""
    role: Role
    content: str

class Issues(BaseModel):
    """Structured output returned by checkers."""
    has_issues: bool
    issues: List[str]
    
    @classmethod
    def no_issues(cls) -> "Issues":
        return cls(has_issues=False, issues=[])

class FewShotIssues(Issues):
    """Output for few-shot contradiction detector including optional rewrite suggestions."""
    rewrite_suggestions: List[str] = Field(default_factory=list)
    
    @classmethod
    def no_issues(cls) -> "FewShotIssues":
        return cls(has_issues=False, issues=[], rewrite_suggestions=[])

class MessagesOutput(BaseModel):
    """Structured output returned by `rewrite_messages_agent`."""
    messages: list[ChatMessage]

class DevRewriteOutput(BaseModel):
    """Rewriter returns the cleaned-up developer prompt."""
    new_developer_message: str
```

## Step 3: Initialize the OpenAI Client

Set up the OpenAI client that all agents will use:

```python
from openai import AsyncOpenAI
from agents import set_default_openai_client

openai_client: AsyncOpenAI | None = None

def _get_openai_client() -> AsyncOpenAI:
    global openai_client
    if openai_client is None:
        openai_client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    return openai_client

set_default_openai_client(_get_openai_client)
```

## Step 4: Create the Specialized Agents

Now, let's define each agent with its specific instructions. Notice how each agent has a narrowly defined scope and clear output requirements.

### 4.1 Contradiction Checker Agent

This agent looks for genuine self-contradictions within the prompt:

```python
from agents import Agent

dev_contradiction_checker = Agent(
    name="contradiction_detector",
    model="gpt-4.1",
    output_type=Issues,
    instructions="""
    You are **Dev-Contradiction-Checker**.

    Goal
    Detect *genuine* self-contradictions or impossibilities **inside** the developer prompt.

    Definitions
    • A contradiction = two clauses that cannot both be followed.
    • Overlaps or redundancies are *not* contradictions.

    What you MUST do
    1. Compare every imperative / prohibition against all others.
    2. List at most FIVE contradictions (each as ONE bullet).
    3. If no contradiction exists, say so.

    Output format (**strict JSON**)
    Return **only** an object that matches the `Issues` schema:
    {"has_issues": <bool>, "issues": ["<bullet 1>", "<bullet 2>"]}
    - has_issues = true IFF the issues array is non-empty.
    - Do not add extra keys, comments or markdown.
    """,
)
```

### 4.2 Format Checker Agent

This agent ensures structured output requirements are clearly specified:

```python
format_checker = Agent(
    name="format_checker",
    model="gpt-4.1",
    output_type=Issues,
    instructions="""
    You are Format-Checker.

    Task
    Decide whether the developer prompt requires a structured output (JSON/CSV/XML/Markdown table, etc.).
    If so, flag any missing or unclear aspects of that format.

    Steps
    Categorise the task as:
    a. "conversation_only", or
    b. "structured_output_required".

    For case (b):
    - Point out absent fields, ambiguous data types, unspecified ordering, or missing error-handling.

    Do NOT invent issues if unsure. Be conservative in flagging format issues.

    Output format
    Return strictly-valid JSON following the Issues schema:
    {"has_issues": <bool>, "issues": ["<desc 1>", "..."]}
    Maximum five issues. No extra keys or text.
    """,
)
```

### 4.3 Few-Shot Consistency Checker Agent

This agent verifies that example responses follow the prompt rules:

```python
fewshot_consistency_checker = Agent(
    name="fewshot_consistency_checker",
    model="gpt-4.1",
    output_type=FewShotIssues,
    instructions="""
    You are FewShot-Consistency-Checker.

    Goal
    Find conflicts between the DEVELOPER_MESSAGE rules and the accompanying **assistant** examples.

    Method
    Extract key constraints from DEVELOPER_MESSAGE:
    - Tone / style
    - Forbidden or mandated content
    - Output format requirements

    Compliance Rubric - read carefully
    Evaluate only what the developer message makes explicit.

    Objective constraints you must check when present:
    - Required output type syntax (e.g., "JSON object", "single sentence", "subject line").
    - Hard limits (length ≤ N chars, language required to be English, forbidden words, etc.).
    - Mandatory tokens or fields the developer explicitly names.

    Out-of-scope (DO NOT FLAG):
    - Whether the reply "sounds generic", "repeats the prompt", or "fully reflects the user's request" - unless the developer text explicitly demands those qualities.
    - Creative style, marketing quality, or depth of content unless stated.
    - Minor stylistic choices (capitalisation, punctuation) that do not violate an explicit rule.

    Pass/Fail rule
    - If an assistant reply satisfies all objective constraints, it is compliant, even if you personally find it bland or loosely related.
    - Only record an issue when a concrete, quoted rule is broken.

    Empty assistant list ⇒ immediately return has_issues=false.

    For each assistant example:
    - USER_EXAMPLES are for context only; never use them to judge compliance.
    - Judge each assistant reply solely against the explicit constraints you extracted from the developer message.
    - If a reply breaks a specific, quoted rule, add a line explaining which rule it breaks.
    - Optionally, suggest a rewrite in one short sentence (add to rewrite_suggestions).
    - If you are uncertain, do not flag an issue.
    - Be conservative—uncertain or ambiguous cases are not issues.

    Output format
    Return JSON matching FewShotIssues:
    {
        "has_issues": <bool>,
        "issues": ["<explanation 1>", "..."],
        "rewrite_suggestions": ["<suggestion 1>", "..."]
    }
    List max five items for both arrays. Provide empty arrays when none.
    No markdown, no extra keys.
    """,
)
```

### 4.4 Developer Rewriter Agent

This agent rewrites the prompt to fix identified issues:

```python
dev_rewriter = Agent(
    name="dev_rewriter",
    model="gpt-4.1",
    output_type=DevRewriteOutput,
    instructions="""
    You are Dev-Rewriter.

    You receive:
    - ORIGINAL_DEVELOPER_MESSAGE
    - CONTRADICTION_ISSUES (may be empty)
    - FORMAT_ISSUES (may be empty)

    Rewrite rules
    Preserve the original intent and capabilities.

    Resolve each contradiction:
    - Keep the clause that preserves the message intent; remove/merge the conflicting one.

    If FORMAT_ISSUES is non-empty:
    - Append a new section titled ## Output Format that clearly defines the schema or gives an explicit example.

    Do NOT change few-shot examples.
    Do NOT add new policies or scope.

    Output format (strict JSON)
    {"new_developer_message": "<full rewritten text>"}
    No other keys, no markdown.
    """,
)
```

### 4.5 Few-Shot Rewriter Agent

This agent updates example responses to comply with the optimized prompt:

```python
fewshot_rewriter = Agent(
    name="fewshot_rewriter",
    model="gpt-4.1",
    output_type=MessagesOutput,
    instructions="""
    You are FewShot-Rewriter.

    Input payload
    - NEW_DEVELOPER_MESSAGE (already optimized)
    - ORIGINAL_MESSAGES (list of user/assistant dicts)
    - FEW_SHOT_ISSUES (non-empty)

    Task
    Regenerate only the assistant parts that were flagged.
    User messages must remain identical.
    Every regenerated assistant reply MUST comply with NEW_DEVELOPER_MESSAGE.

    After regenerating each assistant reply, verify:
    - It matches NEW_DEVELOPER_MESSAGE. ENSURE THAT THIS IS TRUE.

    Output format
    Return strict JSON that matches the MessagesOutput schema:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Guidelines
    - Preserve original ordering and total count.
    - If a message was unproblematic, copy it unchanged.
    """,
)
```

## Step 5: Implement the Optimization Workflow

Now let's create the main function that orchestrates all the agents. This function runs checkers in parallel for efficiency, then applies rewrites if needed:

```python
import asyncio
import json
from agents import Runner, trace

def _normalize_messages(messages: List[Any]) -> List[Dict[str, str]]:
    """Convert list of pydantic message models to JSON-serializable dicts."""
    result = []
    for m in messages:
        if hasattr(m, "model_dump"):
            result.append(m.model_dump())
        elif isinstance(m, dict) and "role" in m and "content" in m:
            result.append({"role": str(m["role"]), "content": str(m["content"])})
    return result

async def optimize_prompt_parallel(
    developer_message: str,
    messages: List["ChatMessage"],
) -> Dict[str, Any]:
    """
    Runs contradiction, format, and few-shot checkers in parallel,
    then rewrites the prompt/examples if needed.
    Returns a unified dict suitable for an API or endpoint.
    """

    with trace("optimize_prompt_workflow"):
        # 1. Run all checkers in parallel
        tasks = [
            Runner.run(dev_contradiction_checker, developer_message),
            Runner.run(format_checker, developer_message),
        ]
        
        # Add few-shot checker only if there are messages
        if messages:
            fs_input = {
                "DEVELOPER_MESSAGE": developer_message,
                "USER_EXAMPLES": [m.content for m in messages if m.role == "user"],
                "ASSISTANT_EXAMPLES": [m.content for m in messages if m.role == "assistant"],
            }
            tasks.append(Runner.run(fewshot_consistency_checker, fs_input))
        else:
            tasks.append(asyncio.sleep(0))  # Placeholder for consistent indexing
        
        # Execute all checkers concurrently
        results = await asyncio.gather(*tasks)
        
        # 2. Extract results
        contradiction_result = results[0]
        format_result = results[1]
        fewshot_result = results[2] if messages else FewShotIssues.no_issues()
        
        # 3. Determine if rewriting is needed
        needs_dev_rewrite = contradiction_result.has_issues or format_result.has_issues
        needs_fewshot_rewrite = fewshot_result.has_issues
        
        new_developer_message = developer_message
        new_messages = messages
        
        # 4. Rewrite developer message if needed
        if needs_dev_rewrite:
            rewrite_input = {
                "ORIGINAL_DEVELOPER_MESSAGE": developer_message,
                "CONTRADICTION_ISSUES": contradiction_result.issues,
                "FORMAT_ISSUES": format_result.issues,
            }
            rewrite_result = await Runner.run(dev_rewriter, rewrite_input)
            new_developer_message = rewrite_result.new_developer_message
        
        # 5. Rewrite few-shot examples if needed
        if needs_fewshot_rewrite:
            rewrite_input = {
                "NEW_DEVELOPER_MESSAGE": new_developer_message,
                "ORIGINAL_MESSAGES": _normalize_messages(messages),
                "FEW_SHOT_ISSUES": fewshot_result.issues,
            }
            rewrite_result = await Runner.run(fewshot_rewriter, rewrite_input)
            new_messages = rewrite_result.messages
        
        # 6. Prepare final output
        return {
            "changes": needs_dev_rewrite or needs_fewshot_rewrite,
            "new_developer_message": new_developer_message,
            "new_messages": _normalize_messages(new_messages),
            "contradiction_issues": contradiction_result.issues,
            "few_shot_contradiction_issues": fewshot_result.issues,
            "format_issues": format_result.issues,
            "general_improvements": []  # Could be extended for other improvements
        }
```

## Step 6: Test the System with Examples

Let's test the optimization system with some real examples. Create a test function to see the system in action:

```python
async def test_optimization():
    """Test the optimization system with example prompts."""
    
    # Example 1: Contradiction in language instructions
    print("=== Example 1: Language Contradiction ===")
    developer_message = "Always answer in **English**.\nNunca respondas en inglés."
    messages = [ChatMessage(role="user", content="¿Qué hora es?")]
    
    result = await optimize_prompt_parallel(developer_message, messages)
    print(f"Original: {developer_message}")
    print(f"Optimized: {result['new_developer_message']}")
    print(f"Contradiction issues found: {result['contradiction_issues']}")
    print()
    
    # Example 2: Few-shot inconsistency
    print("=== Example 2: Few-Shot Inconsistency ===")
    developer_message = "Respond with **only 'yes' or 'no'** – no explanations."
    messages = [
        ChatMessage(role="user", content="Is the sky blue?"),
        ChatMessage(role="assistant", content="Yes, because wavelengths …"),
        ChatMessage(role="user", content="Is water wet?"),
        ChatMessage(role="assistant", content="Yes."),
    ]
    
    result = await optimize_prompt_parallel(developer_message, messages)
    print(f"Original: {developer_message}")
    print(f"Optimized: {result['new_developer_message']}")
    print(f"Few-shot issues found: {result['few_shot_contradiction_issues']}")
    print(f"Updated examples:")
    for msg in result['new_messages']:
        print(f"  {msg['role']}: {msg['content']}")
    print()
    
    # Example 3: Missing format specification
    print("=== Example 3: Missing Format Specification ===")
    developer_message = "Extract the person's name, age, and email from the text."
    messages = []
    
    result = await optimize_prompt_parallel(developer_message, messages)
    print(f"Original: {developer_message}")
    print(f"Optimized: {result['new_developer_message']}")
    print(f"Format issues found: {result['format_issues']}")

# Run the tests
if __name__ == "__main__":
    asyncio.run(test_optimization())
```

## Step 7: Run the Complete Workflow

To use the optimization system with your own prompts, create a simple wrapper function:

```python
async def optimize_your_prompt(developer_message: str, examples: List[Dict] = None):
    """
    Optimize your prompt with optional few-shot examples.
    
    Args:
        developer_message: The prompt you want to optimize
        examples: Optional list of example conversations in the format
                 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    
    Returns:
        Dictionary containing optimized prompt and messages
    """
    if examples is None:
        examples = []
    
    # Convert examples to ChatMessage objects
    chat_messages = []
    for example in examples:
        chat_messages.append(ChatMessage(role=example["role"], content=example["content"]))
    
    # Run optimization
    result = await optimize_prompt_parallel(developer_message, chat_messages)
    
    return result

# Example usage
async def main():
    # Your prompt with a contradiction
    my_prompt = "Provide detailed explanations. Keep responses brief and concise."
    
    # Optional few-shot examples
    my_examples = [
        {"role": "user", "content": "