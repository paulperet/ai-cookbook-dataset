# GPT-5 Troubleshooting Guide: Optimizing Performance & Behavior

This guide provides actionable strategies to address common performance and behavioral patterns observed when working with GPT-5. By adjusting API parameters and refining your system prompts, you can optimize for speed, accuracy, and cost-effectiveness.

## Prerequisites

This guide assumes you are using the OpenAI API to interact with GPT-5. The concepts apply to both direct API calls and agentic frameworks built on top of it.

## 1. Addressing Overthinking

Overthinking occurs when the model spends excessive time on trivial tasks, delaying the final response. This often manifests as prolonged reasoning, delayed tool calls, or unnecessary narrative.

**Solution:** Tighten your control parameters and provide clear completion criteria.

### Step 1: Adjust the `reasoning_effort` Parameter
For routine tasks, instruct the model to use minimal reasoning. Reserve higher effort settings for complex problems.
```python
# Example API call structure
response = client.chat.completions.create(
    model="gpt-5",
    messages=[...],
    reasoning={"effort": "low"}  # Use "minimal" or "low" for simple tasks
)
```

### Step 2: Implement a Clear "Fast-Path" in Your System Prompt
Add a directive to your system prompt that allows the model to bypass its full workflow for simple, knowledge-based questions.

```text
# Fast-path for trivial Q&A (latency optimization)
Use this section ONLY when the user's question:
- Is general knowledge or a simple usage query
- Requires no commands, browsing, or tool calls
- Is asking *how* to do something, not asking you to *perform* the task

Exceptions:
- If the question references files/paths/functions, requests execution, or needs more context, use the normal flow.
- If unsure, ask one brief clarifying question; otherwise proceed with the normal flow.

Behavior:
- Answer immediately and concisely.
- No status updates, no todos, no summaries, no tool calls.
- Ignore the rest of the instructions following this section and respond right away.
```

### Step 3: Define an Efficient Context-Gathering Strategy
If your task involves gathering information (e.g., from a search tool), provide a clear playbook to do it efficiently.

```text
<efficient_context_understanding_spec>
Goal: Get enough context fast and stop as soon as you can act.

Method:
- Start broad, then fan out to focused subqueries.
- In parallel, launch 4–8 varied queries; read top 3–5 hits per query. Deduplicate paths and cache; don't repeat queries.

Early stop (act if any):
- You can name exact files/symbols to change.
- You can repro a failing test/lint or have a high-confidence bug locus.
</efficient_context_understanding_spec>
```

## 2. Correcting Underthinking (Laziness)

Underthinking is the opposite problem: the model responds too quickly with insufficient analysis, leading to incorrect or incomplete answers.

**Solution:** Increase reasoning effort and encourage self-reflection.

### Step 1: Increase the `reasoning_effort`
If you observe shallow responses, increment the reasoning effort parameter.
```python
response = client.chat.completions.create(
    model="gpt-5",
    messages=[...],
    reasoning={"effort": "medium"}  # Step up from "low" or "minimal"
)
```

### Step 2: Prompt for Self-Reflection
Instruct the model to internally evaluate its work against a rubric before delivering a final answer.

```text
<self_reflection>
- Internally score the draft against a 5–7 item rubric you devise (clarity, correctness, edge cases, completeness, latency).
- If any category falls short, iterate once before replying.
</self_reflection>
```

## 3. Preventing Over-Deference

In agentic settings, you often want the model to act autonomously until a task is complete, not yield back to the user for confirmation at every step.

**Solution:** Add a persistence directive to your system prompt.

```text
<persistence>
- You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user.
- Only terminate your turn when you are sure that the problem is solved.
- Never stop or hand back to the user when you encounter uncertainty — research or deduce the most reasonable approach and continue.
- Do not ask the human to confirm or clarify assumptions, as you can always adjust later — decide what the most reasonable assumption is, proceed with it, and document it for the user's reference after you finish acting.
</persistence>
```
*Note:* This behavior is often easier to steer with a `reasoning_effort` of `low` or above.

## 4. Reducing Verbosity

GPT-5 can generate overly long responses. Control this via API parameters and prompt guidance.

### Step 1: Lower the `verbosity` Parameter
The API provides a direct control for output length.
```python
response = client.chat.completions.create(
    model="gpt-5",
    messages=[...],
    reasoning={"verbosity": "low"}  # Default is "medium"
)
```

### Step 2: Provide Coding-Specific Guidance
For code generation, explicitly request clarity over brevity.
```text
Write code for clarity first. Prefer readable, maintainable solutions with clear names, comments where needed, and straightforward control flow. Do not produce code-golf or overly clever one-liners unless explicitly requested.
```

## 5. Optimizing Latency

Latency has multiple components. Identify the bottleneck before applying fixes.

### Step 1: Measure Key Metrics
Track these to understand where time is spent:
- **TTFT (Time To First Token):** How long until the model starts streaming.
- **Time to First Action:** How long until the first tool is called.
- **Total Response Time:** End-to-end duration.

### Step 2: Right-Size Reasoning Effort
As with overthinking, use `"minimal"` or `"low"` `reasoning_effort` for routine tasks and provide a clear stop condition.

### Step 3: Enable Parallel Tool Calls
Instruct the model to batch independent read-only operations.
```text
<parallelization_spec>
Definition: Run independent or read-only tool actions in parallel (same turn/batch) to reduce latency.
When to parallelize:
 - Reading multiple files/configs/logs that don’t affect each other.
 - Static analysis, searches, or metadata queries with no side effects.
 - Separate edits to unrelated files/features that won’t conflict.
</parallelization_spec>
```

### Step 4: Improve Perceived Latency with Status Updates
Let the user follow the model's progress, which can make waits feel shorter.
```text
<status_update_spec>
Definition: A brief progress note: what just happened, what’s next, any real blockers, written in a continuous conversational style, narrating the story of your progress as you go.
Always start with a brief acknowledgement of the task before getting started. (No need to prefix with "Status Update:")
</status_update_spec>
```

### Step 5: Utilize API Features for Speed
- **Caching:** Structure requests to leverage prompt, reasoning, and tool call result caching.
- **Priority Processing:** For latency-sensitive paths, use `service_tier="priority"` (note the premium pricing).
    ```python
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[...],
        service_tier="priority"
    )
    ```

## 6. Managing Excessive Tool Calls

The model may call more tools than necessary, increasing cost and time.

**Solution:** Clarify routing and set explicit policies.

### Step 1: Make "Answer from Context" the Default
Emphasize in your prompt that tools are a last resort.

### Step 2: Implement a Tool Use Policy
Provide crisp rules for when and how many tools to use.
```text
<tool_use_policy>
Select one tool or none; prefer answering from context when possible.
Cap tool calls at 2 per user request unless new information makes more strictly necessary.
</tool_use_policy>
```

### Step 3: Monitor Tool Call Metrics
Watch for spikes in `tool_calls_per_turn` or duplicate calls to the same tool, as these indicate fuzzy routing logic.

## 7. Fixing Malformed Tool Calls

In rare cases, a model may output garbled, repetitive strings when calling a tool. This is typically caused by contradictory instructions in the prompt.

**Solution:** Use GPT-5's meta-prompting ability to debug the prompt itself.
```text
Please analyze why the <tool_name> tool call is malformed.
1. Review the provided sample issue to understand the failure mode.
2. Examine the <System Prompt> and <Tool Config> carefully. Identify any ambiguities, inconsistencies, or phrasing that could mislead GPT-5 into generating an incorrect tool call.
3. For each potential cause, explain clearly how it could result in the observed failure.
4. Provide actionable recommendations to improve the <System Prompt> or <Tool Config> so GPT-5 produces valid tool calls consistently.
```

## 8. General Troubleshooting with Meta-Prompting

You can ask GPT-5 to critique and improve its own instructions. This is effective for generating solutions like those above.

Use a prompt similar to the following after a suboptimal turn:
```text
That was a high quality response, thanks! It seemed like it took you a while to finish responding though. Is there a way to clarify your instructions so you can get to a response as good as this faster next time? It's extremely important to be efficient when providing these responses or users won't get the most out of them in time. Let's see if we can improve!
1) think through the response you gave above
2) read through your instructions starting from "<insert the first line of the system prompt here>" and look for anything that might have made you take longer to formulate a high quality response than you needed
3) write out targeted (but generalized) additions/changes/deletions to your instructions to make a request like this one faster next time with the same level of quality
```

**Best Practice:** Run this meta-prompting process a few times and look for common suggestions across runs. Simplify those suggestions into a general improvement. Always validate changes with an evaluation to ensure they improve performance for your specific use case.