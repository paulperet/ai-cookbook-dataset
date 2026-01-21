# GPT-5.1 Prompting Guide: Optimizing for Agentic and Coding Tasks

## Introduction

GPT-5.1 is OpenAI's newest flagship model, designed to balance intelligence and speed for a wide range of agentic and coding applications. A key feature is the introduction of a `none` reasoning mode, which enables low-latency interactions similar to previous non-reasoning models like GPT-4.1. This guide compiles proven prompt patterns to maximize performance, reliability, and user experience in production deployments.

## Prerequisites & Migration

Before diving into advanced techniques, ensure you're set up correctly.

### Setup

If you're using the OpenAI API, your call might look like this:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-5.1",
    messages=[{"role": "user", "content": "Your prompt here."}],
    reasoning_effort="none",  # Use "none" for low-latency tasks
    # ... other parameters
)
```

### Key Migration Guidance

If you are migrating from GPT-4.1 or GPT-5, follow these principles:

1.  **Emphasize Persistence:** GPT-5.1 is calibrated for efficiency but can sometimes be overly concise. Prompt it to be thorough and complete in its answers.
2.  **Control Output Formatting:** Be explicit about desired verbosity and detail in your instructions.
3.  **Update Coding Agents:** If building a coding agent, migrate from `apply_patch` to the newer, named tool implementation.
4.  **Leverage Instruction Following:** GPT-5.1 excels at following clear, direct instructions. Resolve conflicting guidance in your prompts.

For coding-specific tasks, also refer to the [GPT-5.2 Codex prompting guide](https://cookbook.openai.com/examples/gpt-5/codex_prompting_guide).

---

## Part 1: Shaping Your Agent's Personality and Communication

GPT-5.1 offers fine-grained control over your agent's tone, style, and communication frequency.

### Step 1: Define a Clear Persona

A well-defined persona is crucial for customer-facing agents. The prompt should establish the agent's communication philosophy. Below is an example for a customer support agent that values efficiency and adaptive politeness.

```xml
<final_answer_formatting>
You value clarity, momentum, and respect measured by usefulness rather than pleasantries. Your default instinct is to keep conversations crisp and purpose-driven, trimming anything that doesn't move the work forward.

- **Adaptive Politeness:**
  - When a user is warm or considerate, offer a single, succinct acknowledgment (e.g., 'Got it', 'I understand') before pivoting back to action.
  - In high-stakes situations (urgent issues, deadlines), drop acknowledgments entirely and move straight to solving the problem.

- **Core Inclination:**
  - Speak with grounded directness. Respect the user's time by being efficient.
  - Politeness is demonstrated through precision and responsiveness, not verbal fluff.

- **Conversational Rhythm:**
  - Match the user's energy and tempo. Listen closely and respond accordingly.
  - Never repeat acknowledgments. Once understanding is signaled, fully pivot to the task.
</final_answer_formatting>
```

### Step 2: Control Output Length and Format

You can control verbosity through both the API parameter (`verbosity`) and explicit prompting. For concise outputs, use instructions like this:

```xml
<output_verbosity_spec>
- Respond in plain text styled in Markdown, using at most 2 concise sentences.
- Lead with what you did (or found) and context only if needed.
- For code, reference file paths and show code blocks only if necessary to clarify the change.
</output_verbosity_spec>
```

For coding agents, provide concrete rules on response compactness based on the change size:

```xml
<final_answer_formatting>
- **Final Answer Compactness Rules:**
  - **Tiny change (~10 lines):** 2–5 sentences. 0–1 short code snippet (≤3 lines) only if essential.
  - **Medium change (few files):** ≤6 bullets. At most 1–2 short snippets (≤8 lines each).
  - **Large change (multi-file):** Summarize per file with 1–2 bullets. Avoid inlining large code blocks.
- Never include "before/after" code pairs or full method bodies in the final message. Prefer referencing file and symbol names.
</final_answer_formatting>
```

### Step 3: Configure Informative User Updates

"User updates" (or preambles) are assistant messages that keep the user informed during a long-running task. You can control their frequency, content, and tone.

```xml
<user_updates_spec>
You'll work for stretches with tool calls — it's critical to keep the user updated as you work.

<frequency_and_length>
- Send short updates (1–2 sentences) every few tool calls when there are meaningful changes.
- Post an update at least every 6 execution steps or 8 tool calls (whichever comes first).
- If you expect a longer stretch of work, post a brief heads-down note explaining why and when you’ll report back.
</frequency_and_length>

<content>
- Before the first tool call, provide a quick plan: goal, constraints, and next steps.
- While exploring, call out meaningful discoveries and new information.
- Always state at least one concrete outcome since the prior update (e.g., “found X”, “confirmed Y”).
- If you change the plan, say so explicitly in the next update.
- End with a brief recap and any follow-up steps.
</content>
</user_updates_spec>
```

To improve perceived latency, instruct the model to send an initial commentary message immediately:

```xml
<user_update_immediacy>
Always explain what you're doing in a commentary message FIRST, BEFORE sampling an analysis thinking message. This is critical in order to communicate immediately to the user.
</user_update_immediacy>
```

---

## Part 2: Optimizing Intelligence and Tool Usage

GPT-5.1 pays close attention to instructions regarding solution completeness, tool usage, and parallelism.

### Step 4: Encourage Complete Solutions

Prevent the model from ending a task prematurely by emphasizing persistence and autonomy.

```xml
<solution_persistence>
- Treat yourself as an autonomous senior pair-programmer: once the user gives a direction, proactively gather context, plan, implement, test, and refine without waiting for additional prompts.
- Persist until the task is fully handled end-to-end within the current turn whenever feasible.
- Be extremely biased for action. If a user provides a directive that is somewhat ambiguous, assume you should go ahead and make the change.
</solution_persistence>
```

### Step 5: Structure Effective Tool Definitions and Usage

Tool definitions should clearly describe functionality. Usage rules should be specified in the prompt.

**1. Define the Tool:**
```json
{
  "name": "create_reservation",
  "description": "Create a restaurant reservation for a guest. Use when the user asks to book a table with a given name and time.",
  "parameters": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "Guest full name for the reservation."
      },
      "datetime": {
        "type": "string",
        "description": "Reservation date and time (ISO 8601 format)."
      }
    },
    "required": ["name", "datetime"]
  }
}
```

**2. Provide Usage Rules and Examples in the Prompt:**
```xml
<reservation_tool_usage_rules>
- When the user asks to book, reserve, or schedule a table, you MUST call `create_reservation`.
- Do NOT guess a reservation time or name — ask for whichever detail is missing.
- After calling the tool, confirm the reservation naturally.
</reservation_tool_usage_rules>

<reservation_tool_example>
*Example:*
User: “Book a table for Sarah tomorrow at 7pm.”
Assistant → (calls `create_reservation` with arguments) →
Tool returns: `{ "confirmation_number": "R12345" }`
Assistant: “All set — your reservation for Sarah tomorrow at 7:00pm is confirmed. Your confirmation number is R12345.”
</reservation_tool_example>
```

### Step 6: Leverage Parallel Tool Calling

GPT-5.1 executes parallel tool calls efficiently. Encourage this behavior in your system prompt, especially for batch operations like reading files.

```
Parallelize tool calls whenever possible. Batch reads (read_file) and edits (apply_patch) to speed up the process.
```

---

## Part 3: Using the "None" Reasoning Mode for Efficiency

The `reasoning_effort="none"` mode is ideal for low-latency tasks and works well with hosted tools like Web Search and File Search. Prompting strategies from non-reasoning models (like GPT-4.1) apply here.

### Step 7: Prompt for Careful Planning and Verification

Even without reasoning tokens, you can instruct the model to plan and verify its actions.

**Encourage Internal Planning:**
```
You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls, ensuring the user's query is completely resolved. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.
```

**Prompt for Output Verification:**
```
When selecting a replacement variant, verify it meets all user constraints (cheapest, brand, spec, etc.). Quote the item-id and price back for confirmation before executing.
```

**Prevent Premature Termination:**
```
Remember, you are an agent - please keep going until the user’s query is completely resolved, before ending your turn and yielding back to the user.
```

### Step 8: Implement a Planning Tool for Complex Tasks

For long-running, multi-step tasks, implement a dedicated planning tool to help the model track progress.

```xml
<plan_tool_usage>
- For medium or larger tasks (e.g., multi-file changes, adding features), you must create and maintain a lightweight plan in the TODO/plan tool before your first code/tool action.
- Create 2–5 milestone/outcome items; avoid micro-steps like "open file".
- Maintain statuses in the tool: exactly one item `in_progress` at a time; mark items `complete` when done.
</plan_tool_usage>
```

---

## Summary and Next Steps

This guide provides a foundation for prompting GPT-5.1 effectively. Key takeaways:

1.  **Be Explicit:** Clearly define persona, output format, and update frequency.
2.  **Encourage Completeness:** Prompt the model to persist until a task is fully resolved.
3.  **Structure Tool Use:** Combine clear tool definitions with concrete usage rules and examples.
4.  **Leverage "None" Mode:** For low-latency tasks, use `reasoning_effort="none"` and apply planning/verification prompts.

Prompting is iterative. Use these patterns as a starting point and adapt them to your specific tools, workflows, and user needs for optimal results.