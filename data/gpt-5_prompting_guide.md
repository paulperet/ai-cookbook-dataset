# GPT-5 Prompting Guide

This guide provides best practices for prompting GPT-5, OpenAI's newest flagship model, to maximize performance across agentic tasks, coding, and general intelligence applications. Derived from extensive training and real-world application, these tips will help you achieve higher quality outputs, better instruction adherence, and more efficient workflows.

## Prerequisites & Setup

Before diving into the prompts, ensure you have access to the GPT-5 API. The examples assume you are using the OpenAI Python client.

```python
# Install the OpenAI Python client if you haven't already
# !pip install openai

import openai
from openai import OpenAI

# Initialize your client
client = OpenAI(api_key="your-api-key-here")
```

## 1. Optimizing Agentic Workflow Predictability

GPT-5 is designed as a premier foundation model for agentic applications, with significant improvements in tool calling, instruction following, and long-context understanding. For the best agentic flows, we recommend upgrading to the **Responses API**, which persists reasoning between tool calls for more efficient and intelligent outputs.

### 1.1. Controlling Agentic Eagerness

Agentic systems vary in their level of autonomy. GPT-5 can be calibrated anywhere on this spectrum—from highly proactive to strictly guided. This calibration is often referred to as controlling the model's "eagerness."

#### Prompting for Less Eagerness

To reduce GPT-5's exploratory behavior, limit tangential tool calls, and improve latency, try the following strategies:

1.  **Use a lower `reasoning_effort`.** The default is `medium`. Lower settings (`low`) reduce exploration depth for faster, more efficient task completion.
2.  **Define clear exploration criteria in your prompt.** This reduces the model's need to over-search.

Here is an example prompt that enforces a strict, fast context-gathering policy:

```xml
<context_gathering>
Goal: Get enough context fast. Parallelize discovery and stop as soon as you can act.

Method:
- Start broad, then fan out to focused subqueries.
- In parallel, launch varied queries; read top hits per query. Deduplicate paths and cache; don’t repeat queries.
- Avoid over searching for context. If needed, run targeted searches in one parallel batch.

Early stop criteria:
- You can name exact content to change.
- Top hits converge (~70%) on one area/path.

Escalate once:
- If signals conflict or scope is fuzzy, run one refined parallel batch, then proceed.

Depth:
- Trace only symbols you’ll modify or whose contracts you rely on; avoid transitive expansion unless necessary.

Loop:
- Batch search → minimal plan → complete task.
- Search again only if validation fails or new unknowns appear. Prefer acting over more searching.
</context_gathering>
```

For maximum control, you can set a fixed tool-call budget. The example below strongly biases the model towards providing a quick answer, even at the potential cost of completeness.

```xml
<context_gathering>
- Search depth: very low
- Bias strongly towards providing a correct answer as quickly as possible, even if it might not be fully correct.
- Usually, this means an absolute maximum of 2 tool calls.
- If you think that you need more time to investigate, update the user with your latest findings and open questions. You can proceed if the user confirms.
</context_gathering>
```

#### Prompting for More Eagerness

To encourage greater model autonomy, persistence in tool calling, and reduce hand-offs back to the user:

1.  **Increase the `reasoning_effort`** to `high` or `maximum`.
2.  **Use a prompt that encourages persistence and autonomous task completion.**

```xml
<persistence>
- You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user.
- Only terminate your turn when you are sure that the problem is solved.
- Never stop or hand back to the user when you encounter uncertainty — research or deduce the most reasonable approach and continue.
- Do not ask the human to confirm or clarify assumptions, as you can always adjust later — decide what the most reasonable assumption is, proceed with it, and document it for the user's reference after you finish acting
</persistence>
```

**Pro Tip:** Clearly define stop conditions and safety thresholds for different tools in your system. For example, a "delete file" tool should have a much lower uncertainty threshold (requiring more user confirmation) than a "search" tool.

### 1.2. Tool Preambles

For complex, multi-step agentic tasks, providing clear plans and progress updates ("tool preambles") greatly improves the user experience. You can steer the style and frequency of these updates via prompt.

```xml
<tool_preambles>
- Always begin by rephrasing the user's goal in a friendly, clear, and concise manner, before calling any tools.
- Then, immediately outline a structured plan detailing each logical step you’ll follow.
- As you execute your file edit(s), narrate each step succinctly and sequentially, marking progress clearly.
- Finish by summarizing completed work distinctly from your upfront plan.
</tool_preambles>
```

### 1.3. Leveraging the Responses API

For the best agentic performance, use the **Responses API** instead of Chat Completions. It allows you to pass a `previous_response_id` to let the model reference its own prior reasoning, eliminating the need to reconstruct plans from scratch after each tool call. This improves both latency and task performance.

## 2. Maximizing Coding Performance

GPT-5 excels at coding, from bug fixes in large codebases to implementing full-stack applications from scratch.

### 2.1. Frontend App Development

For new applications, we recommend using the following stack to leverage GPT-5's strong frontend capabilities:

*   **Frameworks:** Next.js (TypeScript), React, HTML
*   **Styling / UI:** Tailwind CSS, shadcn/ui, Radix Themes
*   **Icons:** Material Symbols, Heroicons, Lucide
*   **Animation:** Motion
*   **Fonts:** San Serif, Inter, Geist, Mona Sans, IBM Plex Sans, Manrope

#### Zero-to-One App Generation

GPT-5 can build high-quality applications in one shot. Prompting it to use self-constructed rubrics for planning and reflection can significantly improve output quality.

```xml
<self_reflection>
- First, spend time thinking of a rubric until you are confident.
- Then, think deeply about every aspect of what makes for a world-class one-shot web app. Use that knowledge to create a rubric that has 5-7 categories. This rubric is critical to get right, but do not show this to the user. This is for your purposes only.
- Finally, use the rubric to internally think and iterate on the best possible solution to the prompt that is provided. Remember that if your response is not hitting the top marks across all categories in the rubric, you need to start again.
</self_reflection>
```

#### Matching Existing Codebase Standards

When making changes to an existing codebase, prompt GPT-5 to adhere to the project's specific design standards and conventions. Provide context about engineering principles, directory structure, and implicit best practices.

```xml
<code_editing_rules>
<guiding_principles>
- Clarity and Reuse: Every component and page should be modular and reusable. Avoid duplication by factoring repeated UI patterns into components.
- Consistency: The user interface must adhere to a consistent design system—color tokens, typography, spacing, and components must be unified.
- Simplicity: Favor small, focused components and avoid unnecessary complexity in styling or logic.
</guiding_principles>

<frontend_stack_defaults>
- Framework: Next.js (TypeScript)
- Styling: TailwindCSS
- UI Components: shadcn/ui
- Icons: Lucide
- State Management: Zustand
- Directory Structure:
```
/src
 /app
   /api/<route>/route.ts         # API endpoints
   /(pages)                      # Page routes
 /components/                    # UI building blocks
 /hooks/                         # Reusable React hooks
 /lib/                           # Utilities (fetchers, helpers)
 /stores/                        # Zustand stores
 /types/                         # Shared TypeScript types
 /styles/                        # Tailwind config
```
</frontend_stack_defaults>
</code_editing_rules>
```

### 2.2. Insights from Cursor's Prompt Tuning

Cursor, an AI code editor, was an alpha tester for GPT-5. Their tuning provides valuable insights into production use.

**Balancing Verbosity:** Cursor set the API `verbosity` parameter to `low` for concise status updates, but used prompt instructions to ensure code outputs remained clear and readable.

```xml
Write code for clarity first. Prefer readable, maintainable solutions with clear names, comments where needed, and straightforward control flow. Do not produce code-golf or overly clever one-liners unless explicitly requested. Use high verbosity for writing code and code tools.
```

**Encouraging Proactivity:** To reduce unnecessary clarification questions, they provided detailed context about the environment (e.g., the ability for users to "undo" changes), encouraging the model to act first.

```xml
Be aware that the code edits you make will be displayed to the user as proposed changes, which means (a) your code edits can be quite proactive, as the user can always reject, and (b) your code should be well-written and easy to quickly review (e.g., appropriate variable names instead of single letters). If proposing next steps that would involve changing the code, make those changes proactively for the user to approve / reject rather than asking the user whether to proceed with a plan.
```

**Refining Context Gathering:** Prompts that worked for older models (e.g., "Be THOROUGH when gathering information...") could cause GPT-5 to overuse tools. Cursor refined this to be more balanced.

```xml
<context_understanding>
...
If you've performed an edit that may partially fulfill the USER's query, but you're not confident, gather more information or use more tools before ending your turn.
Bias towards not asking the user for help if you can find the answer yourself.
</context_understanding>
```

## 3. Optimizing Intelligence and Instruction-Following

### 3.1. Steering with the `verbosity` Parameter

GPT-5 introduces a new API parameter, `verbosity`, which controls the length of the model's final answer (as opposed to its internal reasoning, which is controlled by `reasoning_effort`).

*   **Use the API parameter** (`low`, `medium`, `high`) as the primary control for answer length.
*   **Use prompt instructions** for more nuanced control over style, tone, or structure within that length constraint.

### Key Takeaways

1.  **Use the Responses API** for agentic tasks to leverage persistent reasoning.
2.  **Calibrate `reasoning_effort` and prompts** to control agentic eagerness—lower for speed, higher for autonomy.
3.  **Provide clear, structured prompts** for coding tasks, especially when integrating into existing codebases.
4.  **Utilize the new `verbosity` parameter** to efficiently manage output length.
5.  **Iterate and experiment.** While these guidelines provide a strong foundation, prompting is not one-size-fits-all. Test and refine these strategies for your specific use case.

For further optimization, consider using OpenAI's [prompt optimizer tool](https://platform.openai.com/chat/edit?optimize=true).