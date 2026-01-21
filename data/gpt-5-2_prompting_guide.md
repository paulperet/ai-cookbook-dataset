# GPT-5.2 Prompting Guide

## Introduction

GPT-5.2 is OpenAI's newest flagship model for enterprise and agentic workloads. It is engineered to deliver higher accuracy, stronger instruction following, and more disciplined execution across complex workflows. Building on GPT-5.1, it offers improved token efficiency, cleaner formatting, and significant gains in structured reasoning, tool grounding, and multimodal understanding.

This guide provides practical prompting patterns and migration practices to maximize GPT-5.2's performance in production systems. Small adjustments to prompt structure, verbosity constraints, and reasoning settings can lead to substantial improvements in correctness, latency, and developer trust.

## Key Behavioral Differences

Compared to previous models like GPT-5 and GPT-5.1, GPT-5.2 exhibits several key characteristics:

*   **More Deliberate Scaffolding:** It builds clearer plans and intermediate structures by default.
*   **Generally Lower Verbosity:** Outputs are more concise and task-focused, though this remains prompt-sensitive.
*   **Stronger Instruction Adherence:** It demonstrates less drift from user intent and improved formatting.
*   **Tool Efficiency Trade-offs:** It may take more tool actions in interactive flows, which can be optimized via prompting.
*   **Conservative Grounding Bias:** It tends to favor correctness and explicit reasoning, with improved ambiguity handling.

## Prompting Patterns for GPT-5.2

### 1. Controlling Verbosity and Output Shape

Provide clear, concrete length constraints to steer the model's output, especially in enterprise and coding contexts.

**Example Verbosity Specification:**
```xml
<output_verbosity_spec>
- Default: 3–6 sentences or ≤5 bullets for typical answers.
- For simple “yes/no + short explanation” questions: ≤2 sentences.
- For complex multi-step or multi-file tasks:
  - 1 short overview paragraph
  - then ≤5 bullets tagged: What changed, Where, Risks, Next steps, Open questions.
- Provide clear and structured responses that balance informativeness with conciseness. Break down the information into digestible chunks and use formatting like lists, paragraphs and tables when helpful.
- Avoid long narrative paragraphs; prefer compact bullets and short sections.
- Do not rephrase the user’s request unless it changes semantics.
</output_verbosity_spec>
```

### 2. Preventing Scope Drift

GPT-5.2 is strong at structured code but may add unnecessary UX features or styling. Explicit constraints help keep it focused.

**Example Scope Constraint:**
```xml
<design_and_scope_constraints>
- Explore any existing design systems and understand it deeply.
- Implement EXACTLY and ONLY what the user requests.
- No extra features, no added components, no UX embellishments.
- Style aligned to the design system at hand.
- Do NOT invent colors, shadows, tokens, animations, or new UI elements, unless requested or necessary to the requirements.
- If any instruction is ambiguous, choose the simplest valid interpretation.
</design_and_scope_constraints>
```

### 3. Handling Long Context and Recall

For tasks involving long documents, force the model to summarize and re-ground itself to improve recall.

**Example Long-Context Handling:**
```xml
<long_context_handling>
- For inputs longer than ~10k tokens (multi-chapter docs, long threads, multiple PDFs):
  - First, produce a short internal outline of the key sections relevant to the user’s request.
  - Re-state the user’s constraints explicitly (e.g., jurisdiction, date range, product, team) before answering.
  - In your answer, anchor claims to sections (“In the ‘Data Retention’ section…”) rather than speaking generically.
- If the answer depends on fine details (dates, thresholds, clauses), quote or paraphrase them.
</long_context_handling>
```

### 4. Mitigating Ambiguity and Hallucination Risk

Configure prompts to handle ambiguous queries and reduce overconfident hallucinations.

**Example Uncertainty Handling:**
```xml
<uncertainty_and_ambiguity>
- If the question is ambiguous or underspecified, explicitly call this out and:
  - Ask up to 1–3 precise clarifying questions, OR
  - Present 2–3 plausible interpretations with clearly labeled assumptions.
- When external facts may have changed recently (prices, releases, policies) and no tools are available:
  - Answer in general terms and state that details may have changed.
- Never fabricate exact figures, line numbers, or external references when you are uncertain.
- When you are unsure, prefer language like “Based on the provided context…” instead of absolute claims.
</uncertainty_and_ambiguity>
```

**Optional Self-Check for High-Risk Contexts:**
```xml
<high_risk_self_check>
Before finalizing an answer in legal, financial, compliance, or safety-sensitive contexts:
- Briefly re-scan your own answer for:
  - Unstated assumptions,
  - Specific numbers or claims not grounded in context,
  - Overly strong language (“always,” “guaranteed,” etc.).
- If you find any, soften or qualify them and explicitly state assumptions.
</high_risk_self_check>
```

## Compaction for Extended Context

For long-running, tool-heavy workflows that exceed the standard context window, GPT-5.2 with Reasoning supports response compaction via the `/responses/compact` endpoint. This feature performs a loss-aware compression of prior conversation state, allowing the model to continue reasoning across extended workflows.

**When to Use Compaction:**
*   Multi-step agent flows with many tool calls.
*   Long conversations where earlier turns must be retained.
*   Iterative reasoning beyond the maximum context window.

**Best Practices:**
*   Monitor context usage and plan ahead.
*   Compact after major milestones (e.g., tool-heavy phases), not every turn.
*   Keep prompts functionally identical when resuming to avoid behavior drift.
*   Treat compacted items as opaque; don't parse or depend on their internal structure.

**Example: Compacting a Response**
```python
from openai import OpenAI
import json

client = OpenAI()

# 1. Create an initial response
response = client.responses.create(
   model="gpt-5.2",
   input=[
       {
           "role": "user",
           "content": "write a very long poem about a dog.",
       },
   ]
)

output_json = [msg.model_dump() for msg in response.output]

# 2. Compact the conversation to reduce token footprint
compacted_response = client.responses.compact(
   model="gpt-5.2",
   input=[
       {
           "role": "user",
           "content": "write a very long poem about a dog.",
       },
       output_json[0]
   ]
)

print(json.dumps(compacted_response.model_dump(), indent=2))
```

## Agentic Steerability & User Updates

GPT-5.2 excels at agentic scaffolding and multi-step execution. To optimize updates during long-running tasks, clamp verbosity and enforce scope discipline.

**Example User Update Specification:**
```xml
<user_updates_spec>
- Send brief updates (1–2 sentences) only when:
  - You start a new major phase of work, or
  - You discover something that changes the plan.
- Avoid narrating routine tool calls (“reading file…”, “running tests…”).
- Each update must include at least one concrete outcome (“Found X”, “Confirmed Y”, “Updated Z”).
- Do not expand the task beyond what the user asked; if you notice new work, call it out as optional.
</user_updates_spec>
```

## Tool-Calling and Parallelism

GPT-5.2 shows improved tool reliability. Use clear, concise tool descriptions and encourage parallelism for independent operations.

**Example Tool Usage Rules:**
```xml
<tool_usage_rules>
- Prefer tools over internal knowledge whenever:
  - You need fresh or user-specific data (tickets, orders, configs, logs).
  - You reference specific IDs, URLs, or document titles.
- Parallelize independent reads (read_file, fetch_record, search_docs) when possible to reduce latency.
- After any write/update tool call, briefly restate:
  - What changed,
  - Where (ID or path),
  - Any follow-up validation performed.
</tool_usage_rules>
```

## Structured Extraction from Documents

GPT-5.2 shows strong improvements in extracting structured data from PDFs, tables, and Office documents. Always provide a clear output schema.

**Example Extraction Specification:**
```xml
<extraction_spec>
You will extract structured data from tables/PDFs/emails into JSON.

- Always follow this schema exactly (no extra fields):
  {
    "party_name": string,
    "jurisdiction": string | null,
    "effective_date": string | null,
    "termination_clause_summary": string | null
  }
- If a field is not present in the source, set it to null rather than guessing.
- Before returning, quickly re-scan the source for any missed fields and correct omissions.
</extraction_spec>
```

For multi-document extraction, add guidance to serialize results per document and include a stable identifier (e.g., filename, page range).

## Prompt Migration Guide to GPT-5.2

Follow this step-by-step process to migrate your prompts and model configurations to GPT-5.2 while maintaining stable behavior and predictable cost/latency.

### Step 1: Understand the `reasoning_effort` Parameter
GPT-5-class models support a `reasoning_effort` parameter (`none`|`minimal`|`low`|`medium`|`high`|`xhigh`) that trades off speed/cost for deeper reasoning. Use the following mapping as a starting point:

| Current Model | Target Model | Target `reasoning_effort` | Notes |
| :--- | :--- | :--- | :--- |
| GPT-4o | GPT-5.2 | `none` | Treat as "fast/low-deliberation" by default. Increase only if evals regress. |
| GPT-4.1 | GPT-5.2 | `none` | Same mapping as GPT-4o to preserve snappy behavior. |
| GPT-5 | GPT-5.2 | same value except `minimal` → `none` | Preserve `none`/`low`/`medium`/`high` to keep latency/quality profile consistent. |
| GPT-5.1 | GPT-5.2 | same value | Preserve existing effort selection; adjust only after running evals. |

**Note:** The default `reasoning_effort` for GPT-5 is `medium`, and for GPT-5.1 and GPT-5.2 it is `none`.

### Step 2: Execute the Migration Process
1.  **Switch Models, Keep Prompts Identical:** Change only the model name in your API calls. This isolates the effect of the model change.
2.  **Pin `reasoning_effort`:** Explicitly set the `reasoning_effort` for GPT-5.2 based on the mapping table above to match your previous model's latency and depth profile.
3.  **Run Evaluations for a Baseline:** Execute your evaluation suite. If results are good (often they improve at `medium`/`high`), you are ready to proceed.
4.  **If Regressions Occur, Tune the Prompt:** Use the [Prompt Optimizer](https://platform.openai.com/chat/edit?optimize=true) and apply targeted constraints (verbosity, scope, schema) to restore parity or improve performance.
5.  **Re-run Evals After Each Change:** Iterate by either adjusting the `reasoning_effort` one notch or making incremental prompt tweaks, then re-measure with your evals.

## Web Search and Research Best Practices

GPT-5.2 is highly capable at synthesizing information from multiple sources. Provide clear instructions to steer its research behavior.

**Example Web Search Rules:**
```xml
<web_search_rules>
- Act as an expert research assistant; default to comprehensive, well-structured answers.
- Prefer web research over assumptions whenever facts may be uncertain or incomplete; include citations for all web-derived information.
- Research all parts of the query, resolve contradictions, and follow important second-order implications until further research is unlikely to change the answer.
- Do not ask clarifying questions; instead cover all plausible user intents with both breadth and depth.
- Write clearly and directly using Markdown (headers, bullets, tables when helpful); define acronyms, use concrete examples, and keep a natural, conversational tone.
</web_search_rules>
```

## Conclusion

GPT-5.2 represents a significant advancement for building production-grade AI agents that prioritize accuracy, reliability, and disciplined execution. Its improved instruction following, cleaner output, and consistent behavior make it well-suited for complex, tool-heavy workflows.

Most existing prompts will migrate cleanly. The key to a successful transition is to:
1.  Change one variable at a time (start with the model).
2.  Explicitly set the `reasoning_effort` parameter to match your previous performance profile.
3.  Rely on evaluations to validate behavior before making any prompt changes.
4.  Apply the prompting patterns outlined in this guide—particularly around verbosity, scope, and structure—to further optimize performance.

With careful prompting and measured iteration, GPT-5.2 can deliver higher quality outcomes while maintaining predictable cost and latency.