# GPT-OSS-Safeguard User Guide: A Technical Tutorial

## Introduction & Overview

This guide explains how to effectively use **gpt-oss-safeguard**, an open-weight reasoning model specifically trained for safety classification tasks. You'll learn to write policy prompts that maximize its reasoning power, choose optimal policy lengths, and integrate its outputs into production Trust & Safety systems.

### What is gpt-oss-safeguard?

gpt-oss-safeguard is a fine-tuned version of [gpt-oss](https://openai.com/index/introducing-gpt-oss/) designed for safety classification. Unlike prebaked safety models with fixed taxonomies, it follows **your own written policies**, enabling "bring-your-own-policy" Trust & Safety AI. Its reasoning capabilities allow it to handle nuanced content, explain borderline decisions, and adapt to contextual factors.

### Prerequisites & Setup

Before you begin, ensure you have a method to run the model. gpt-oss-safeguard can be deployed using several popular inference solutions. Choose one based on your infrastructure:

#### Option 1: HuggingFace Transformers (Local/Server)

The Transformers library provides a flexible way to load and run models. First, install the required package:

```bash
pip install transformers
```

You can then interact with the model via the CLI or HTTP API.

**Using the transformers chat CLI:**
```bash
transformers chat localhost:8000 --model-name-or-path openai/gpt-oss-safeguard-20b
```

**Sending an HTTP request with cURL:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-safeguard-20b",
    "stream": true,
    "messages": [
      { "role": "system", "content": "<your policy>" },
      { "role": "user", "content": "<user content to verify>" }
    ]
  }'
```

#### Option 2: Ollama (Local)

Ollama supports direct model downloads. Run one of the following commands to start the model:

**For the 20B parameter model:**
```bash
ollama run gpt-oss-safeguard:20b
```

**For the 120B parameter model:**
```bash
ollama run gpt-oss-safeguard:120b
```

Ollama provides OpenAI-compatible and native APIs for integration.

#### Option 3: LM Studio (Local)

Use LM Studio's CLI to download and run the model:

**For the 20B model:**
```bash
lms get openai/gpt-oss-safeguard-20b
```

**For the 120B model:**
```bash
lms get openai/gpt-oss-safeguard-120b
```

#### Option 4: vLLM (Server - Recommended for GPUs)

vLLM is recommended for dedicated GPU servers. Install it using `uv`:

```bash
uv pip install vllm==0.10.2 --torch-backend=auto
```

Start the server with:
```bash
vllm serve openai/gpt-oss-safeguard-120b
```

## Core Concepts: The Harmony Response Format

gpt-oss-safeguard uses the **Harmony prompt format** to structure its output, which is critical for auditability in Trust & Safety workflows. Harmony separates the response into two channels:

1.  **Reasoning Channel:** The model's internal analysis, considering policy, edge cases, and logic.
2.  **Output Channel:** The final, formatted classification decision you specified.

You control the depth of reasoning via the `reasoning_effort` parameter in your system message (`low`, `medium`, or `high`). `medium` is the default.

When using vLLM or other inference servers with chat message inputs, format your requests as follows:
*   **System Message:** Contains your policy prompt. Optionally include `Reasoning: high` to control effort.
*   **User Message:** Contains the content you want to classify.

## Tutorial: Writing and Using Effective Policy Prompts

A policy prompt defines the operational boundaries for the model. Think of it as combining a policy document with training examples for a human reviewer.

### Step 1: Structure Your Policy Prompt

Effective prompts should have four distinct sections for clarity. Use this consistent pattern:

```markdown
# Policy Name

## INSTRUCTIONS
Describe what oss-safeguard should do and how to respond. Be explicit about the output format.

## DEFINITIONS
Clarify key terms and context to avoid ambiguity.

## VIOLATES (1)
Describe the specific behaviors or content that should be flagged as a violation.

## SAFE (0)
Describe content that should **not** be flagged, helping to define boundaries.

## EXAMPLES
Provide 4–6 short, concrete examples near the decision boundary, labeled with `0` or `1`.

Content: [INPUT]
Answer (0 or 1):
```

### Step 2: Design Reliable Output Instructions

The output instructions tell the model *how* to communicate its decision. Be explicit and literal. These instructions must be reinforced in both the **INSTRUCTIONS** section and just before the **EXAMPLES**.

#### Option A: Binary Output (Simple Yes/No)
Use this for speed when detailed reasoning is not required.

```markdown
Return exactly one character: 0 or 1.
Do not include any explanation or punctuation.

0 = Content does NOT violate this policy.
1 = Content violates this policy.
```

#### Option B: Policy-Referencing Output
This format encourages the model to identify which specific policy rule applies.

```markdown
If the content violates this policy, return:
{"violation": 1, "policy_category": "<category_or_rule_id>"}

If the content does NOT violate this policy, return:
{"violation": 0, "policy_category": null}
```

#### Option C: Output with Rationale (Recommended)
This leverages the model's core strength by requiring it to justify its decision with policy citations.

```json
{
  "violation": 1,
  "policy_category": "H2.f",
  "rule_ids": ["H2.d", "H2.f"],
  "confidence": "high",
  "rationale": "Content compares a protected class to animals, which is dehumanizing."
}
```

### Step 3: Optimize Policy Length

Policy length acts as a "context budget" that influences how deeply the model can reason.

1.  **Tokenize Your Prompt:** Use a tool like the [OpenAI Tokenizer](https://platform.openai.com/tokenizer) to check length.
2.  **Find the Sweet Spot:** Early testing suggests the optimal range is **400-600 tokens**. The model can handle up to ~10,000 tokens, but performance may degrade.
3.  **Leave Room for Output:** Do not cap the maximum output tokens too low. The model needs space to reason. If you need to limit processing time, adjust the `reasoning_effort` to `low` instead of restricting token count.
4.  **Pre-compress Long Policies:** If you have a long policy with multiple categories, try compressing each individual policy section to 300–600 tokens, including definitions and examples.

### Step 4: Integrate into Your Trust & Safety Stack

gpt-oss-safeguard is designed to fit into existing infrastructure. Consider these best practices for integration:

1.  **Pre-filter Content:** Since gpt-oss-safeguard can be more computationally intensive than traditional classifiers, use small, high-recall classifiers first to determine if content is relevant to your priority risks before sending it for deep analysis.
2.  **Balance Cost and Nuance:** Traditional classifiers offer lower latency and cost. Use them for clear-cut cases. Deploy gpt-oss-safeguard for nuanced, borderline, or high-stakes content where its reasoning and policy alignment are most valuable.
3.  **Enable Auditing:** Use the **rationale output format** (Option C above) to generate an audit trail. This is crucial for understanding enforcement decisions and improving policies over time.

## Summary and Next Steps

You are now equipped to:
1.  **Run** gpt-oss-safeguard using your preferred inference solution (Transformers, Ollama, LM Studio, or vLLM).
2.  **Structure** effective policy prompts with clear Instructions, Definitions, Criteria, and Examples.
3.  **Design** output instructions that range from simple binary decisions to detailed, auditable rationales.
4.  **Optimize** policy length for performance and integrate the model efficiently into a larger Trust & Safety system.

Experiment with different policy structures and output formats to find what works best for your specific use case. The model's flexibility allows you to update policies instantly without retraining, making it a powerful tool for adaptive and transparent content safety.