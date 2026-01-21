# A Developer's Guide to Prompting Large Language Models

## Introduction

Large Language Models (LLMs) are powerful functions that map text to text. Given an input string, an LLM predicts the most likely subsequent text. Through training on vast datasets, these models learn complex patterns including grammar, reasoning, coding, and conversational skills.

This guide walks you through the core techniques for controlling LLM behavior through effective prompting, with practical examples you can apply immediately.

## Prerequisites

This guide focuses on prompt design concepts. To run the examples, you'll need access to an LLM API (like OpenAI's) or a local model. The principles apply universally across most modern LLMs.

## Core Prompting Techniques

LLMs respond differently based on how you structure your input. Here are the five primary approaches, each with distinct use cases.

### 1. Instruction Prompting

**When to use:** Direct tasks where you want explicit, concise output.

Write clear instructions telling the model exactly what you want. Place instructions at the beginning, end, or both, and be specific about your requirements.

```text
Extract the name of the author from the quotation below.

“Some humans theorize that intelligent species go extinct before they can expand into outer space. If they're correct, then the hush of the night sky is the silence of the graveyard.”
― Ted Chiang, Exhalation
```

**Expected Output:**
```
Ted Chiang
```

**Key Insight:** Instructions work best when detailed. Don't hesitate to write a paragraph specifying format, tone, and constraints, but remain mindful of token limits.

### 2. Completion Prompting

**When to use:** When you want the model to naturally continue a pattern or thought.

Start writing the pattern you want completed. This method often requires careful crafting and may need stop sequences to prevent over-generation.

```text
“Some humans theorize that intelligent species go extinct before they can expand into outer space. If they're correct, then the hush of the night sky is the silence of the graveyard.”
― Ted Chiang, Exhalation

The author of this quote is
```

**Expected Output:**
```
 Ted Chiang
```

**Note:** The model continues from where you left off, so ensure your prompt leads naturally to the desired completion.

### 3. Scenario Prompting

**When to use:** Complex tasks, role-playing, or when you need creative, contextual responses.

Set up a scenario, role, or hypothetical situation for the model to inhabit, then present your query.

```text
Your role is to extract the name of the author from any given text

“Some humans theorize that intelligent species go extinct before they can expand into outer space. If they're correct, then the hush of the night sky is the silence of the graveyard.”
― Ted Chiang, Exhalation
```

**Expected Output:**
```
 Ted Chiang
```

**Advantage:** This approach helps the model adopt a specific perspective, which is useful for specialized tasks like acting as a customer service agent or technical expert.

### 4. Few-Shot Learning (Demonstration Prompting)

**When to use:** When examples communicate the task better than instructions.

Show the model what you want by providing a few examples of input-output pairs directly in the prompt.

```text
Quote:
“When the reasoning mind is forced to confront the impossible again and again, it has no choice but to adapt.”
― N.K. Jemisin, The Fifth Season
Author: N.K. Jemisin

Quote:
“Some humans theorize that intelligent species go extinct before they can expand into outer space. If they're correct, then the hush of the night sky is the silence of the graveyard.”
― Ted Chiang, Exhalation
Author:
```

**Expected Output:**
```
 Ted Chiang
```

**Best Practice:** Use 2-5 clear, diverse examples. Ensure your examples follow a consistent format that the model can easily recognize and replicate.

### 5. Fine-Tuned Model Prompting

**When to use:** When you have hundreds or thousands of training examples for a specific task.

After fine-tuning a model on your dataset, you can use simpler prompts. Include separator sequences to clearly indicate where the prompt ends.

```text
“Some humans theorize that intelligent species go extinct before they can expand into outer space. If they're correct, then the hush of the night sky is the silence of the graveyard.”
― Ted Chiang, Exhalation

###


```

**Expected Output:**
```
 Ted Chiang
```

**Important:** The separator (`###` in this example) tells the model to stop processing the input and start generating the output. Choose a separator that doesn't appear in your typical content.

## Applying LLMs to Code Tasks

Modern LLMs like GPT-4 excel at code generation, explanation, and completion. They power tools like GitHub Copilot, Replit, and Cursor. While these models are advanced, clear prompting remains essential for optimal results.

### Prompt Engineering Tips for Better Outputs

Regardless of your chosen prompting technique, these strategies will improve your results:

1. **Be Specific About Format**
   Instead of "List the items," try "Return a comma-separated list of items."
   
2. **Provide Context**
   Give background information that helps the model understand the bigger picture of your request.

3. **Ask for Expert-Level Output**
   Phrases like "Explain as if to an expert" or "Provide a detailed, step-by-step solution" often yield higher quality responses.

4. **Request Chain-of-Thought Reasoning**
   Adding "Let's think step by step" before complex problems can significantly improve accuracy by encouraging the model to articulate its reasoning process.

5. **Iterate and Experiment**
   Prompt engineering is iterative. Try slight variations, test different phrasings, and learn what works best for your specific use case.

## Next Steps

Now that you understand the core prompting techniques:

1. **Experiment** with each approach using your preferred LLM
2. **Combine techniques** (e.g., scenario + few-shot learning)
3. **Explore advanced patterns** like chain-of-thought and self-consistency
4. **Consider fine-tuning** if you have a large, consistent dataset for a specific task

Remember: The prompt is your primary interface with the model. Invest time in crafting clear, specific prompts, and you'll be rewarded with higher quality, more reliable outputs.