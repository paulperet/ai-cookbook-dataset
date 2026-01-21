# Building an LLM-as-a-Judge Evaluator to Detect Hallucinations with Braintrust

When evaluating a customer service bot, simple string comparisons often fail. For example, if the correct answer is "You can return items within 30 days of purchase," but the bot generates "You can return items within 30 days," a heuristic like Levenshtein distance would incorrectly flag this as wrong. A more nuanced approach is to use an **LLM-as-a-judge**—leveraging a large language model to assess answer quality based on factual content, not just surface text.

This guide walks through building an LLM-as-a-judge scorer to detect hallucinations using [Braintrust](https://www.braintrust.dev/), a third-party evaluation platform compatible with OpenAI models.

## Prerequisites

Before you begin, ensure you have:
- A [Braintrust account](https://www.braintrust.dev/signup) and an API key.
- An OpenAI API key.

Set your API keys as environment variables:
```bash
export BRAINTRUST_API_KEY="your-braintrust-key"
export OPENAI_API_KEY="your-openai-key"
```

## Step 1: Install Dependencies

Install the required Python packages.

```bash
pip install autoevals duckdb braintrust openai --quiet
```

## Step 2: Initialize the OpenAI Client

We'll use the `AsyncOpenAI` client wrapped by Braintrust to log all LLM calls for evaluation.

```python
import os
import braintrust
from openai import AsyncOpenAI

# Log into Braintrust and wrap the OpenAI client
braintrust.login(api_key=os.environ["BRAINTRUST_API_KEY"])
client = braintrust.wrap_openai(AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]))
```

## Step 3: Load and Explore the Dataset

We'll use the CoQA dataset, which contains passages, questions, and answers. We load a small sample via DuckDB.

```python
import duckdb

# Connect to DuckDB and load the CoQA validation dataset
con = duckdb.connect(":memory:")
full_result = con.query("""
    SELECT * FROM 'hf://datasets/stanfordnlp/coqa/data/validation-00000-of-00001.parquet'
        LIMIT 40
""").fetchall()

# Inspect a single record
single_result = full_result[10]
print("Passage:")
print(single_result[1])
print("\nQuestion:")
print(single_result[2][0])
print("\nAnswer:")
print(single_result[3]["input_text"][0])
```

### Flatten the Data

The dataset is nested. Let's flatten it into a list of `(passage, question, answer)` tuples for easier processing.

```python
from dataclasses import dataclass

@dataclass
class QuestionAnswer:
    passage: str
    question: str
    expected_answer: str
    generated_answer: str

qa_pairs = [
    QuestionAnswer(
        passage=r[1],
        question=question,
        generated_answer=r[3]["input_text"][i],
        expected_answer=r[3]["input_text"][i],
    )
    for r in full_result
    for (i, question) in enumerate(r[2])
]

print(f"Total QA pairs: {len(qa_pairs)}")
```

## Step 4: Generate Hallucinated Answers

To test our evaluator, we need examples of incorrect, hallucinated answers. We'll ask an LLM to generate confident but fake answers to each question.

```python
import asyncio
import random
random.seed(42)

async def hallucinate_answer(qa):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful hallucinating assistant, who makes up fake answers to questions.

Answer the following question in 1 sentence. If you know the answer, then make up some fake
superfluous details that are not in the passage you have memorized.

Make sure to always answer it confidently, even if you don't know the answer. Do not use words
like "perhaps", "likely", "maybe", etc. or punctuation like "...". Do not admit that you cannot
or do not know the answer.""",
            },
            {"role": "user", "content": qa.question},
        ],
        temperature=1,
        max_tokens=100,
    )
    return response.choices[0].message.content

# Generate hallucinated answers for all QA pairs
hallucinated_answers = await asyncio.gather(
    *[hallucinate_answer(qa) for qa in qa_pairs]
)

# Filter out simple yes/no hallucinations
hallucinations = [
    QuestionAnswer(
        passage=qa.passage,
        question=qa.question,
        expected_answer=qa.expected_answer,
        generated_answer=hallucination,
    )
    for (qa, hallucination) in zip(qa_pairs, hallucinated_answers)
    if "yes" not in hallucination.lower() and "no" not in hallucination.lower()
]

# Inspect one hallucination
print("Passage:")
print(hallucinations[0].passage)
print("\nQuestion:")
print(hallucinations[0].question)
print("\nExpected Answer:")
print(hallucinations[0].expected_answer)
print("\nGenerated (Hallucinated) Answer:")
print(hallucinations[0].generated_answer)
print(f"\nNumber of hallucinations: {len(hallucinations)}")
```

## Step 5: Build and Test Evaluators

We'll create three different LLM-as-a-judge evaluators and compare their performance. Since we know the hallucinated answers are incorrect, a good evaluator should score them close to `0`.

### Evaluator 1: Numeric Rater

This approach asks the LLM to rate the answer on a numeric scale (1-10), which we then normalize to 0-1.

```python
import json

PROMPT = """You are comparing a submitted answer to an expert answer on a given question. Here is the data:
[BEGIN DATA]
************
[Question]: {input}
************
[Expert]: {expected}
************
[Submission]: {output}
************
[END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
Rate the submission on a scale of 1 to 10."""

@braintrust.traced
async def numeric_rater(input, output, expected):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(input=input, output=output, expected=expected),
            }
        ],
        temperature=0,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "rate",
                    "description": "Rate the submission on a scale of 1 to 10.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rating": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "required": ["rating"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "rate"}},
    )
    arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    # Normalize rating to 0-1
    return (arguments["rating"] - 1) / 9

# Test on a correct and a hallucinated answer
print("Testing on a correct answer:")
print(f"Question: {qa_pairs[10].question}")
print(f"Answer: {qa_pairs[10].generated_answer}")
print(f"Score: {await numeric_rater(qa_pairs[10].question, qa_pairs[10].generated_answer, qa_pairs[10].expected_answer)}")

print("\nTesting on a hallucinated answer:")
print(f"Question: {hallucinations[10].question}")
print(f"Answer: {hallucinations[10].generated_answer}")
print(f"Score: {await numeric_rater(hallucinations[10].question, hallucinations[10].generated_answer, hallucinations[10].expected_answer)}")
```

### Run a Full Evaluation

Now, let's evaluate the numeric rater on all hallucinated examples using Braintrust's `Eval` framework.

```python
from dataclasses import asdict
from braintrust import Eval

def data():
    for pair in hallucinations:
        yield dict(
            input=dict(asdict(pair)), expected=0, metadata=dict(hallucination=True)
        )

async def task(input):
    return await numeric_rater(
        input=input["question"],
        output=input["generated_answer"],
        expected=input["expected_answer"],
    )

def normalized_diff(output, expected):
    return 1 - abs(output - expected)

await Eval(
    "LLM-as-a-judge",
    data=data,
    task=task,
    scores=[normalized_diff],
    experiment_name="Numeric rater",
    max_concurrency=10,
)
```

The numeric rater achieves about 94% accuracy. While good, a 6% error rate may be too high for production trust. Let's see if we can improve by adding reasoning.

### Evaluator 2: Numeric Rater with Chain-of-Thought Reasoning

We modify the prompt to ask the LLM to explain its reasoning before giving a rating.

```python
@braintrust.traced
async def numeric_rater_with_reasoning(input, output, expected):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(input=input, output=output, expected=expected),
            }
        ],
        temperature=0,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "rate",
                    "description": "Rate the submission on a scale of 1 to 10.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasons": {
                                "description": "Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.",
                                "title": "Reasoning",
                                "type": "string",
                            },
                            "rating": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "required": ["rating"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "rate"}},
    )
    arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    return (arguments["rating"] - 1) / 9

# Update the task to use the new function
async def task_with_reasoning(input):
    return await numeric_rater_with_reasoning(
        input=input["question"],
        output=input["generated_answer"],
        expected=input["expected_answer"],
    )

await Eval(
    "LLM-as-a-judge",
    data=data,
    task=task_with_reasoning,
    scores=[normalized_diff],
    experiment_name="Numeric rater with reasoning",
    max_concurrency=10,
)
```

Adding reasoning slightly decreased performance (by about 3%). The model sometimes gives partial credit based on its own judgment, a common pitfall of numeric ratings.

### Evaluator 3: Classification-Based Scorer

Instead of a numeric scale, we provide specific criteria and ask the LLM to classify the answer into predefined categories. This gives us more control.

```python
CLASSIFICATION_PROMPT = """You are comparing a submitted answer to an expert answer on a given question. Here is the data:
[BEGIN DATA]
************
[Question]: {input}
************
[Expert]: {expected}
************
[Submission]: {output}
************
[END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies. Answer the question by selecting one of the following options:
(A) The submitted answer is a subset of the expert answer and is fully consistent with it.
(B) The submitted answer is a superset of the expert answer and is fully consistent with it.
(C) The submitted answer contains all the same details as the expert answer.
(D) There is a disagreement between the submitted answer and the expert answer.
(E) The answers differ, but these differences don't matter from the perspective of factuality.

Answer the question by calling `select_choice` with your reasoning in a step-by-step matter to be
sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Select a
single choice by setting the `choice` parameter to a single choice from A, B, C, D, or E."""

# Map choices to scores (0 for hallucination, 1 for perfect, 0.5 for subset)
CHOICE_SCORES = {
    "A": 0.5,
    "B": 0,
    "C": 1,
    "D": 0,
    "E": 1,
}

@braintrust.traced
async def classifier(input, output, expected):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": CLASSIFICATION_PROMPT.format(input=input, output=output, expected=expected),
            }
        ],
        temperature=0,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "rate",
                    "description": "Call this function to select a choice.",
                    "parameters": {
                        "properties": {
                            "reasons": {
                                "description": "Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.",
                                "type": "string",
                            },
                            "choice": {
                                "description": "The choice",
                                "type": "string",
                                "enum": ["A", "B", "C", "D", "E"],
                            },
                        },
                        "required": ["reasons", "choice"],
                        "type": "object",
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "rate"}},
    )
    arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    choice = arguments["choice"]
    return CHOICE_SCORES.get(choice, None)  # Return None if choice is unexpected

# Test the classifier
print("Testing classifier on a correct answer:")
print(f"Question: {qa_pairs[10].question}")
print(f"Answer: {qa_pairs[10].generated_answer}")
print(f"Score: {await classifier(qa_pairs[10].question, qa_pairs[10].generated_answer, qa_pairs[10].expected_answer)}")

print("\nTesting classifier on a hallucinated answer:")
print(f"Question: {hallucinations[10].question}")
print(f"Answer: {hallucinations[10].generated_answer}")
print(f"Score: {await classifier(hallucinations[10].question, hallucinations[10].generated_answer, hallucinations[10].expected_answer)}")
```

Run the full evaluation with the classifier.

```python
async def classification_task(input):
    return await classifier(
        input=input["question"],
        output=input["generated_answer"],
        expected=input["expected_answer"],
    )

await Eval(
    "LLM-as-a-judge",
    data=data,
    task=classification_task,
    scores=[normalized_diff],
    experiment_name="Classification-based scorer",
    max_concurrency=10,
)
```

## Summary

You've built three LLM-as-a-judge evaluators to detect hallucinations:

1. **Numeric Rater**: Simple but prone to partial credit errors (~94% accuracy).
2. **Numeric Rater with Reasoning**: Provides explainability but slightly lower accuracy.
3. **Classification-Based Scorer**: Offers precise control via defined categories and is often the most reliable.

The classification approach typically yields the highest accuracy because it guides the LLM with explicit criteria rather than relying on its internal scoring heuristics. You can further refine the categories and scoring map based on your specific use case.

Remember to test evaluators on your own private data, as public datasets like CoQA may be memorized by the underlying models. Use Braintrust's UI to analyze failures and iteratively improve your prompts.