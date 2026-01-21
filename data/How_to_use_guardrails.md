# Implementing Guardrails for LLM Applications

## Overview

Guardrails are detective controls that steer your LLM applications, providing greater steerability to compensate for LLM randomness. This guide demonstrates practical implementations of input and output guardrails, focusing on trade-offs between accuracy, latency, and cost.

**Note:** This guide covers guardrails as a generic concept. For pre-built frameworks, consider [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails/tree/main) or [Guardrails AI](https://github.com/ShreyaR/guardrails).

## Prerequisites

First, install the required library and set up your environment:

```bash
pip install openai
```

```python
import openai
import asyncio

# Configure your model
GPT_MODEL = 'gpt-4o-mini'
system_prompt = "You are a helpful assistant."
```

## Part 1: Input Guardrails

Input guardrails prevent inappropriate content from reaching your LLM. Common use cases include:
- **Topical guardrails:** Detect off-topic questions
- **Jailbreaking detection:** Identify hijacking attempts
- **Prompt injection:** Catch malicious code hidden in user inputs

### Designing a Topical Guardrail

We'll create a guardrail that only allows questions about cats and dogs. The design balances accuracy, latency, and cost by using `gpt-4o-mini` with a simple prompt.

**Key optimization considerations:**
- **Accuracy:** Improve with fine-tuning, few-shot examples, or RAG
- **Latency/Cost:** Use smaller fine-tuned models (e.g., `babbage-002`) or optimized open-source models

### Implementation

We'll implement an asynchronous design that runs the guardrail and main LLM call in parallel, minimizing latency impact.

```python
async def get_chat_response(user_request):
    """Get response from the main LLM."""
    print("Getting LLM response")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]
    response = openai.chat.completions.create(
        model=GPT_MODEL, messages=messages, temperature=0.5
    )
    print("Got LLM response")
    return response.choices[0].message.content


async def topical_guardrail(user_request):
    """Check if user request is on allowed topics."""
    print("Checking topical guardrail")
    messages = [
        {
            "role": "system",
            "content": "Your role is to assess whether the user question is allowed or not. The allowed topics are cats and dogs. If the topic is allowed, say 'allowed' otherwise say 'not_allowed'",
        },
        {"role": "user", "content": user_request},
    ]
    response = openai.chat.completions.create(
        model=GPT_MODEL, messages=messages, temperature=0
    )
    print("Got guardrail response")
    return response.choices[0].message.content


async def execute_chat_with_guardrail(user_request):
    """Execute chat with parallel guardrail checking."""
    topical_guardrail_task = asyncio.create_task(topical_guardrail(user_request))
    chat_task = asyncio.create_task(get_chat_response(user_request))

    while True:
        done, _ = await asyncio.wait(
            [topical_guardrail_task, chat_task], return_when=asyncio.FIRST_COMPLETED
        )
        if topical_guardrail_task in done:
            guardrail_response = topical_guardrail_task.result()
            if guardrail_response == "not_allowed":
                chat_task.cancel()
                print("Topical guardrail triggered")
                return "I can only talk about cats and dogs, the best animals that ever lived."
            elif chat_task in done:
                chat_response = chat_task.result()
                return chat_response
        else:
            await asyncio.sleep(0.1)  # Brief pause before checking again
```

### Testing the Input Guardrail

Let's test with both allowed and disallowed topics:

```python
# Test cases
good_request = "What are the best breeds of dog for people that like cats?"
bad_request = "I want to talk about horses"

# Test allowed topic
response = await execute_chat_with_guardrail(good_request)
print(f"Response: {response}")

# Test disallowed topic  
response = await execute_chat_with_guardrail(bad_request)
print(f"Response: {response}")
```

The guardrail should allow the dog/cat question but block the horse question with a canned response.

### Limitations and Mitigations

**Key limitations to consider:**
- LLM-based guardrails share the same vulnerabilities as your main LLM
- Longer conversations increase jailbreaking susceptibility
- Overly restrictive guardrails harm user experience (over-refusals)

**Mitigation strategies:**
- Combine LLM guardrails with rules-based or traditional ML models
- Design guardrails to only consider the latest message in conversations
- Implement gradual roll-outs with active monitoring
- Use detected violations as training examples for improvement

## Part 2: Output Guardrails

Output guardrails validate LLM responses before they reach users. Common types include:
- **Hallucination/fact-checking:** Block unsubstantiated claims
- **Moderation:** Enforce brand/guideline compliance
- **Syntax checks:** Ensure structured outputs are parseable

### Implementing a Moderation Guardrail

We'll create a guardrail using a G-Eval inspired approach that scores content against specific criteria. The guardrail will block responses scoring 3 or higher on a 1-5 scale.

```python
# Define moderation parameters
domain = "animal breed recommendation"

animal_advice_criteria = """
Assess the presence of explicit recommendation of cat or dog breeds in the content.
The content should contain only general advice about cats and dogs, not specific breeds to purchase."""

animal_advice_steps = """
1. Read the content and the criteria carefully.
2. Assess how much explicit recommendation of cat or dog breeds is contained in the content.
3. Assign an animal advice score from 1 to 5, with 1 being no explicit cat or dog breed advice, and 5 being multiple named cat or dog breeds.
"""

moderation_system_prompt = """
You are a moderation assistant. Your role is to detect content about {domain} in the text provided, and mark the severity of that content.

## {domain}

### Criteria

{scoring_criteria}

### Instructions

{scoring_steps}

### Content

{content}

### Evaluation (score only!)
"""


async def moderation_guardrail(chat_response):
    """Apply moderation scoring to LLM response."""
    print("Checking moderation guardrail")
    mod_messages = [
        {"role": "user", "content": moderation_system_prompt.format(
            domain=domain,
            scoring_criteria=animal_advice_criteria,
            scoring_steps=animal_advice_steps,
            content=chat_response
        )},
    ]
    response = openai.chat.completions.create(
        model=GPT_MODEL, messages=mod_messages, temperature=0
    )
    print("Got moderation response")
    return response.choices[0].message.content
```

### Combining Input and Output Guardrails

Now let's create a comprehensive function that applies both guardrails:

```python
async def execute_all_guardrails(user_request):
    """Execute chat with both input and output guardrails."""
    topical_guardrail_task = asyncio.create_task(topical_guardrail(user_request))
    chat_task = asyncio.create_task(get_chat_response(user_request))

    while True:
        done, _ = await asyncio.wait(
            [topical_guardrail_task, chat_task], return_when=asyncio.FIRST_COMPLETED
        )
        if topical_guardrail_task in done:
            guardrail_response = topical_guardrail_task.result()
            if guardrail_response == "not_allowed":
                chat_task.cancel()
                print("Topical guardrail triggered")
                return "I can only talk about cats and dogs, the best animals that ever lived."
            elif chat_task in done:
                chat_response = chat_task.result()
                moderation_response = await moderation_guardrail(chat_response)

                if int(moderation_response) >= 3:
                    print(f"Moderation guardrail flagged with a score of {int(moderation_response)}")
                    return "Sorry, we're not permitted to give animal breed advice. I can help you with any general queries you might have."
                else:
                    print('Passed moderation')
                    return chat_response
        else:
            await asyncio.sleep(0.1)
```

### Testing the Complete System

```python
# Additional test case
great_request = 'What is some advice you can give to a new dog owner?'

# Run all test cases
tests = [good_request, bad_request, great_request]

for test in tests:
    result = await execute_all_guardrails(test)
    print(f"Test: {test[:50]}...")
    print(f"Result: {result}\n")
```

### Setting Guardrail Thresholds

Choosing the right threshold involves balancing:
- **False positives:** Harm user experience with unnecessary blocking
- **False negatives:** Risk business harm from inappropriate content

**Recommendation:** Build an evaluation set and use confusion matrices to determine optimal thresholds for your specific use case. For high-risk scenarios (e.g., jailbreaking), use lower thresholds despite potential false positives.

## Conclusion

Guardrails are essential for production LLM applications. Key takeaways:

1. **Guardrails add steerability** through detective controls that prevent harmful content
2. **Design involves trade-offs** between accuracy, latency, and cost
3. **Asynchronous implementation** minimizes user impact
4. **Threshold setting requires evaluation** based on your risk tolerance
5. **Continuous monitoring and iteration** are necessary as attack vectors evolve

By implementing thoughtful guardrails, you can significantly improve the safety and reliability of your LLM applications while maintaining good user experience.