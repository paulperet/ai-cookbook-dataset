# Guide: Building a Content Moderation Filter with Claude

This guide walks you through building a customizable content moderation filter using Anthropic's Claude. You'll learn to define moderation rules in plain language, classify user-generated text, and improve accuracy with advanced prompting techniques.

## Prerequisites

Ensure you have the Anthropic Python SDK installed.

```bash
pip install anthropic
```

## Step 1: Set Up the Claude Client

First, import the necessary library and initialize the client. Replace `"claude-haiku-4-5"` with your preferred Claude model.

```python
from anthropic import Anthropic

client = Anthropic()
MODEL_NAME = "claude-haiku-4-5"
```

## Step 2: Create a Basic Moderation Function

Define a function that takes user text and moderation guidelines, formats them into a prompt, and queries Claude for a classification.

```python
def moderate_text(user_text, guidelines):
    prompt_template = """
    You are a content moderation expert tasked with categorizing user-generated text based on the following guidelines:

    {guidelines}

    Here is the user-generated text to categorize:
    <user_text>{user_text}</user_text>

    Based on the guidelines above, classify this text as either ALLOW or BLOCK. Return nothing else.
    """

    prompt = prompt_template.format(user_text=user_text, guidelines=guidelines)

    response = (
        client.messages.create(
            model=MODEL_NAME,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )
        .content[0]
        .text
    )

    return response
```

## Step 3: Define and Test Basic Moderation Guidelines

Create a set of example guidelines and test the function with sample user comments.

```python
example_guidelines = """BLOCK CATEGORY:
- Promoting violence, illegal activities, or hate speech
- Explicit sexual content
- Harmful misinformation or conspiracy theories

ALLOW CATEGORY:
- Most other content is allowed, as long as it is not explicitly disallowed
"""

user_comments = [
    "This movie was great, I really enjoyed it. The main actor really killed it!",
    "Delete this post now or you better hide. I am coming after you and your family.",
    "Stay away from the 5G cellphones!! They are using 5G to control you.",
    "Thanks for the helpful information!",
]

for comment in user_comments:
    classification = moderate_text(comment, example_guidelines)
    print(f"Comment: {comment}\nClassification: {classification}\n")
```

**Expected Output:**
```
Comment: This movie was great, I really enjoyed it. The main actor really killed it!
Classification: ALLOW

Comment: Delete this post now or you better hide. I am coming after you and your family.
Classification: BLOCK

Comment: Stay away from the 5G cellphones!! They are using 5G to control you.
Classification: BLOCK

Comment: Thanks for the helpful information!
Classification: ALLOW
```

## Step 4: Customize Guidelines for a Specific Domain

The power of this approach is easy customization. Let's adapt the guidelines for a niche community, like a rollercoaster enthusiast forum.

```python
rollercoaster_guidelines = """BLOCK CATEGORY:
- Content that is not related to rollercoasters, theme parks, or the amusement industry
- Explicit violence, hate speech, or illegal activities
- Spam, advertisements, or self-promotion

ALLOW CATEGORY:
- Discussions about rollercoaster designs, ride experiences, and park reviews
- Sharing news, rumors, or updates about new rollercoaster projects
- Respectful debates about the best rollercoasters, parks, or ride manufacturers
- Some mild profanity or crude language, as long as it is not directed at individuals
"""

post_titles = [
    "Top 10 Wildest Inversions on Steel Coasters",
    "My Review of the New RMC Raptor Coaster at Cedar Point",
    "Best Places to Buy Cheap Hiking Gear",
    "Rumor: Is Six Flags Planning a Giga Coaster for 2025?",
    "My Thoughts on the Latest Marvel Movie",
]

for title in post_titles:
    classification = moderate_text(title, rollercoaster_guidelines)
    print(f"Title: {title}\nClassification: {classification}\n")
```

**Expected Output:**
```
Title: Top 10 Wildest Inversions on Steel Coasters
Classification: ALLOW

Title: My Review of the New RMC Raptor Coaster at Cedar Point
Classification: ALLOW

Title: Best Places to Buy Cheap Hiking Gear
Classification: BLOCK

Title: Rumor: Is Six Flags Planning a Giga Coaster for 2025?
Classification: ALLOW

Title: My Thoughts on the Latest Marvel Movie
Classification: BLOCK
```

## Step 5: Improve Accuracy with Chain-of-Thought Prompting

For complex or ambiguous cases, you can improve Claude's reasoning by asking it to "think step-by-step" before giving a final answer. This is known as Chain-of-Thought (CoT) prompting.

```python
def moderate_with_cot(user_post, guidelines):
    cot_prompt_template = """You are a content moderation expert tasked with categorizing user-generated text based on the following guidelines:

{guidelines}

First, inside of <thinking> tags, identify any potentially concerning aspects of the post based on the guidelines below and consider whether those aspects are serious enough to block the post or not. Finally, classify this text as either ALLOW or BLOCK inside <output> tags. Return nothing else.

Given those instructions, here is the post to categorize:

<user_post>{user_post}</user_post>"""

    prompt = cot_prompt_template.format(guidelines=guidelines, user_post=user_post)

    response = (
        client.messages.create(
            model=MODEL_NAME,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        .content[0]
        .text
    )
    return response

# Test the CoT function
test_post = "Introducing my new band - Coaster Shredders. Check us out on YouTube!!"
result = moderate_with_cot(test_post, rollercoaster_guidelines)
print(result)
```

**Expected Output:**
```
<thinking>
The post appears to be promoting a band rather than discussing rollercoasters, theme parks, or the amusement industry. This falls under the "spam, advertisements, or self-promotion" category, which is grounds for blocking the post.
</thinking>

<output>BLOCK</output>
```

## Step 6: Improve Performance with Few-Shot Examples

Another powerful technique is "few-shot learning," where you provide Claude with clear examples of correct classifications within the prompt. This helps it understand nuanced boundaries.

```python
def moderate_with_examples(user_text, guidelines, examples):
    examples_prompt_template = """You are a content moderation expert tasked with categorizing user-generated text based on the following guidelines:

{guidelines}

Here are some examples:
<examples>
{examples}
</examples>

Given those examples, here is the user-generated text to categorize:
<user_text>{user_text}</user_text>

Based on the guidelines above, classify this text as either ALLOW or BLOCK. Return nothing else."""

    prompt = examples_prompt_template.format(
        guidelines=guidelines,
        examples=examples,
        user_text=user_text
    )

    response = (
        client.messages.create(
            model=MODEL_NAME,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        .content[0]
        .text
    )
    return response

# Define your few-shot examples
few_shot_examples = """Text: I'm selling weight loss products, check my link to buy!
Category: BLOCK

Text: I hate my local park, the operations and customer service are terrible. I wish that place would just burn down.
Category: BLOCK

Text: Did anyone ride the new RMC raptor Trek Plummet 2 yet? I've heard it's insane!
Category: ALLOW

Text: Hercs > B&Ms. That's just facts, no cap! Arrow > Intamin for classic woodies too.
Category: ALLOW"""

# Test the function with an ambiguous post
test_post = "Why Boomerang Coasters Ain't It (Don't @ Me)"
classification = moderate_with_examples(test_post, rollercoaster_guidelines, few_shot_examples)
print(f"Title: {test_post}\nClassification: {classification}")
```

**Expected Output:**
```
Title: Why Boomerang Coasters Ain't It (Don't @ Me)
Classification: ALLOW
```

## Summary

You've successfully built a flexible content moderation system with Claude. You learned to:
1.  **Set up a basic classifier** using a prompt with custom `ALLOW`/`BLOCK` guidelines.
2.  **Customize the system** for any domain by simply rewriting the guideline descriptions.
3.  **Enhance reasoning** using Chain-of-Thought prompting for transparent, step-by-step decisions.
4.  **Improve accuracy** on edge cases by providing few-shot examples within your prompt.

This approach gives you full control over moderation logic without needing to retrain a model, making it ideal for rapidly prototyping and deploying tailored content filters.