# Vision with Claude: A Practical Guide to Best Practices

This guide walks you through effective techniques for using Claude's vision capabilities. You'll learn how to structure prompts, improve accuracy, and leverage multiple images.

## Prerequisites

First, install the required packages and set up your environment.

```bash
pip install anthropic IPython
```

```python
import base64
from anthropic import Anthropic

# Initialize the client
client = Anthropic()
MODEL_NAME = "claude-opus-4-1"

def get_base64_encoded_image(image_path):
    """Helper function to encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base64_string = base64.b64encode(binary_data).decode("utf-8")
        return base64_string
```

## 1. Improving Accuracy with Prompt Engineering

Vision models can sometimes misinterpret details. Traditional prompt engineering techniques, like role assignment and chain-of-thought, can significantly improve accuracy.

### 1.1 The Problem: Incorrect Object Counting

Let's start with a simple task: counting dogs in an image. Without specific guidance, Claude might miscount.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": get_base64_encoded_image("../images/best_practices/nine_dogs.jpg"),
                },
            },
            {"type": "text", "text": "How many dogs are in this picture?"},
        ],
    }
]

response = client.messages.create(model=MODEL_NAME, max_tokens=2048, messages=message_list)
print(response.content[0].text)
```

**Output:**
```
The image shows a group of 10 dogs of various breeds...
```

Claude incorrectly counted 10 dogs when there are actually 9.

### 1.2 The Solution: Enhanced Prompting

By refining the prompt with role assignment and explicit reasoning instructions, we guide Claude to be more accurate.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": get_base64_encoded_image("../images/best_practices/nine_dogs.jpg"),
                },
            },
            {
                "type": "text",
                "text": "You have perfect vision and pay great attention to detail which makes you an expert at counting objects in images. How many dogs are in this picture? Before providing the answer in <answer> tags, think step by step in <thinking> tags and analyze every part of the image.",
            },
        ],
    }
]

response = client.messages.create(model=MODEL_NAME, max_tokens=2048, messages=message_list)
print(response.content[0].text)
```

**Output:**
```
<thinking>
To accurately count the number of dogs in this image, I'll visually scan the image from left to right, focusing on each individual dog...
</thinking>
<answer>There are 9 dogs total in this image...</answer>
```

With the enhanced prompt, Claude now correctly identifies 9 dogs. The key improvements were:
- Assigning an expert role ("expert at counting objects")
- Requiring step-by-step reasoning in `<thinking>` tags
- Structuring the final answer in `<answer>` tags

## 2. Visual Prompting: Instructions Within Images

You can embed prompts directly within images using text annotations and visual cues.

### 2.1 Basic Image Description

When you provide an image without context, Claude will describe what it sees.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/best_practices/circle.png"),
                },
            },
        ],
    }
]

response = client.messages.create(model=MODEL_NAME, max_tokens=2048, messages=message_list)
print(response.content[0].text)
```

**Output:**
```
The image shows a simple black circle outline on a white background...
```

### 2.2 Embedded Questions in Images

When you add text questions directly to the image, Claude can interpret and answer them.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/best_practices/labeled_circle.png"),
                },
            },
        ],
    }
]

response = client.messages.create(model=MODEL_NAME, max_tokens=2048, messages=message_list)
print(response.content[0].text)
```

**Output:**
```
The area of the circle is πr^2, where r is the radius of the circle. The question states that the radius is 12, so the area would be π(12)^2 = 144π square units.
```

### 2.3 Referencing Specific Image Elements

You can highlight specific parts of an image and ask questions about them.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/best_practices/table.png"),
                },
            },
            {"type": "text", "text": "What's the difference between these two numbers?"},
        ],
    }
]

response = client.messages.create(model=MODEL_NAME, max_tokens=2048, messages=message_list)
print(response.content[0].text)
```

**Output:**
```
The difference between North America's net sales for the twelve months ended December 31, 2023 ($352,828) and December 31, 2022 ($315,880) is $36,948.
```

## 3. Few-Shot Learning with Images

Providing examples in your prompt helps Claude understand the specific format or context you're working with.

### 3.1 The Problem: Misinterpreting Units

When reading a speedometer, Claude might misinterpret the units without context.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/best_practices/140.png"),
                },
            },
            {"type": "text", "text": "What speed am I going?"},
        ],
    }
]

response = client.messages.create(model=MODEL_NAME, max_tokens=2048, messages=message_list)
print(response.content[0].text)
```

**Output:**
```
The speedometer in the image is showing a speed of 140 kilometers/hour (or about 87 miles/hour).
```

Claude incorrectly assumed kilometers per hour.

### 3.2 The Solution: Providing Examples

By showing Claude examples of how to interpret similar images, we teach it the correct format.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/best_practices/70.png"),
                },
            },
            {"type": "text", "text": "What speed am I going?"},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "You are going 70 miles per hour."}],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/best_practices/100.png"),
                },
            },
            {"type": "text", "text": "What speed am I going?"},
        ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "You are going 100 miles per hour."}],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/best_practices/140.png"),
                },
            },
            {"type": "text", "text": "What speed am I going?"},
        ],
    },
]

response = client.messages.create(
    model=MODEL_NAME, max_tokens=2048, messages=message_list, temperature=0
)
print(response.content[0].text)
```

**Output:**
```
The speedometer in the image shows that you are going 140 miles per hour.
```

With the examples, Claude now correctly interprets the speed in miles per hour. Note that we set `temperature=0` for more deterministic outputs when using few-shot examples.

## 4. Working with Multiple Images

Claude can process and reason across multiple images in a single prompt, which is useful for analyzing large documents or comparing items.

### 4.1 Processing Split Documents

When you have a large image (like a long receipt), you can split it and feed the pieces to Claude.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/best_practices/receipt1.png"),
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/best_practices/receipt2.png"),
                },
            },
            {"type": "text", "text": "Output the name of the restaurant and the total."},
        ],
    }
]

response = client.messages.create(model=MODEL_NAME, max_tokens=2048, messages=message_list)
print(response.content[0].text)
```

**Output:**
```
The name of the restaurant is The Breakfast Club and the total amount on the receipt is $78.86.
```

### 4.2 Object Identification from Examples

You can provide reference images to help Claude identify objects in a target image.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/best_practices/wrinkle.png"),
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/best_practices/officer.png"),
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image("../images/best_practices/chinos.png"),
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": get_base64_encoded_image(
                        "../images/best_practices/officer_example.png"
                    ),
                },
            },
            {
                "type": "text",
                "text": "These pants are (in order) WRINKLE-RESISTANT DRESS PANT, ITALIAN MELTON OFFICER PANT, SLIM RAPID MOVEMENT CHINO. What pant is shown in the last image?",
            },
        ],
    }
]

response = client.messages.create(model=MODEL_NAME, max_tokens=2048, messages=message_list)
print(response.content[0].text)
```

**Output:**
```
The last image shows a person wearing light gray wool dress pants or trousers... these appear to be the Italian Melton Officer pants that were shown in the second product image.
```

## Key Takeaways

1. **Prompt Engineering Matters**: Even with vision models, careful prompt design (role assignment, chain-of-thought) significantly improves accuracy.
2. **Visual Prompting is Powerful**: You can embed questions and instructions directly within images using text annotations.
3. **Few-Shot Learning Works**: Providing examples helps Claude understand specific formats, units, or contexts.
4. **Multiple Images Enable Complex Tasks**: Claude can reason across multiple images, enabling analysis of large documents or comparative identification.
5. **Reference Images Aid Identification**: Providing example images alongside a target image helps with accurate object classification.

Remember that while these techniques generally improve performance, results may vary based on your specific use case. Experiment with different approaches to find what works best for your application.