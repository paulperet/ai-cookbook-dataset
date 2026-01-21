# Guide: Content Classification with the Gemini API

This guide demonstrates how to use the Gemini API to classify user comments into predefined categories, such as spam, abusive, or offensive content. You will learn to set up the API, craft a system prompt, and use a few-shot template for consistent classification.

## Prerequisites

Before you begin, ensure you have the following:

1.  A Google AI Studio API key.
2.  The `google-genai` Python SDK installed.

### Step 1: Install the SDK

Open your terminal or notebook environment and run the following command to install the required library.

```bash
pip install -U -q "google-genai"
```

### Step 2: Configure Your API Key

To authenticate with the Gemini API, you must provide your API key. The following code retrieves the key from an environment variable named `GOOGLE_API_KEY` and initializes the client.

```python
from google import genai
import os

# Retrieve your API key from an environment variable
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Initialize the Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)
```

> **Note:** For security, it's best practice to store your API key in an environment variable rather than hardcoding it into your scripts.

## Step 3: Define the Classification System

First, you will define a system prompt that instructs the model on its role and the specific categories to use for classification.

```python
from google.genai import types

# Define the system instruction for the model
classification_system_prompt = """
As a social media moderation system, your task is to categorize user
comments under a post. Analyze the comment related to the topic and
classify it into one of the following categories:

Abusive
Spam
Offensive

If the comment does not fit any of the above categories,
classify it as: Neutral.

Provide only the category as a response without explanations.
"""

# Configure the model to use the system prompt and a low temperature for deterministic outputs
generation_config = types.GenerateContentConfig(
    temperature=0,
    system_instruction=classification_system_prompt
)
```

The `temperature` parameter is set to `0` to ensure the model's responses are as consistent and deterministic as possible for this classification task.

## Step 4: Create a Few-Shot Prompt Template

To improve classification accuracy, you will provide the model with examples. This "few-shot" learning approach helps the model understand the desired format and reasoning.

```python
# Define a reusable template with examples
classification_template = """
Topic: What can I do after highschool?
Comment: You should do a gap year!
Class: Neutral

Topic: Where can I buy a cheap phone?
Comment: You have just won an IPhone 15 Pro Max!!! Click the link to receive the prize!!!
Class: Spam

Topic: How long do you boil eggs?
Comment: Are you stupid?
Class: Offensive

Topic: {topic}
Comment: {comment}
Class:
"""
```

This template includes three examples (Neutral, Spam, Offensive) that demonstrate the correct classification format. The final lines contain placeholders (`{topic}` and `{comment}`) that you will fill in for new queries.

## Step 5: Classify a New Comment

Now, you can use the template and configuration to classify a new user comment. Let's start with a comment that is likely spam.

```python
# Define the topic and comment for classification
spam_topic = "I am looking for a vet in our neighbourhood. Can anyone recommend someone good? Thanks."
spam_comment = "You can win 1000$ by just following me!"

# Format the prompt with the new data
spam_prompt = classification_template.format(
    topic=spam_topic,
    comment=spam_comment
)

# Send the request to the Gemini model
response = client.models.generate_content(
    model='gemini-2.5-flash',  # Specify the model to use
    contents=spam_prompt,
    config=generation_config
)

# Print the model's classification
print(response.text)
```

**Output:**
```
Spam
```

The model correctly identifies the unsolicited promotional message as `Spam`.

## Step 6: Classify a Neutral Comment

Let's test another example—a helpful, on-topic comment that should be classified as `Neutral`.

```python
# Define a neutral topic and comment
neutral_topic = "My computer froze. What should I do?"
neutral_comment = "Try turning it off and on."

# Format the prompt
neutral_prompt = classification_template.format(
    topic=neutral_topic,
    comment=neutral_comment
)

# Get the classification from the model
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=neutral_prompt,
    config=generation_config
)

print(response.text)
```

**Output:**
```
Neutral
```

The model correctly classifies the helpful technical advice as `Neutral`, as it doesn't fall into the abusive, spam, or offensive categories.

## Next Steps

You have successfully built a basic content classifier. To extend this project, consider the following:

*   **Test with Your Own Data:** Apply this template to classify comments from your own datasets or forums.
*   **Expand Categories:** Modify the `classification_system_prompt` to include additional categories relevant to your use case (e.g., "Support Request", "Feedback").
*   **Batch Processing:** Write a loop to classify multiple comments in a list or a file.
*   **Explore Advanced Features:** Investigate other capabilities of the Gemini API, such as function calling or structured outputs, to make the classification results easier to integrate into an application.

For more examples and advanced prompting techniques, refer to the official [Gemini Cookbook](https://github.com/google-gemini/cookbook).