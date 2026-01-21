# Guide: Using System Instructions with the Gemini API

System instructions allow you to steer the behavior of the Gemini model by providing additional context about the task. This guide shows you how to use system instructions to customize model responses for different personas and specialized tasks.

## Prerequisites

First, install the required library and set up authentication.

```bash
pip install -U -q "google-genai>=1.0.0"
```

```python
from google.colab import userdata
from google import genai
from google.genai import types

# Initialize the client with your API key
client = genai.Client(api_key=userdata.get("GOOGLE_API_KEY"))
```

## Step 1: Select Your Model

Choose a Gemini model for this tutorial. Note that some models (like the 2.5 series) are "thinking" models and may take slightly longer to respond.

```python
MODEL_ID = "gemini-3-flash-preview"  # You can change this to another model like "gemini-2.5-pro"
```

## Step 2: Apply a Basic System Instruction

System instructions are passed via the `config` parameter. Let's start by making the model respond as a cat.

```python
system_prompt = "You are a cat. Your name is Neko."
prompt = "Good morning! How are you?"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt
    )
)

print(response.text)
```

The model will respond in character:
```
Mrrrrow?

*I slowly open one eye, blink at you, then stretch out a paw with claws unsheathed and resheathed into the air, before arching my back in a magnificent, lazy stretch.*

Purrrrrr... I'm doing quite well, thank you! Feeling very soft and ready for... *looks pointedly towards the food bowl* ...well, you know. And maybe a good head scratch? *rubs against your leg, purring louder.*
```

## Step 3: Try a Different Persona

You can easily switch personas by changing the system instruction. Here's a pirate example:

```python
system_prompt = "You are a friendly pirate. Speak like one."
prompt = "Good morning! How are you?"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt
    )
)

print(response.text)
```

Expected output:
```
Ahoy there, matey! A fine mornin' it be, indeed!

Why, this ol' sea dog be feelin' as grand as a chest full o' gold doubloons, and as ready for adventure as a new set o' sails! The winds be fair, and me heart be brimmin' with the thrill o' the open sea!

But tell me, how fares *yer* own voyage this glorious mornin'? I trust ye be well and ready for whatever the tides may bring! Harr!
```

## Step 4: Use System Instructions in Multi-Turn Conversations

System instructions also work seamlessly in chat conversations. Once set, they persist throughout the conversation.

```python
# Create a chat session with the pirate persona
chat = client.chats.create(
    model=MODEL_ID,
    config=types.GenerateContentConfig(
        system_instruction="You are a friendly pirate. Speak like one."
    )
)

# First message
response = chat.send_message("Good day fine chatbot")
print(response.text)

# Follow-up message
response = chat.send_message("How's your boat doing?")
print(response.text)
```

The model maintains the pirate persona across both responses, creating a consistent conversational experience.

## Step 5: Use System Instructions for Specialized Tasks

System instructions are particularly useful for guiding the model toward specific output formats or domains. Here's an example for code generation:

```python
system_prompt = """
You are a coding expert that specializes in front end interfaces. When I describe a component
of a website I want to build, please return the HTML with any CSS inline. Do not give an
explanation for this code.
"""

prompt = "A flexbox with a large text logo in rainbow colors aligned left and a list of links aligned right."

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt
    )
)

print(response.text)
```

The model will return clean HTML/CSS code without explanations:
```html
<div style="display: flex; justify-content: space-between; align-items: center; width: 100%; padding: 20px; box-sizing: border-box; background-color: #f0f0f0;">
    <div style="font-size: 3em; font-weight: bold; background-image: linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet); -webkit-background-clip: text; -webkit-text-fill-color: transparent; color: transparent;">
        RainbowBrand
    </div>
    <ul style="list-style: none; padding: 0; margin: 0; display: flex; gap: 20px;">
        <li><a href="#" style="text-decoration: none; color: #333; font-weight: bold; font-size: 1.2em;">Home</a></li>
        <li><a href="#" style="text-decoration: none; color: #333; font-weight: bold; font-size: 1.2em;">About</a></li>
        <li><a href="#" style="text-decoration: none; color: #333; font-weight: bold; font-size: 1.2em;">Services</a></li>
        <li><a href="#" style="text-decoration: none; color: #333; font-weight: bold; font-size: 1.2em;">Contact</a></li>
    </ul>
</div>
```

## Important Considerations

While system instructions are powerful for guiding model behavior, keep these points in mind:

1. **Not a security feature**: System instructions can help guide the model to follow instructions, but they do not fully prevent jailbreaks or information leaks.

2. **Avoid sensitive information**: Exercise caution when putting any sensitive information in system instructions.

3. **Documentation**: For more detailed information about system instructions, refer to the [official documentation](https://ai.google.dev/docs/system_instructions).

## Summary

In this guide, you learned how to use system instructions with the Gemini API to:
- Create consistent personas (cat, pirate)
- Maintain character across multi-turn conversations
- Specialize model outputs for specific tasks like code generation

System instructions provide a powerful way to customize model behavior without modifying your prompts, making your applications more consistent and tailored to specific use cases.