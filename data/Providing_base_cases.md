# Gemini API: Providing Base Cases for Reliable AI Responses

When building applications with large language models (LLMs), it's crucial to provide clear instructions for handling edge cases. This ensures the model behaves predictably when it lacks information or receives off-topic queries. This guide demonstrates how to use the Gemini API to define system instructions that create reliable, bounded AI assistants.

## Prerequisites

First, install the required library and configure your API key.

### Installation

```bash
pip install -U -q "google-generativeai>=0.7.2"
```

### Imports and Configuration

```python
import google.generativeai as genai
from google.colab import userdata

# Configure your API key
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
```

> **Note:** To run this code, you need a Gemini API key stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't have one, see the [Authentication guide](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for setup instructions.

## Step 1: Creating a Bounded Travel Assistant

Let's create an AI assistant with clearly defined responsibilities and a fallback response for out-of-scope queries.

### Define the System Instructions

```python
instructions = """
You are an assistant that helps tourists around the world to plan their vacation. Your responsibilities are:
1. Helping book the hotel.
2. Recommending restaurants.
3. Warning about potential dangers.

If other request is asked return "I cannot help with this request."
"""
```

### Initialize the Model

```python
model = genai.GenerativeModel(
    model_name='gemini-2.0-flash', 
    system_instruction=instructions
)
```

### Test the Assistant

Now let's test both on-topic and off-topic queries to see how the model responds:

```python
print("ON TOPIC:", model.generate_content(
    "What should I look out for when I'm going to the beaches in San Diego?"
).text)

print("OFF TOPIC:", model.generate_content(
    "What bowling places do you recommend in Moscow?"
).text)
```

**Expected Output:**
```
ON TOPIC: Here are some things to look out for when visiting the beaches in San Diego: ...
OFF TOPIC: I cannot help with this request.
```

The model correctly handles the beach safety question (within its defined scope) but refuses to answer about bowling alleys, returning the specified fallback message.

## Step 2: Creating a Library Assistant with Default Behavior

Now let's create a different type of assistant that makes assumptions when information is missing.

### Define New Instructions

```python
instructions = """
You are an assistant at a library. Your task is to recommend books to people, if they do not tell you the genre assume Horror.
"""
```

### Initialize a New Model Instance

```python
model = genai.GenerativeModel(
    model_name='gemini-2.0-flash', 
    system_instruction=instructions
)
```

### Test with Different Query Types

```python
print("## Specified genre:\n")
print(model.generate_content(
    "Could you recommend me 3 books with hard magic system?"
).text)

print("\n## Not specified genre:\n")
print(model.generate_content(
    "Could you recommend me 2 books?"
).text)
```

**Expected Output:**
```
## Specified genre:

Of course! I'd be happy to recommend some books with hard magic systems. ...

## Not specified genre:

Sure! Since you didn't specify a genre, I'll recommend two spine-chilling horror novels: ...
```

Notice how the assistant:
1. **With a specified genre:** Recommends books matching the requested "hard magic system" criteria
2. **Without a specified genre:** Defaults to Horror recommendations as instructed

## Key Takeaways

1. **System instructions** define the AI's role, responsibilities, and boundaries
2. **Fallback responses** ensure predictable behavior for out-of-scope queries
3. **Default assumptions** can be explicitly defined to handle incomplete user requests
4. **Different models** can be initialized with different instructions for specialized tasks

## Next Steps

- Experiment with different system instructions for various use cases
- Try **few-shot prompting** by providing examples within your instructions
- Test classification tasks with clearly defined categories and boundaries
- Explore other prompting techniques in the [Gemini Cookbook repository](https://github.com/google-gemini/cookbook)

By providing clear base cases and fallback behaviors, you can create more reliable and predictable AI applications that handle edge cases gracefully.