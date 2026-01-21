# Guide: Getting Reliable JSON Output from Claude

This guide demonstrates techniques for obtaining structured JSON output from Anthropic's Claude models, which do not have a formal "JSON Mode" like some other LLMs. You'll learn how to prompt Claude effectively and parse its responses to extract clean JSON data.

## Prerequisites

First, install the required library and import necessary modules.

```bash
pip install anthropic
```

```python
import json
import re
from pprint import pprint
from anthropic import Anthropic
```

Initialize the Anthropic client and specify the model you'll use.

```python
client = Anthropic()
MODEL_NAME = "claude-3-opus-20240229"  # Or your preferred Claude model
```

## Step 1: Understanding Claude's Default Behavior

When you ask Claude for JSON without special prompting, it typically provides a response with explanatory text before the JSON content.

```python
response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Give me a JSON dict with names of famous athletes & their sports.",
        },
    ],
).content[0].text

print(response)
```

Example output:
```
Here is a JSON dictionary with famous athletes and their sports:

{
  "LeBron James": "Basketball",
  "Serena Williams": "Tennis",
  "Lionel Messi": "Soccer",
  "Simone Biles": "Gymnastics",
  "Tom Brady": "Football"
}
```

To extract the JSON from this response, you can use a simple parsing function:

```python
def extract_json(response):
    """Extract JSON from a response that may contain surrounding text."""
    json_start = response.index("{")
    json_end = response.rfind("}")
    return json.loads(response[json_start : json_end + 1])

athletes_data = extract_json(response)
print(athletes_data)
```

## Step 2: Using Prefilled Responses for Cleaner Output

If you want Claude to skip the preamble and start with JSON immediately, you can prefetch the response by providing a partial assistant message that begins with `{`.

```python
response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Give me a JSON dict with names of famous athletes & their sports.",
        },
        {"role": "assistant", "content": "Here is the JSON requested:\n{"},
    ],
).content[0].text

print(response)
```

The response will now start with the JSON content (after your prefilled `{`). To reconstruct the complete JSON:

```python
# Add back the opening brace and extract to the closing brace
complete_json = "{" + response[: response.rfind("}") + 1]
output_data = json.loads(complete_json)
print(output_data)
```

**Note:** This technique forces Claude to begin with JSON, which prevents it from using "Chain of Thought" reasoning before output. Use this when you need clean JSON without explanatory text.

## Step 3: Using XML Tags for Complex JSON Extraction

For more complex prompts that require multiple JSON outputs, you can instruct Claude to wrap each JSON object in specific XML-style tags. This makes extraction straightforward and reliable.

```python
response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": """Give me a JSON dict with the names of 5 famous athletes & their sports.
Put this dictionary in <athlete_sports> tags.

Then, for each athlete, output an additional JSON dictionary. In each of these additional dictionaries:
- Include two keys: the athlete's first name and last name.
- For the values, list three words that start with the same letter as that name.
Put each of these additional dictionaries in separate <athlete_name> tags.""",
        },
        {"role": "assistant", "content": "Here is the JSON requested:"},
    ],
).content[0].text

print(response)
```

Create a helper function to extract content between tags:

```python
def extract_between_tags(tag: str, string: str, strip: bool = False) -> list[str]:
    """Extract all content between specified XML-style tags."""
    ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        ext_list = [e.strip() for e in ext_list]
    return ext_list
```

Now extract and parse the JSON from the tagged response:

```python
# Extract the main athlete-sports dictionary
athlete_sports_text = extract_between_tags("athlete_sports", response)[0]
athlete_sports_dict = json.loads(athlete_sports_text)

# Extract individual athlete name dictionaries
athlete_name_texts = extract_between_tags("athlete_name", response)
athlete_name_dicts = [json.loads(text) for text in athlete_name_texts]

print("Athlete Sports Dictionary:")
pprint(athlete_sports_dict)

print("\nIndividual Athlete Dictionaries:")
pprint(athlete_name_dicts, width=1)
```

## Summary of Techniques

1. **Basic Extraction**: Use string parsing to find JSON between `{` and `}` when Claude adds explanatory text.

2. **Prefilled Responses**: Start Claude's response with `{` to force immediate JSON output, though this eliminates any reasoning preamble.

3. **XML Tag Wrapping**: For complex multi-JSON outputs, instruct Claude to wrap each JSON object in specific tags (`<tag>...</tag>`) for easy extraction.

4. **Stop Sequences**: You can use the `stop_sequences` parameter in the API call to prevent Claude from adding text after the JSON, though this isn't shown in the examples above.

Choose the technique that best fits your use case:
- Use **basic extraction** for simple requests where explanatory text is acceptable.
- Use **prefilled responses** when you need clean JSON without any preamble.
- Use **XML tags** for complex prompts requiring multiple structured outputs.