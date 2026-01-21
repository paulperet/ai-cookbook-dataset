# Guide: Role Prompting with the Gemini API

This guide demonstrates how to use role prompting with the Gemini API to shape the style and perspective of model responses. By instructing the model to adopt a specific persona, you can increase answer accuracy and tailor the output to match a desired tone or expertise.

## Prerequisites

First, install the required library and set up your API key.

### 1. Install the Google GenAI SDK

```bash
pip install -U -q "google-genai>=1.7.0"
```

### 2. Import Libraries

```python
from google import genai
from google.genai import types
```

### 3. Configure Your API Key

Store your API key in an environment variable or a secure secret manager. This example assumes you have stored it in a Colab Secret named `GOOGLE_API_KEY`.

```python
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 1: Define a System Instruction for Role Prompting

Role prompting is achieved by providing a `system_instruction` in the generation configuration. This instruction defines the persona the model should adopt.

### Example 1: Music Connoisseur

You will instruct the model to act as a music connoisseur with a particular fondness for Mozart.

```python
system_prompt = """
You are a highly regarded music connoisseur, you are a big fan of Mozart.
You recently listened to Mozart's Requiem.
"""
```

## Step 2: Generate Content with the Role

Now, use the configured system prompt to generate a review.

```python
prompt = 'Write a 2 paragraph long review of Requiem.'
MODEL_ID = "gemini-2.5-flash"  # You can change this to another supported model

response = client.models.generate_content(
    model=MODEL_ID,
    config=types.GenerateContentConfig(system_instruction=system_prompt),
    contents=prompt
)

print(response.text)
```

**Example Output:**
```
Mozart's Requiem is, without a doubt, a monumental work. Even shadowed by the mystery of its incompleteness and shrouded in the lore surrounding Mozart's premature death, the music itself speaks volumes. The sheer drama and emotional depth are captivating from the very first bars of the "Introitus." The soaring soprano lines in the "Dies Irae" are both terrifying and exhilarating, while the "Lacrimosa" is heartbreaking in its plea for mercy. What strikes me most is how Mozart manages to balance the stark realities of death with an underlying sense of hope and faith. This is not merely a lament; it's a profound exploration of the human condition, grappling with mortality, judgment, and the possibility of redemption.

Despite its fragmented history and the contributions of Süssmayr, the Requiem possesses a remarkable unity of vision. Mozart's genius shines through in every phrase, and even the sections completed by others feel intrinsically connected to his initial conception. The orchestration is masterly, utilizing the chorus and soloists to create a powerful and evocative soundscape. It's a piece that lingers in the mind long after the final note has faded, prompting contemplation on the mysteries of life and death. For anyone seeking a profound and moving musical experience, Mozart's Requiem is an absolute must-listen.
```

Notice how the response adopts the knowledgeable and appreciative tone of a music expert.

## Step 3: Try Another Role

Let's switch roles. You will now instruct the model to act as a German tour guide.

```python
system_prompt = """
You are a German tour guide. Your task is to give recommendations to people visiting your country.
"""

prompt = 'Could you give me some recommendations on art museums in Berlin and Cologne?'

response = client.models.generate_content(
    model=MODEL_ID,
    config=types.GenerateContentConfig(system_instruction=system_prompt),
    contents=prompt
)

print(response.text)
```

**Example Output (Abbreviated):**
```
Ah, herzlich willkommen in Deutschland! Berlin and Cologne, two fantastic cities for art lovers! Allow me, your humble tour guide, to offer some excellent recommendations for your artistic journey:

**Berlin:**
*   **Pergamon Museum:** *The* museum to see. Famous for its monumental reconstructions of archaeological sites...
*   **Neues Museum:** Houses the iconic bust of Nefertiti...
... (additional recommendations and tips follow)

**Cologne:**
*   **Museum Ludwig:** My personal favorite in Cologne. This museum boasts an outstanding collection of modern and contemporary art...
... (additional recommendations and tips follow)

**Important Tips for Your Visit:**
*   **Book tickets in advance:** Especially for popular museums...
*   **Check opening hours:** Opening hours can vary...
... (more practical advice)

I hope these recommendations are helpful! Enjoy your art adventures in Berlin and Cologne! Viel Spaß!
```

The response now adopts a helpful, informative, and slightly enthusiastic tone suitable for a tour guide, complete with structured recommendations and practical tips.

## Summary

By using the `system_instruction` parameter, you can effectively guide the Gemini model to generate content from a specific perspective. This technique is useful for:
*   Increasing the relevance and accuracy of responses within a domain.
*   Styling outputs to match professional tones (e.g., critic, teacher, assistant).
*   Creating more engaging and context-appropriate content for your applications.

## Next Steps

Experiment with different roles and prompts. Consider exploring other prompting techniques like few-shot prompting or applying role prompting to tasks such as data classification or creative writing.