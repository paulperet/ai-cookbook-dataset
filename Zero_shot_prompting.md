# Gemini API: Zero-shot prompting

You can use the Gemini models to answer many queries without any additional context. Zero-shot prompting is useful for situations when your queries are not complicated and do not require a specific schema.

```
%pip install -U -q "google-genai>=1.0.0"
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.

```
from google import genai
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Examples

Here are a few examples with zero-shot prompting. Note that in each of these examples, you can simply provide the task, with zero examples.

```
MODEL_ID="gemini-2.5-flash" # @param ["gemini-2.5-flash-lite","gemini-2.5-flash","gemini-2.5-pro","gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

```
from IPython.display import Markdown
prompt = """
    Sort following animals from biggest to smallest:
    fish, elephant, dog
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

Markdown(response.text)
```

Here's the order of the animals from biggest to smallest:

1.  **Elephant**
2.  **Dog**
3.  **Fish**

```
prompt = """
    Classify sentiment of review as positive, negative or neutral:
    I go to this restaurant every week, I love it so much.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

Markdown(response.text)
```

Positive

```
prompt = """
    Extract capital cities from the text:
    During the summer I visited many countries in Europe. First I visited Italy, specifically Sicily and Rome.
    Then I visited Cologne in Germany and the trip ended in Berlin.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

Markdown(response.text)
```

Rome and Berlin

```
prompt = """
    Find and fix the error in this Python code:
    def add_numbers(a, b):
        return a + b
    print(add_numbers(5, "10"))
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

Markdown(response.text)
```

```python
def add_numbers(a, b):
    return a + b

print(add_numbers(5, 10))
```

**Error:**

The error is that the code is trying to add an integer (5) to a string ("10").  Python doesn't allow direct addition between integers and strings without explicit type conversion. This causes a `TypeError`.

**Fix:**

To fix this, you can either:

1. **Convert the string "10" to an integer:**  Change `print(add_numbers(5, "10"))` to `print(add_numbers(5, int("10")))`

   *OR*

2. **Pass the second argument as an integer directly:** Change `print(add_numbers(5, "10"))` to `print(add_numbers(5, 10))` (as shown in the corrected code above).

The corrected code shown above takes the second approach, assuming the intention was to add the numbers 5 and 10. This is generally the preferred solution if you can control the input.  If you're getting input from a source that might provide a string, the first approach (using `int()`) might be more robust.

```
prompt = """
    Solve this math problem:
    A train travels 120 km in 2 hours. What is its average speed?
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

Markdown(response.text)
```

Average speed is calculated by dividing the total distance traveled by the total time taken.

*   **Distance:** 120 km
*   **Time:** 2 hours

Average speed = Distance / Time = 120 km / 2 hours = 60 km/h

So, the train's average speed is $\boxed{60}$ km/h.

```
prompt = """
    Identify the names of people, places, and countries in this text:
    Emmanuel Macron, the president of France, announced a AI partnership in collaboration with the United Arab Emirates.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

display(Markdown(response.text))
```

Here's the breakdown of the names in the text:

*   **People:** Emmanuel Macron
*   **Countries:**  France, United Arab Emirates

```
prompt = """
    Correct the grammar in this sentence:
    She don't like playing football but she enjoy to watch it.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

display(Markdown(response.text))
```

Here are a few options for correcting the grammar, depending on the intended meaning:

*   **She doesn't like playing football, but she enjoys watching it.** (This is the most common and likely correct version, fixing the subject-verb agreement and using the correct verb form after "enjoy.")

*   **She doesn't like playing football, but she enjoys to watch.** (While grammatically correct, using "to watch" after enjoy is uncommon. Usually "watching" is preferred.)

**Explanation of the Errors:**

*   **"She don't like"** is incorrect because the third-person singular pronoun "she" requires the verb "doesn't" (does not).
*   **"she enjoy to watch it"** is incorrect. After the verb "enjoy," we generally use the gerund form (verb + -ing) or "watching." While "to watch" is grammatically correct it is not commonly used.

## Next steps

Be sure to explore other examples of prompting in the repository. Try writing prompts about classifying your own data, or try some of the other prompting techniques such as [few-shot prompting](https://github.com/google-gemini/cookbook/blob/main/examples/prompting/Few_shot_prompting.ipynb).