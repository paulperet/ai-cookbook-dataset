##### Copyright 2025 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: Prompting Quickstart

This notebook contains examples of how to write and run your first prompts with the Gemini API.

```
%pip install -U -q "google-genai>=1.4.0" # 1.4.0 is needed for chat history
```

[First Entry, ..., Last Entry]

### Set up your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.

```
from google.colab import userdata
from google import genai

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Select your model

Now select the model you want to use in this guide, either by selecting one in the list or writing it down. Keep in mind that some models, like the 2.5 ones are thinking models and thus take slightly more time to respond (cf. [thinking notebook](./Get_started_thinking.ipynb) for more details and in particular learn how to switch the thiking off).

```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

## Run your first prompt

Use the `generate_content` method to generate responses to your prompts. You can pass text directly to generate_content, and use the `.text` property to get the text content of the response.

```
from IPython.display import Markdown

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Give me python code to sort a list"
)

display(Markdown(response.text))
```

Python offers very convenient and powerful ways to sort lists, both in-place and by returning a new sorted list.

Here are the primary methods:

1.  **`list.sort()` (In-place sort)**
    *   Modifies the original list directly.
    *   Returns `None`.
    *   Generally more efficient if you don't need the original unsorted list.

2.  **`sorted()` (Returns a new sorted list)**
    *   Returns a *new* sorted list.
    *   Leaves the original list unchanged.
    *   Can be used on any iterable (tuples, sets, strings, etc.), not just lists.

Let's look at examples for both, along with common options like reverse order and custom sorting keys.

---

### 1. `list.sort()` (In-place)

```python
# --- Example 1: Basic sorting of numbers ---
my_list = [3, 1, 4, 1, 5, 9, 2, 6]
print("Original list:", my_list)

my_list.sort() # Sorts in ascending order by default
print("Sorted list (in-place):", my_list)

# --- Example 2: Sorting strings alphabetically ---
fruits = ["orange", "apple", "banana", "grape"]
print("\nOriginal fruits:", fruits)

fruits.sort()
print("Sorted fruits (in-place):", fruits)

# --- Example 3: Sorting in descending (reverse) order ---
numbers = [10, 5, 8, 2, 7]
print("\nOriginal numbers (for reverse):", numbers)

numbers.sort(reverse=True) # Sorts in descending order
print("Sorted numbers (in-place, reverse):", numbers)

# --- Example 4: Sorting with a custom key (e.g., by length of string) ---
words = ["apple", "banana", "kiwi", "orange", "grape"]
print("\nOriginal words (for custom key):", words)

# Sorts by the length of each word
words.sort(key=len)
print("Sorted words (in-place, by length):", words)

# Sorts by the second element of tuples
data = [("apple", 5), ("banana", 2), ("cherry", 8), ("date", 1)]
print("\nOriginal data (for custom key tuple):", data)

data.sort(key=lambda item: item[1]) # Sorts based on the second element (index 1) of each tuple
print("Sorted data (in-place, by 2nd element of tuple):", data)

# Sorting a list of dictionaries by a specific key
people = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
]
print("\nOriginal people (for custom key dict):", people)

people.sort(key=lambda person: person["age"]) # Sorts by the 'age' value
print("Sorted people (in-place, by age):", people)
```

---

### 2. `sorted()` (Returns a new list)

```python
# --- Example 1: Basic sorting of numbers ---
my_list = [3, 1, 4, 1, 5, 9, 2, 6]
print("Original list (for sorted()):", my_list)

new_sorted_list = sorted(my_list) # Returns a new sorted list
print("New sorted list:", new_sorted_list)
print("Original list after sorted():", my_list) # Original list remains unchanged

# --- Example 2: Sorting strings alphabetically ---
fruits = ["orange", "apple", "banana", "grape"]
print("\nOriginal fruits (for sorted()):", fruits)

new_sorted_fruits = sorted(fruits)
print("New sorted fruits:", new_sorted_fruits)

# --- Example 3: Sorting in descending (reverse) order ---
numbers = [10, 5, 8, 2, 7]
print("\nOriginal numbers (for sorted() reverse):", numbers)

new_sorted_numbers_reverse = sorted(numbers, reverse=True)
print("New sorted numbers (reverse):", new_sorted_numbers_reverse)

# --- Example 4: Sorting with a custom key (e.g., by length of string) ---
words = ["apple", "banana", "kiwi", "orange", "grape"]
print("\nOriginal words (for sorted() custom key):", words)

new_sorted_words_by_len = sorted(words, key=len)
print("New sorted words (by length):", new_sorted_words_by_len)

# --- Example 5: Sorting a list of dictionaries by a specific key ---
people = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
]
print("\nOriginal people (for sorted() custom key dict):", people)

new_sorted_people_by_age = sorted(people, key=lambda person: person["age"])
print("New sorted people (by age):", new_sorted_people_by_age)

# --- Example 6: Using sorted() on other iterables (e.g., a tuple) ---
my_tuple = (5, 2, 8, 1, 9)
print("\nOriginal tuple:", my_tuple)

sorted_from_tuple = sorted(my_tuple) # Returns a list, even if input is a tuple
print("Sorted from tuple (new list):", sorted_from_tuple)
```

---

### When to use which?

*   **Use `list.sort()`** when you want to modify the list in place and don't need the original unsorted version. It's generally more memory-efficient as it doesn't create a new list.
*   **Use `sorted()`** when you need a new sorted list and want to keep the original list unchanged. It's also useful when you want to sort an iterable that isn't a list (like a tuple, set, or even a string, though it will return a list of characters).

Both methods are "stable" sorts, meaning if two records have equal keys, their relative order is preserved.

## Use images in your prompt

Here you will download an image from a URL and pass that image in our prompt.

First, you download the image and load it with PIL:

```
!curl -o image.jpg "https://storage.googleapis.com/generativeai-downloads/images/jetpack.jpg"
```

[First Entry, ..., Last Entry]

```
import PIL.Image
img = PIL.Image.open('image.jpg')
img
```

```
prompt = """
    This image contains a sketch of a potential product along with some notes.
    Given the product sketch, describe the product as thoroughly as possible based on what you
   see in the image, making sure to note all of the product features. Return output in json format:
   {description: description, features: [feature1, feature2, feature3, etc]}
"""
```

Then you can include the image in our prompt by just passing a list of items to `generate_content`.

```
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, img],
)

print(response.text)
```

```json
{
  "description": "The \"Jetpack Backpack\" is a conceptual personal flight device disguised as a conventional backpack. It is designed to be lightweight and maintain the appearance of a normal backpack, making it discreet. It offers practical storage, capable of accommodating an 18-inch laptop, and features padded strap support for user comfort. The primary function is flight, powered by steam, which is highlighted as a 'green/clean' energy source. It includes retractable boosters for propulsion and can be recharged via USB-C. The device has a battery life of 15 minutes.",
  "features": [
    "Looks like a normal backpack",
    "Lightweight",
    "Fits 18\" laptop",
    "Padded strap support",
    "Retractable boosters",
    "Steam-powered",
    "Green/Clean (environmental friendly propulsion)",
    "USB-C charging",
    "15-min battery life"
  ]
}
```

## Have a chat

The Gemini API enables you to have freeform conversations across multiple turns.

The [ChatSession](https://ai.google.dev/api/python/google/generativeai/ChatSession) class will store the conversation history for multi-turn interactions.

```
chat = client.chats.create(model=MODEL_ID)
```

```
response = chat.send_message(
    message="In one sentence, explain how a computer works to a young child."
)

print(response.text)
```

A computer is a super-smart helper that follows your instructions very, very fast to play games, show videos, and help you learn new things.

You can see the chat history:

```
messages = chat.get_history()
for message in messages:
  print(f"{message.role}: {message.parts[0].text}")
```

user: In one sentence, explain how a computer works to a young child.
model: A computer is a super-smart helper that follows your instructions very, very fast to play games, show videos, and help you learn new things.

You can keep sending messages to continue the conversation:

```
response = chat.send_message("Okay, how about a more detailed explanation to a high schooler?")

print(response.text)
```

A computer works by taking your input, translating it into binary (0s and 1s), processing that information using its Central Processing Unit (CPU) and temporary memory (RAM), storing data long-term on drives (SSD/HDD), and then outputting the results you see, hear, or feel, all managed by an Operating System.

## Set the temperature

Every prompt you send to the model includes parameters that control how the model generates responses. Use a `types.GenerateContentConfig` to set these, or omit it to use the defaults.

Temperature controls the degree of randomness in token selection. Use higher values for more creative responses, and lower values for more deterministic responses.

Note: Although you can set the `candidate_count` in the generation_config, 2.0 and later models will only return a single candidate at the this time.

```
from google.genai import types

response = client.models.generate_content(
    model=MODEL_ID,
    contents='Give me a numbered list of cat facts.',
    config=types.GenerateContentConfig(
        max_output_tokens=2000,
        temperature=1.9,
        stop_sequences=['\n6'] # Limit to 5 facts.
    )
)

display(Markdown(response.text))
```

Here are some cat facts for you:

1.  Domestic cats spend about **70% of their day sleeping** and 15% grooming.
2.  The average lifespan of an outdoor cat is significantly shorter (2-5 years) compared to an **indoor cat (10-15+ years)**.
3.  Cats use their **whiskers** to "feel" the world around them, gauge openings, and detect changes in air currents. They are highly sensitive tactile organs.
4.  A group of cats is called a **clowder**, a group of kittens is called a kindle.
5.  Cats have a unique scent gland on their paws, which is why they **knead** — it's a way of marking territory and showing contentment.

## Learn more

There's lots more to learn!

* For more fun prompts, check out [Market a Jetpack](https://github.com/google-gemini/cookbook/blob/main/examples/Market_a_Jet_Backpack.ipynb).
* Check out the [safety quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Safety.ipynb) next to learn about the Gemini API's configurable safety settings, and what to do if your prompt is blocked.
* For lots more details on using the Python SDK, check out the [get started notebook](./Get_started.ipynb) or the [documentation's quickstart](https://ai.google.dev/tutorials/python_quickstart).