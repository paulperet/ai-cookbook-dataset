

# Gemini API: Authentication Quickstart

The Gemini API uses API keys for authentication. This notebook walks you through creating an API key, and using it with the Python SDK or a command-line tool like `curl`.

## Create an API key

You can [create](https://aistudio.google.com/app/apikey) your API key using Google AI Studio with a single click.

Remember to treat your API key like a password. Don't accidentally save it in a notebook or source file you later commit to GitHub. This notebook shows you two ways you can securely store your API key.

* If you're using Google Colab, it's recommended to store your key in Colab Secrets.

* If you're using a different development environment (or calling the Gemini API through `cURL` in your terminal), it's recommended to store your key in an [environment variable](https://en.wikipedia.org/wiki/Environment_variable).

Let's start with Colab Secrets.

## Add your key to Colab Secrets

Add your API key to the Colab Secrets manager to securely store it.

1. Open your Google Colab notebook and click on the 🔑 **Secrets** tab in the left panel.

2. Create a new secret with the name `GOOGLE_API_KEY`.
3. Copy and paste your API key into the `Value` input box of `GOOGLE_API_KEY`.
4. Toggle the button on the left to allow all notebooks access to the secret.

## Install the Python SDK


```
%pip install -qU 'google-genai>=1.0.0'
```

[First Entry, ..., Last Entry]

## Configure the SDK with your API key

You create a client using your API key, but instead of pasting your key into the notebook, you'll read it from Colab Secrets thanks to `userdata`.


```
from google import genai
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

Now choose a model. The Gemini API offers different models that are optimized for specific use cases. For more information check [Gemini models](https://ai.google.dev/gemini-api/docs/models)


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

And that's it! Now you're ready to call the Gemini API.


```
from IPython.display import Markdown

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Please give me python code to sort a list."
)

display(Markdown(response.text))
```

Python provides incredibly convenient and efficient ways to sort lists. There are two primary built-in functions you'll use:

1.  **`list.sort()`**: This method sorts the list *in-place*, meaning it modifies the original list and doesn't return a new one. It returns `None`.
2.  **`sorted()`**: This built-in function returns a *new* sorted list, leaving the original list unchanged. It can be used on any iterable (lists, tuples, strings, etc.).

Let's look at examples for both.

---

### 1. Using `list.sort()` (Sorts In-Place)

This is suitable when you don't need to preserve the original order of the list.

```python
# --- Basic Sorting (Ascending) ---
my_numbers = [3, 1, 4, 1, 5, 9, 2, 6]
my_numbers.sort() # Modifies my_numbers directly
print("Sorted numbers (in-place):", my_numbers) # Output: [1, 1, 2, 3, 4, 5, 6, 9]

my_strings = ["banana", "apple", "cherry", "date"]
my_strings.sort()
print("Sorted strings (in-place):", my_strings) # Output: ['apple', 'banana', 'cherry', 'date']

# --- Sorting in Descending Order (reverse=True) ---
my_numbers = [3, 1, 4, 1, 5, 9, 2, 6]
my_numbers.sort(reverse=True)
print("Sorted numbers (descending, in-place):", my_numbers) # Output: [9, 6, 5, 4, 3, 2, 1, 1]

# --- Sorting with a Custom Key (e.g., by length of strings) ---
words = ["apple", "banana", "kiwi", "grapefruit", "cat"]
words.sort(key=len) # Sorts by the length of each string
print("Sorted by length (in-place):", words) # Output: ['cat', 'kiwi', 'apple', 'banana', 'grapefruit']

# --- Sorting Case-Insensitively (for strings) ---
names = ["Alice", "bob", "Charlie", "David", "frank"]
names.sort(key=str.lower) # Converts each string to lowercase for comparison
print("Sorted case-insensitively (in-place):", names) # Output: ['Alice', 'bob', 'Charlie', 'David', 'frank']

# --- Sorting a list of dictionaries by a specific value ---
people = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35},
    {"name": "David", "age": 25} # David and Bob have same age
]
people.sort(key=lambda person: person["age"]) # Sorts by the 'age' value
print("Sorted people by age (in-place):", people)
# Output: [{'name': 'Bob', 'age': 25}, {'name': 'David', 'age': 25}, {'name': 'Alice', 'age': 30}, {'name': 'Charlie', 'age': 35}]
# Note: The original relative order of Bob and David (same age) is preserved because Python's sort is stable.
```

---

### 2. Using `sorted()` (Returns a New Sorted List)

This is preferred when you want to keep the original list intact.

```python
# --- Basic Sorting (Ascending) ---
original_numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_numbers = sorted(original_numbers) # Creates a new list
print("Original numbers:", original_numbers) # Output: [3, 1, 4, 1, 5, 9, 2, 6] (unchanged)
print("New sorted numbers:", sorted_numbers) # Output: [1, 1, 2, 3, 4, 5, 6, 9]

original_strings = ["banana", "apple", "cherry", "date"]
new_sorted_strings = sorted(original_strings)
print("Original strings:", original_strings) # Output: ['banana', 'apple', 'cherry', 'date']
print("New sorted strings:", new_sorted_strings) # Output: ['apple', 'banana', 'cherry', 'date']

# --- Sorting in Descending Order (reverse=True) ---
data = [10, 5, 8, 2, 12]
descending_data = sorted(data, reverse=True)
print("New sorted data (descending):", descending_data) # Output: [12, 10, 8, 5, 2]

# --- Sorting with a Custom Key (e.g., by length of strings) ---
fruits = ["strawberry", "pear", "blueberry", "kiwi"]
sorted_by_length = sorted(fruits, key=len)
print("New sorted fruits by length:", sorted_by_length) # Output: ['pear', 'kiwi', 'strawberry', 'blueberry']

# --- Sorting Case-Insensitively (for strings) ---
countries = ["USA", "canada", "Mexico", "france"]
sorted_countries = sorted(countries, key=str.lower)
print("New sorted countries (case-insensitive):", sorted_countries) # Output: ['canada', 'france', 'Mexico', 'USA']

# --- Sorting a list of dictionaries by a specific value ---
students = [
    {"name": "Zoe", "score": 85},
    {"name": "Alex", "score": 92},
    {"name": "Chris", "score": 78},
    {"name": "Ben", "score": 92} # Alex and Ben have same score
]
sorted_students = sorted(students, key=lambda student: student["score"])
print("New sorted students by score:", sorted_students)
# Output: [{'name': 'Chris', 'score': 78}, {'name': 'Zoe', 'score': 85}, {'name': 'Alex', 'score': 92}, {'name': 'Ben', 'score': 92}]
# Again, Python's sort is stable, preserving Alex's and Ben's original relative order.

# --- Sorting a list of tuples (default is lexicographical) ---
points = [(1, 5), (3, 2), (1, 2), (2, 4)]
sorted_points = sorted(points) # Sorts primarily by the first element, then the second
print("Sorted points (lexicographical):", sorted_points) # Output: [(1, 2), (1, 5), (2, 4), (3, 2)]

# --- Sorting a list of tuples by the second element ---
sorted_points_by_y = sorted(points, key=lambda p: p[1])
print("Sorted points by Y-coordinate:", sorted_points_by_y) # Output: [(3, 2), (1, 2), (2, 4), (1, 5)]
```

---

### When to use which:

*   Use **`list.sort()`** if you *don't need the original list* and want to save memory by modifying it in place.
*   Use **`sorted()`** if you *need to preserve the original list* or if you are sorting an iterable that isn't a list (like a tuple, string, or set).

Both methods are very efficient, leveraging Python's highly optimized **Timsort** algorithm.

## Store your key in an environment variable

If you're using a different development environment (or calling the Gemini API through `cURL` in your terminal), it's recommended to store your key in an environment variable.

To store your key in an environment variable, open your terminal and run:

```export GOOGLE_API_KEY="YOUR_API_KEY"```

If you're using Python, you can add these two lines to your notebook to read the key:

```
import os
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
```

Alternatively, if it isn't provided explicitly, the client will look for the API key.

```
client = genai.Client()
```

Or, if you're calling the API through your terminal using `cURL`, you can copy and paste this code to read your key from the environment variable.

```
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$GOOGLE_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{
          "text": "Please give me Python code to sort a list."
        }]
      }]
    }'
```

## Learning more

Now that you know how to manage your API key, you've everything to [get started](./Get_started.ipynb) with Gemini. Check all the [quickstart guides](https://github.com/google-gemini/cookbook/tree/main/quickstarts) from the Cookbook, and in particular the [Get started](./Get_started.ipynb) one.