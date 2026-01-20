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

# Gemini API: Basic code generation

This notebook demonstrates how to use prompting to perform basic code generation using the Gemini API's Python SDK. Two use cases are explored: error handling and code generation.

The Gemini API can be a great tool to save you time during the development process. Tasks such as code generation, debugging, or optimization can be done with the assistance of the Gemini model.


```
%pip install -U -q "google-genai>=1.0.0"
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google import genai
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

Additionally, select the model you want to use from the available options below:


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

## Examples

### Error handling

For code generation, you should prioritize accuracy over creativity.
A temperature of 0 ensures that the generated content is deterministic,
producing the most sensible output every time.


```
from google.genai import types

error_handling_system_prompt =f"""
  Your task is to explain exactly why this error occurred and how to fix it.
"""

error_handling_model_config = types.GenerateContentConfig(
    temperature=0,
    system_instruction=error_handling_system_prompt
)
```


```
from IPython.display import Markdown

error_message = """
    1 my_list = [1,2,3]
  ----> 2 print(my_list[3])

  IndexError: list index out of range
"""

error_prompt = f"""
  You've encountered the following error message:
  Error Message: {error_message}
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=error_prompt,
    config=error_handling_model_config
)

Markdown(response.text)
```

Okay, let's break down this `IndexError: list index out of range` error.

**What the Error Means**

The error `IndexError: list index out of range` specifically tells you that you're trying to access an element in a list using an index (a number representing its position) that doesn't exist within that list.  Think of it like trying to find a page number in a book that's beyond the total number of pages.

**Why the Error Occurred in this Code**

1.  **`my_list = [1, 2, 3]`**: This line creates a list named `my_list` containing the numbers 1, 2, and 3.

2.  **`print(my_list[3])`**: This line attempts to print the element at index 3 of `my_list`.  **Crucially, in Python (and most programming languages), list indices start at 0.**

    *   Index 0 refers to the first element (which is 1).
    *   Index 1 refers to the second element (which is 2).
    *   Index 2 refers to the third element (which is 3).

    Since `my_list` only has three elements (at indices 0, 1, and 2), there is no element at index 3.  Trying to access `my_list[3]` goes "out of range" of the valid indices, hence the error.

**How to Fix the Error**

The fix depends on what you *intended* to do.  Here are a few possibilities and their solutions:

1.  **If you wanted to access the last element:**

    *   You could use index 2 (because the list has 3 elements, and the last element is at index 2):

        ```python
        my_list = [1, 2, 3]
        print(my_list[2])  # Output: 3
        ```

    *   Alternatively, you can use negative indexing.  `-1` refers to the last element, `-2` to the second-to-last, and so on:

        ```python
        my_list = [1, 2, 3]
        print(my_list[-1])  # Output: 3
        ```

2.  **If you wanted to access an element at a specific position (e.g., the fourth element if you *thought* there were four elements):**

    *   You need to make sure your list actually *has* that many elements.  You would need to add more elements to the list:

        ```python
        my_list = [1, 2, 3, 4]  # Now the list has four elements
        print(my_list[3])  # Output: 4
        ```

3.  **If you're unsure how many elements are in the list:**

    *   Use the `len()` function to find the number of elements and then adjust your index accordingly:

        ```python
        my_list = [1, 2, 3]
        list_length = len(my_list)  # list_length will be 3
        if list_length > 3:  # Check if index 3 is valid
            print(my_list[3])
        else:
            print("Index 3 is out of range for this list.")
        ```

**In summary:** The core problem is that you're trying to access an element at an index that doesn't exist in your list.  Carefully examine your code to determine the correct index you want to use or to ensure your list has the necessary number of elements.  Remember that list indices start at 0.

### Code generation


```
code_generation_system_prompt = f"""
  You are a coding assistant. Your task is to generate a code snippet that
  accomplishes a specific goal. The code snippet must be concise, efficient,
  and well-commented for clarity. Consider any constraints or requirements
  provided for the task.

  If the task does not specify a programming language, default to Python.
"""

code_generation_model_config = types.GenerateContentConfig(
    temperature= 0,
    system_instruction=code_generation_system_prompt
  )
```


```
code_generation_prompt = """
  Create a countdown timer that ticks down every second and prints
  "Time is up!" after 20 seconds
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=code_generation_prompt,
    config=code_generation_model_config
)
Markdown(response.text)
```

```python
import time

def countdown_timer(seconds):
    """
    Counts down from a specified number of seconds and prints "Time is up!" when finished.

    Args:
        seconds (int): The number of seconds to count down from.
    """
    for i in range(seconds, 0, -1):  # Iterate from seconds down to 1
        print(f"{i} seconds remaining...")
        time.sleep(1)  # Pause for 1 second
    print("Time is up!")

if __name__ == "__main__":
    countdown_timer(20)  # Start a 20-second countdown
```

Let's check if generated code works.


```
import re
matchFound = re.search(r"python\n(.*?)```", response.text, re.DOTALL)
if matchFound:
  code = matchFound.group(1)
  exec(code)
```

[20 seconds remaining..., ..., 1 seconds remaining..., Time is up!]

## Next steps

Be sure to explore other examples of prompting in the repository. Try writing prompts around your own code as well using the examples in this notebook.