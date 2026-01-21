# Build an "Opossum Search" Web App with Gemini

This guide walks you through using the Gemini API to generate a functional, single-file web application. You'll prompt the model to create "Opossum Search," a Google-like search page that automatically prepends "opossum" to every query.

## Prerequisites

Before you begin, ensure you have the following:

1.  A Google AI API key. If you don't have one, you can [get one here](https://aistudio.google.com/app/apikey).
2.  The API key stored securely. This guide assumes it's stored in an environment variable or a secrets manager. For Google Colab, you would use a Secret named `GOOGLE_API_KEY`.

## Step 1: Install the Required Library

First, install the official Google Generative AI Python SDK.

```bash
pip install -q -U "google-genai>=1.0.0"
```

## Step 2: Configure the API Client

Import the necessary modules and initialize the Gemini client with your API key.

```python
from google import genai
from google.genai.types import GenerateContentConfig

# Replace this with your method of loading the API key.
# For example, using an environment variable:
import os
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Initialize the client
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 3: Craft the System Instruction and Prompt

To guide the model effectively, you will provide both a system instruction defining its role and a detailed user prompt describing the desired application.

Define the system instruction to set the model's behavior as a coding expert.

```python
instruction = """
You are a coding expert that specializes in creating web pages based on a user request.
You create correct and simple code that is easy to understand.
You implement all the functionality requested by the user.
You ensure your code works properly, and you follow best practices for HTML programming.
"""
```

Now, create the specific prompt for the "Opossum Search" app.

```python
prompt = """
Create a web app called Opossum Search:
1. Every time you make a search query, it should redirect you to a Google search
with the same query, but with the word opossum before it.
2. It should be visually similar to Google search.
3. Instead of the google logo, it should have a picture of this opossum:
https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Opossum_2.jpg/292px-Opossum_2.jpg.
4. It should be a single HTML file, with no separate JS or CSS files.
5. It should say Powered by opossum search in the footer.
6. Do not use any unicode characters.
Thank you!
"""
```

## Step 4: Generate the HTML Code

With the prompt ready, you can now call the Gemini model. This example uses the `gemini-3-flash-preview` model for its speed and capability.

```python
MODEL_ID = "gemini-3-flash-preview"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=GenerateContentConfig(
        system_instruction=instruction
    )
)

# Display the generated code in a formatted markdown block
from IPython.display import display, Markdown
display(Markdown(response.text))
```

The model will generate a complete HTML file. The output should look similar to the following code block:

```html
<!DOCTYPE html>
<html>
<head>
<title>Opossum Search</title>
<style>
body {
font-family: Arial, sans-serif;
margin: 0;
display: flex;
flex-direction: column;
align-items: center;
height: 100vh;
}
.container {
text-align: center;
margin-top: 10vh;
}
img {
width: 200px;
margin-bottom: 20px;
}
input[type=text] {
padding: 10px 20px;
width: 500px;
border: 1px solid #ccc;
border-radius: 24px;
font-size: 16px;
}
button {
padding: 10px 20px;
background-color: #f8f9fa;
border: 1px solid #f0f0f1;
color: #3c4043;
border-radius: 4px;
cursor: pointer;
margin: 5px;
}
button:hover {
border: 1px solid #ccc;
}
footer {
margin-top: auto;
text-align: center;
padding: 20px;
font-size: small;
color: #70757a;
}
</style>
</head>
<body>
<div class="container">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Opossum_2.jpg/292px-Opossum_2.jpg" alt="Opossum Logo">
<form action="https://www.google.com/search" method="get" target="_blank" onsubmit="addOpossum()">
<input type="text" id="searchQuery" name="q">
<br>
<button type="submit">Opossum Search</button>
</form>
</div>
<footer>Powered by Opossum Search</footer>
<script>
function addOpossum() {
var query = document.getElementById("searchQuery").value;
document.getElementById("searchQuery").value = "opossum " + query;
}
</script>
</body>
</html>
```

**Key Implementation Details:**
*   **Form Action:** The form submits directly to `https://www.google.com/search`.
*     **JavaScript Logic:** The `addOpossum()` function prepends "opossum " to the search query before the form is submitted.
*   **Styling:** CSS is embedded to mimic the clean, centered layout of Google's homepage.
*   **Single File:** All HTML, CSS, and JavaScript is contained within one file as requested.

## Step 5: Run the Web Application

You have two main options to run and test the generated application.

### Option A: Run a Local Web Server

1.  Save the generated HTML code to a file named `search.html`.
2.  Open a terminal in the directory containing the file.
3.  Start a simple HTTP server:
    ```bash
    python3 -m http.server 8000
    ```
4.  Open your web browser and navigate to `http://localhost:8000/search.html`.

### Option B: Render Directly in IPython (for quick preview)

If you are working in a Jupyter notebook or IPython environment, you can render the HTML directly.

```python
import IPython

# Extract the HTML code from the model's response
# This assumes the code was wrapped in a markdown code block.
code = response.text.split('```')[1][len('html'):]
IPython.display.HTML(code)
```

## Experiment and Iterate

Like all LLMs, the output may not always be perfect on the first try. You can experiment by:
*   **Rerunning the prompt** to get a variation of the code.
*   **Improving the prompt** with more specific instructions (e.g., "Use Flexbox for centering").
*   **Refining the system instruction** to better guide the model's coding style.

Have fun building and customizing your Opossum Search engine!