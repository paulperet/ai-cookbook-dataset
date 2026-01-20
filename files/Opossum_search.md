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

# Opossum search

This notebook contains a simple example of generating code with the Gemini API and Gemini Flash. Just for fun, you'll prompt the model to create a web app called "Opossum Search" that searches Google with "opossum" appended to the query.

> The opossum image above is from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Opossum_2.jpg), and shared under a CC BY-SA 2.5 license.


```
%pip install -q -U "google-genai>=1.0.0"
```

## Set up your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](../quickstarts/Authentication.ipynb) quickstart for an example.


```
from google import genai
from google.genai.types import GenerateContentConfig
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

Prompt the model to generate the web app.


```
instruction = """
    You are a coding expert that specializes in creating web pages based on a user request.
    You create correct and simple code that is easy to understand.
    You implement all the functionality requested by the user.
    You ensure your code works properly, and you follow best practices for HTML programming.
"""
```


```
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
```
```


```
from IPython.display import display, Markdown
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```


```
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=GenerateContentConfig(
        system_instruction=instruction
    )
)
display(Markdown(response.text))
```


Here is the HTML code for your Opossum Search web application:

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

Key points:

*   **Form Submission**: The form now correctly redirects to a Google search.
*   **JavaScript Function**: A JavaScript function `addOpossum()` is used to prepend "opossum" to the search query before submitting to Google.
*   **Styling**: It includes styling to make it look similar to the Google search page.
*   **Opossum Image**:  It uses the provided opossum image URL.
*   **Footer**: Includes the "Powered by Opossum Search" footer.
*   **Single HTML File**: It's contained within a single HTML file.
*   **No Unicode**: Uses standard HTML and avoids special characters.
*   **Visual Similarity**: Includes basic styling to resemble the Google search interface.
*   **Target Blank**: Opens the search result in a new tab.


## Run the output locally

You can start a web server as follows.

* Save the HTML output to a file called `search.html`
* In your terminal run `python3 -m http.server 8000`
* Open your web browser, and point it to `http://localhost:8000/search.html`

## Display the output in IPython

Like all LLMs, the output may not always be correct. You can experiment by rerunning the prompt, or by writing an improved one (and/or better system instructions). Have fun!


```
import IPython
code = response.text.split('```')[1][len('html'):]
IPython.display.HTML(code)
```





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