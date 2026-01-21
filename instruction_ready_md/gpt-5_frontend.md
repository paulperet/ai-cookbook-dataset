# Building Frontend Applications with GPT-5: A Step-by-Step Guide

GPT-5 represents a significant leap forward in frontend development. It excels at generating full-stack applications in a single pass, performing complex refactors with ease, and making precise edits within large codebases.

This guide demonstrates how to leverage GPT-5 for frontend development, providing practical examples and key learnings.

## Prerequisites & Setup

Before you begin, ensure you have the necessary Python packages installed and your OpenAI API key configured.

```bash
pip install openai
```

```python
import os
import re
import webbrowser
import base64
from pathlib import Path

import openai
from openai.types.responses import ResponseInputParam, ResponseInputImageParam

# Initialize the OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

## 1. Creating Helper Functions

To streamline the process of generating and viewing websites, we'll create a set of utility functions.

### 1.1 Fetch a Response from GPT-5
This function sends a prompt to the GPT-5 model and returns the generated text.

```python
def get_response_output_text(input: str | ResponseInputParam):
    response = client.responses.create(
        model="gpt-5",
        input=input,
    )
    return response.output_text
```

### 1.2 Extract HTML from the Response
GPT-5 often returns code within markdown code blocks. This function extracts the HTML content.

```python
def extract_html_from_text(text: str):
    """Extract an HTML code block from text; fallback to first code block, else full text."""
    html_block = re.search(r"```html\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if html_block:
        result = html_block.group(1)
        return result
    any_block = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if any_block:
        result = any_block.group(1)
        return result
    return text
```

### 1.3 Save the Generated HTML
This function saves the extracted HTML to a file in an `outputs/` directory.

```python
def save_html(html: str, filename: str) -> Path:
    """Save HTML to outputs/ directory and return the path."""
    try:
        base_dir = Path(__file__).parent
    except NameError:
        base_dir = Path.cwd()

    folder = "outputs"
    outputs_dir = base_dir / folder
    outputs_dir.mkdir(parents=True, exist_ok=True)

    output_path = outputs_dir / filename
    output_path.write_text(html, encoding="utf-8")
    return output_path
```

### 1.4 Open the File in a Browser
Once saved, this helper opens the HTML file in your default web browser.

```python
def open_in_browser(path: Path) -> None:
    """Open a file in the default browser (macOS compatible)."""
    try:
        webbrowser.open(path.as_uri())
    except Exception:
        os.system(f'open "{path}"')
```

### 1.5 Combine Helpers into a Single Function
For ease of use, we combine all steps into one function.

```python
def make_website_and_open_in_browser(*, website_input: str | ResponseInputParam, filename: str = "website.html"):
    response_text = get_response_output_text(website_input)
    html = extract_html_from_text(response_text)
    output_path = save_html(html, filename)
    open_in_browser(output_path)
```

## 2. Generating Websites from Text Prompts

Now, let's use our helper to generate websites with different themes and styles.

### 2.1 Create a Dark, Retro Gaming Store
We'll start with a simple, one-line prompt to create a landing page with a specific aesthetic.

```python
make_website_and_open_in_browser(
    website_input="Make me landing page for a retro-games store. Retro-arcade noir some might say",
    filename="retro_dark.html",
)
```

**Result:** GPT-5 generates a complete, styled landing page matching the "retro-arcade noir" theme.

### 2.2 Steer the Style to a Lighter, Pastel Theme
Demonstrating steerability, we can completely alter the visual style with a modified prompt.

```python
make_website_and_open_in_browser(
    website_input="Make me landing page for a retro-games store. Make it light, more pastel coloured & flowery (think Mario, not cyberpunk)", 
    filename="retro_light.html"
)
```

**Result:** The new page will have a light, pastel, and playful aesthetic, showcasing GPT-5's ability to interpret nuanced style directives.

## 3. Enhancing Prompts with Image Input

GPT-5 is natively multimodal. You can provide an image of an existing design to guide the generation of new components that match the theme.

### 3.1 Encode an Image for the API
First, create a function to encode a local image file.

```python
def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
```

### 3.2 Generate a Matching Login Page
Provide an image of an existing dashboard and ask GPT-5 to create a complementary login page.

```python
# Replace with the path to your reference image
image_path = "path/to/your/dashboard_screenshot.png"
encoded_image = encode_image(image_path)

input_image: ResponseInputImageParam = {
    "type": "input_image",
    "image_url": f"data:image/png;base64,{encoded_image}",
    "detail": "auto"
}

input: ResponseInputParam = [
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Can you make a login page for this website that maintains the same theme"},
            input_image,
        ],
    }
]

make_website_and_open_in_browser(
    website_input=input, 
    filename="login_page.html"
)
```

**Result:** GPT-5 analyzes the provided image and generates a login page that seamlessly matches the color scheme, layout, and overall "vibe" of the original dashboard.

## 4. Building Interactive Applications

GPT-5 can also generate complete, interactive web applications, including JavaScript logic.

### 4.1 Create a Themed Snake Game
Let's prompt for a fully functional game with a specific visual style.

```python
make_website_and_open_in_browser(
    website_input="Make me a snake game. It should be futuristic, neon, cyberpunk style. Make sure the typography is suitably cool.", 
    filename="snake_game.html"
)
```

**Result:** You will get a complete, playable Snake game with cyberpunk-themed visuals, interactive controls, and cohesive typography—all in a single HTML file.

## 5. Build Your Own Website

Now it's your turn. Use the template below to generate any website you can imagine.

```python
your_prompt = "Make me a portfolio website for a digital artist specializing in 3D animation."

make_website_and_open_in_browser(
    website_input=your_prompt, 
    filename="your_website.html"
)
```

## Summary & Next Steps

This guide demonstrated how GPT-5 can generate diverse, high-quality frontend code from simple prompts. Key takeaways include:
*   **Steerability:** Small changes in your text prompt lead to significant changes in the output style.
*   **Multimodal Input:** Use images as a reference to maintain design consistency across pages.
*   **Complexity:** GPT-5 can generate not just static pages but also interactive applications with embedded logic.

Experiment with different prompts, combine text and image inputs, and iterate on the generated code to rapidly prototype and build your frontend projects.