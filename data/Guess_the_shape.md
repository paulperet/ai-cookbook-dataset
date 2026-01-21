# Guide: Using Multiple Images in a Prompt with the Gemini API

This tutorial demonstrates how to use the Gemini API to analyze a sequence of images within a single prompt. You will show the model three shapes (triangle, square, pentagon) and ask it to predict the next shape in the sequence.

## Prerequisites

Before you begin, ensure you have the following:

1.  A Google AI API key. You can obtain one from [Google AI Studio](https://makersuite.google.com/app/apikey).
2.  The API key stored securely. In Google Colab, you can use the "Secrets" tool (`userdata`). For local execution, use environment variables.

## Step 1: Install and Import Required Libraries

First, install the `google-genai` Python client library.

```bash
pip install -U -q "google-genai>=1.0.0"
```

Now, import the necessary modules and configure your client with the API key.

```python
from google import genai
from google.colab import userdata  # For Colab. Use `os.getenv` locally.
import PIL.Image
from IPython.display import display, Markdown

# Set your API key
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')  # Or: os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)

# Verify the library version (optional)
print(f"Using google-genai version: {genai.__version__}")
```

## Step 2: Download the Sample Images

For this example, we'll use three simple shape images hosted publicly. You will download them to your local environment.

```bash
curl -o triangle.png "https://storage.googleapis.com/generativeai-downloads/images/triangle.png" --silent
curl -o square.png "https://storage.googleapis.com/generativeai-downloads/images/square.png" --silent
curl -o pentagon.png "https://storage.googleapis.com/generativeai-downloads/images/pentagon.png" --silent
```

## Step 3: Load and Inspect the Images

Load the downloaded images using the Python Imaging Library (PIL) to ensure they are ready for use.

```python
# Load the images
triangle = PIL.Image.open('triangle.png')
square = PIL.Image.open('square.png')
pentagon = PIL.Image.open('pentagon.png')

# Display them (optional, for verification)
print("Triangle:")
display(triangle)
print("Square:")
display(square)
print("Pentagon:")
display(pentagon)
```

## Step 4: Define the Model and Prompt

Choose a Gemini model. For this task, a Flash model is sufficient and cost-effective. Then, craft your prompt.

```python
# Select your model
MODEL_ID = "gemini-2.5-flash-lite"  # You can also use "gemini-2.5-flash" or "gemini-2.5-pro"

# Define the instructional prompt
prompt = """
Look at this sequence of three shapes. What shape should come as the fourth shape? Explain
your reasoning with detailed descriptions of the first shapes.
"""
```

## Step 5: Generate the Content Response

Combine the prompt text and the images into a single `contents` list, then send the request to the Gemini API.

```python
# Assemble the prompt contents (text + images)
contents = [
    prompt,
    triangle,
    square,
    pentagon
]

# Call the API
response = client.models.generate_content(
    model=MODEL_ID,
    contents=contents
)

# Display the model's response formatted as Markdown
display(Markdown(response.text))
```

## Step 6: Interpret the Result

The model should analyze the sequence and provide a logical prediction. A correct response would follow this reasoning:

1.  **Triangle:** 3 sides.
2.  **Square:** 4 sides.
3.  **Pentagon:** 5 sides.
4.  **Pattern:** The number of sides increases by one each time.
5.  **Prediction:** The next shape should have 6 sides, which is a **hexagon**.

Your output will look similar to this:

> Okay, let's analyze the sequence and predict the next shape.
>
> **Shapes and their characteristics:**
> *   **First Shape:** A triangle. It has three sides and three angles.
> *   **Second Shape:** A square. It has four sides and four angles.
> *   **Third Shape:** A pentagon. It has five sides and five angles.
>
> **Reasoning:**
> The sequence shows a progression where each shape adds one more side and angle than the previous shape.
>
> **Prediction:**
> Following this pattern, the next shape should have six sides and six angles. Therefore, the fourth shape in the sequence should be a **hexagon**.

## Summary and Best Practices

You have successfully used the Gemini API to process a multi-modal prompt containing both text and multiple images. The model identified the visual pattern and provided a reasoned answer.

**Key Takeaway:** You can pass images directly in the prompt `contents` list alongside text. This method is ideal for prototyping and when your total prompt size is manageable.

**For Production & Larger Files:** If your prompts (including images, videos, or PDFs) exceed 100MB, consider using the [Files API](https://github.com/google-gemini/cookbook/tree/main/quickstarts/file-api) to upload your media in advance and reference them by URI. This approach improves efficiency and reliability for larger payloads.