# Creating Slides with the Assistants API (GPT-4) and DALL·E-3

This guide demonstrates how to use the OpenAI Assistants API (GPT-4) and DALL·E-3 to automate the creation of a professional slide deck. You'll learn how to generate data visualizations, extract key insights, and create a title image—all without touching PowerPoint or Google Slides.

## Prerequisites

Before you begin, ensure you have the following:

*   An OpenAI API key.
*   The required Python libraries installed.

### Setup

First, install the necessary packages and set up your environment.

```bash
pip install openai pandas pillow requests
```

Now, import the required modules and initialize the OpenAI client.

```python
import os
import json
import time
import pandas as pd
import requests
from openai import OpenAI
from PIL import Image
from IPython.display import display

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Helper functions for the Assistants API
def show_json(obj):
    """Helper to display JSON objects."""
    display(json.loads(obj.model_dump_json()))

def submit_message(assistant_id, thread, user_message, file_ids=None):
    """Submits a user message to a thread and starts a run."""
    params = {
        'thread_id': thread.id,
        'role': 'user',
        'content': user_message,
    }
    if file_ids:
        params['file_ids'] = file_ids

    client.beta.threads.messages.create(**params)
    return client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )

def get_response(thread):
    """Retrieves all messages from a thread."""
    return client.beta.threads.messages.list(thread_id=thread.id)
```

## Step 1: Prepare and Analyze Your Data

In this tutorial, we'll create a fictional quarterly financial review for "NotReal Corporation." We'll start by loading and examining the dataset.

### 1.1 Load the Financial Data

The dataset contains quarterly revenue, costs, and customer counts across different distribution channels.

```python
# Load the data
financial_data_path = 'data/NotRealCorp_financial_data.json'
financial_data = pd.read_json(financial_data_path)

# Display the first few rows
print(financial_data.head())
```

### 1.2 Upload the Data File

To allow the Assistant to analyze the data, you must upload it to OpenAI's servers.

```python
# Upload the data file
file = client.files.create(
    file=open('data/NotRealCorp_financial_data.json', "rb"),
    purpose='assistants',
)
```

## Step 2: Create a Data Analyst Assistant

Now, create an Assistant configured as a data scientist with access to the Code Interpreter tool.

```python
# Create the Assistant
assistant = client.beta.assistants.create(
    instructions="You are a data scientist assistant. When given data and a query, write the proper code and create the proper visualization.",
    model="gpt-4-1106-preview",
    tools=[{"type": "code_interpreter"}],
    file_ids=[file.id]
)
```

## Step 3: Generate a Profit Visualization

You'll ask the Assistant to calculate quarterly profits and create a line chart.

### 3.1 Create a Thread and Submit the Task

Start a new conversation thread and provide the initial instruction.

```python
# Create a thread with the initial user request
thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": "Calculate profit (revenue minus cost) by quarter and year, and visualize as a line plot across the distribution channels, where the colors of the lines are green, light red, and light blue.",
            "file_ids": [file.id]
        }
    ]
)
```

### 3.2 Execute the Run

Start the Assistant's execution on the thread.

```python
# Start the run
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
```

### 3.3 Wait for the Plot to be Generated

The Assistant will process the data, calculate profits, and generate the visualization. This may take a minute.

```python
# Poll for completion
while True:
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    try:
        # Check if an image file has been created
        messages.data[0].content[0].image_file
        time.sleep(5)  # Brief pause to ensure the run is complete
        print('Plot created!')
        break
    except:
        time.sleep(10)
        print('Assistant still working...')
```

### 3.4 Retrieve and Save the Generated Plot

Once complete, extract the image file ID and save the plot locally.

```python
# Helper function to save the image
def convert_file_to_png(file_id, write_path):
    """Downloads a file from the Assistants API and saves it as a PNG."""
    data = client.files.content(file_id)
    data_bytes = data.read()
    with open(write_path, "wb") as file:
        file.write(data_bytes)

# Get the latest messages and extract the image file ID
messages = client.beta.threads.messages.list(thread_id=thread.id)
plot_file_id = messages.data[0].content[0].image_file.file_id

# Save the plot
image_path = "NotRealCorp_chart.png"
convert_file_to_png(plot_file_id, image_path)
print(f"Plot saved to {image_path}")
```

**Note:** The Assistant may make several attempts to parse and process the data correctly, demonstrating its adaptability in handling complex tasks.

## Step 4: Extract Key Insights from the Plot

With the visualization ready, ask the Assistant to generate concise, actionable insights suitable for a slide.

### 4.1 Request Insight Bullet Points

```python
# Submit a request for insights
submit_message(
    assistant.id,
    thread,
    "Give me two medium length sentences (~20-30 words per sentence) of the most important insights from the plot you just created. These will be used for a slide deck, and they should be about the 'so what' behind the data."
)

# Wait for the response
time.sleep(10)
response = get_response(thread)
bullet_points = response.data[0].content[0].text.value
print("Insights:\n", bullet_points)
```

### 4.2 Generate a Compelling Slide Title

Now, ask the Assistant to create a brief, impactful title based on the plot and insights.

```python
# Request a title
submit_message(
    assistant.id,
    thread,
    "Given the plot and bullet points you created, come up with a very brief title for a slide. It should reflect just the main insights you came up with."
)

# Wait for the response
time.sleep(10)
response = get_response(thread)
title = response.data[0].content[0].text.value
print("Slide Title:", title)
```

## Step 5: Create a Title Image with DALL·E-3

For the final touch, use DALL·E-3 to generate a custom title image for the presentation.

**Note:** DALL·E-3 is accessed via the Images API, not the Assistants API.

### 5.1 Generate the Image

Provide a brief company description to guide the image generation.

```python
company_summary = "NotReal Corp is a prominent hardware company that manufactures and sells processors, graphics cards and other essential computer hardware."

# Generate an image with DALL·E-3
response = client.images.generate(
    model='dall-e-3',
    prompt=f"Given this company summary: '{company_summary}', create an inspirational photo showing growth and the path forward. This will be used at a quarterly financial planning meeting.",
    size="1024x1024",
    quality="hd",
    n=1
)
image_url = response.data[0].url
print("DALL·E-3 Image URL:", image_url)
```

### 5.2 Download and Save the Image

```python
# Download the image
dalle_img_path = 'dalle_title_image.png'
img_data = requests.get(image_url).content

# Save locally
with open(dalle_img_path, 'wb') as handler:
    handler.write(img_data)

print(f"Title image saved to {dalle_img_path}")
```

## Summary

You have successfully automated the creation of key slide components:
1.  **Data Visualization:** An Assistant calculated profits and generated a professional line chart.
2.  **Narrative:** The same Assistant extracted key insights and crafted a compelling slide title.
3.  **Visual Asset:** DALL·E-3 created a custom title image based on your company description.

You now have a title, a data plot, insightful bullet points, and a title image—all the core elements needed for a professional slide deck, generated automatically with the Assistants API and DALL·E-3.