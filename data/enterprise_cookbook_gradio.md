# Creating Demos with Spaces and Gradio

_Authored by: [Diego Maniloff](https://huggingface.co/dmaniloff)_

## Introduction

This guide demonstrates how to bring any machine learning model to life using [Gradio](https://www.gradio.app/), a library that allows you to create a web demo from any Python function and share it with the world.

**What you will learn:**
- Building a basic "Hello, World!" demo to understand Gradio's core concepts.
- Moving your demo to Hugging Face Spaces for permanent, shareable hosting.
- Creating a real-world application: a meeting transcription tool.
- Exploring powerful built-in Gradio features like APIs and flagging.

## Prerequisites

To get started, install the required libraries.

```bash
pip install gradio==4.36.1
pip install transformers==4.41.2
```

## Step 1: Your First Demo - Gradio Basics

At its core, Gradio turns any Python function into a web interface.

### 1.1 Define a Simple Function

Let's start with a basic function that takes a name and an intensity level, then returns a greeting.

```python
import gradio as gr

def greet(name: str, intensity: int) -> str:
    return "Hello, " + name + "!" * int(intensity)
```

You can test this function directly:

```python
print(greet("Diego", 3))
```

### 1.2 Create a Gradio Interface

We can build a web interface for this function using `gr.Interface`. We need to specify the function and the types of inputs and outputs it expects.

```python
demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"], # A text box and a slider component
    outputs=["text"],          # A text box for the output
)
```

The `inputs` and `outputs` parameters use Gradio **Components** like `"text"` and `"slider"`.

### 1.3 Launch the Demo

Launch the interface to see your first web demo.

```python
demo.launch()
```

A local web server will start. Open the provided link in your browser, type a name, adjust the slider, and click **Submit** to see the function in action.

## Step 2: A Real-World Example - Meeting Transcription Tool

Now, let's build something more practical: a tool that transcribes audio from a meeting and organizes the text. We'll break this into two parts:
1. **Audio-to-Text:** Convert an audio file to raw text.
2. **Text Organization:** Structure and summarize the transcribed text.

### 2.1 Part 1: Transcribe Audio to Text

First, we need a function that handles speech recognition. We'll use the `transformers` library with the `distil-whisper` model for efficient transcription.

```python
import os
import tempfile
import torch
import gradio as gr
from transformers import pipeline

# Set device to GPU if available
device = 0 if torch.cuda.is_available() else "cpu"
AUDIO_MODEL_NAME = "distil-whisper/distil-large-v3"
BATCH_SIZE = 8

# Initialize the speech recognition pipeline
pipe = pipeline(
    task="automatic-speech-recognition",
    model=AUDIO_MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

def transcribe(audio_input):
    """Convert an audio file to text."""
    if audio_input is None:
        raise gr.Error("No audio file submitted!")

    output = pipe(
        audio_input,
        batch_size=BATCH_SIZE,
        generate_kwargs={"task": "transcribe"},
        return_timestamps=True
    )
    return output["text"]
```

**Explanation:**
- The `transcribe` function takes a file path to an audio file.
- It uses a pre-trained Whisper model via the `pipeline` utility.
- If no audio is provided, it raises a user-friendly error.

Now, create a Gradio interface for this function. We'll use the `gr.Audio` component for input.

```python
part_1_demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"), # Accepts an audio file
    outputs=gr.Textbox(show_copy_button=True), # Output with a copy button
    title="Transcribe Audio to Text",
)

part_1_demo.launch()
```

Launch the demo. You can upload an `.mp3` file or record your voice directly.

> **Tip:** For testing, you can use sample meetings from the [MeetingBank_Audio dataset](https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio).

### 2.2 Part 2: Organize and Summarize Text

Next, we need to clean up the raw transcript. We'll use a language model to structure the text. Instead of running a model locally, we'll use the **Hugging Face Serverless Inference API**, which is free and requires no local GPU.

First, authenticate with the Hugging Face Hub.

```python
from huggingface_hub import notebook_login

# This will prompt you for your Hugging Face User Access Token
notebook_login()
```

You can manage your tokens in your [Hugging Face settings](https://huggingface.co/settings/tokens). Use fine-grained tokens for better security.

Now, let's define the text processing function using the Inference API.

```python
from huggingface_hub import InferenceClient

TEXT_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
client = InferenceClient()

def organize_text(meeting_transcript):
    """Send a transcript to an LLM for organization."""
    messages = build_messages(meeting_transcript)
    response = client.chat_completion(
        messages,
        model=TEXT_MODEL_NAME,
        max_tokens=250,
        seed=430  # Seed for reproducibility
    )
    return response.choices[0].message.content

def build_messages(meeting_transcript) -> list:
    """Format the prompt for the LLM."""
    system_input = "You are an assistant that organizes meeting minutes."
    user_input = f"""Take this raw meeting transcript and return an organized version.
    Here is the transcript:
    {meeting_transcript}
    """

    messages = [
        {"role": "system", "content": system_input},
        {"role": "user", "content": user_input},
    ]
    return messages
```

**Explanation:**
- `organize_text` sends the transcript to the specified model via the API.
- `build_messages` structures the prompt with a system instruction and the user's transcript.

Create a demo for this text organization step.

```python
# A sample transcript for the input box
sample_transcript = """
Good evening. Welcome to the Denver City Council meeting of Monday, May 8, 2017...
"""

part_2_demo = gr.Interface(
    fn=organize_text,
    inputs=gr.Textbox(value=sample_transcript, label="Raw Transcript"),
    outputs=gr.Textbox(show_copy_button=True, label="Organized Transcript"),
    title="Clean Up Transcript Text",
)

part_2_demo.launch()
```

Try submitting the sample text. The output will be a cleaner, structured version. Experiment by changing the `user_input` in `build_messages` to request a summary instead.

### 2.3 Combine Both Parts into a Complete Tool

We now have two independent functions. Let's combine them into a single pipeline.

```python
def meeting_transcript_tool(audio_input):
    """The complete pipeline: audio -> text -> organized text."""
    meeting_text = transcribe(audio_input)
    organized_text = organize_text(meeting_text)
    return organized_text

full_demo = gr.Interface(
    fn=meeting_transcript_tool,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Textbox(show_copy_button=True),
    title="The Complete Meeting Transcription Tool",
)

full_demo.launch()
```

This demo takes an audio file as input, transcribes it, and returns the organized text—all in one step.

## Step 3: Deploy Your Demo to Hugging Face Spaces

You've built a functional demo locally. Now, let's deploy it permanently on [Hugging Face Spaces](https://huggingface.co/spaces).

### 3.1 Create a New Space

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
2. Select **Gradio** as the SDK.
3. Choose a name and visibility (public or private), then click **Create Space**.

### 3.2 Prepare Your Files

Your Space needs two key files:

1.  **`app.py`**: This contains your Gradio application code. It should include all your functions and the final `gr.Interface` definition.
    ```python
    # app.py
    import torch
    import gradio as gr
    from transformers import pipeline
    from huggingface_hub import InferenceClient

    # ... Include all your function definitions here (transcribe, organize_text, meeting_transcript_tool) ...

    # Define and launch the interface
    demo = gr.Interface(
        fn=meeting_transcript_tool,
        inputs=gr.Audio(type="filepath"),
        outputs=gr.Textbox(show_copy_button=True),
        title="Meeting Transcription Tool"
    )

    demo.launch()
    ```

2.  **`requirements.txt`**: This file lists the Python dependencies your Space needs.
    ```
    # requirements.txt
    torch
    transformers
    gradio==4.36.1
    huggingface_hub
    ```

### 3.3 Upload and Deploy

Upload these files to your Space's repository. The Space will automatically build and deploy your app. You can share the public URL with anyone.

> **Note:** Free Spaces go to sleep after periods of inactivity but wake up on the next visit.

## Step 4: Explore Built-in Gradio Features

Gradio includes powerful features "out of the box." Here are three useful ones:

### 4.1 Access Your Demo as an API

Every Gradio app automatically provides an API. When your app is running, you'll see an **"Use via API"** section with a link (e.g., `http://127.0.0.1:7860/`). You can interact with your function programmatically.

**Example using `curl`:**
```bash
curl -X POST "http://127.0.0.1:7860/api/predict/" \
     -H "Content-Type: application/json" \
     -d '{"data": ["your_audio_file_path.mp3"]}'
```

**Example using Python's `requests`:**
```python
import requests

response = requests.post("http://127.0.0.1:7860/api/predict/",
                         json={"data": ["your_audio_file_path.mp3"]})
print(response.json())
```

### 4.2 Share via Public URL

When you launch a demo locally, Gradio can create a temporary public URL (valid for 72 hours). Use the `share=True` parameter.

```python
demo.launch(share=True)
```

This is perfect for quick sharing during development before deploying to Spaces.

### 4.3 Collect Feedback with Flagging

The Flagging feature allows users to submit feedback on model outputs, which is invaluable for improving your application.

Enable it by adding `allow_flagging="manual"` to your interface.

```python
demo = gr.Interface(
    fn=meeting_transcript_tool,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.Textbox(show_copy_button=True),
    title="Meeting Transcription Tool",
    allow_flagging="manual"  # Adds a "Flag" button below the output
)
```

When a user clicks **Flag**, the input and output are saved to a CSV file (by default, `flagged/log.csv` in your working directory).

## Conclusion

You've successfully learned how to:
1. Create interactive web demos for any Python function using Gradio.
2. Build a practical two-step meeting transcription tool.
3. Deploy your application permanently on Hugging Face Spaces.
4. Utilize advanced Gradio features like APIs and feedback collection.

Gradio lowers the barrier to showcasing machine learning work. Combine it with Spaces to easily share your creations with the world.

## Further Reading

- [Gradio Documentation](https://www.gradio.app/docs/)
- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Serverless Inference API Cookbook](https://huggingface.co/learn/cookbook/en/enterprise_hub_serverless_inference_api)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)