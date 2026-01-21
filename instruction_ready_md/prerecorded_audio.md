# Guide: Transcribe Audio with Deepgram & Generate Interview Questions with Anthropic

This guide walks you through creating a workflow that transcribes an audio file using Deepgram's speech-to-text API and then uses Anthropic's Claude to generate a set of thoughtful interview questions based on the transcript.

## Prerequisites

Before you begin, ensure you have:
1.  A **Deepgram API Key**. You can sign up for a free account at [Deepgram](https://dpgr.am/prerecorded-notebook-signup).
2.  An **Anthropic API Key** for access to Claude.
3.  The URL of a publicly accessible audio file you wish to transcribe.

## Step 1: Install Dependencies

First, install the required Python libraries. Run the following commands in your terminal or notebook environment.

```bash
pip install requests ffmpeg-python
pip install deepgram-sdk --upgrade
pip install anthropic
```

## Step 2: Transcribe Audio with Deepgram

In this step, you will write a script to download an audio file and send it to Deepgram for transcription. The transcript will be saved as a JSON file.

1.  Create a new Python script (e.g., `transcribe.py`).
2.  Replace the placeholder API key and audio URL with your own.
3.  Run the script to generate the transcript.

```python
import requests
from deepgram import DeepgramClient, FileSource, PrerecordedOptions

# Configure with your credentials and audio file
DG_KEY = "YOUR_DEEPGRAM_API_KEY_HERE"
AUDIO_FILE_URL = "https://static.deepgram.com/examples/nasa-spacewalk-interview.wav"
TRANSCRIPT_FILE = "transcript.json"

def main():
    try:
        # Initialize the Deepgram client
        deepgram = DeepgramClient(DG_KEY)

        # Download the audio file from the URL
        print("Downloading audio file...")
        response = requests.get(AUDIO_FILE_URL, timeout=60)
        if response.status_code == 200:
            buffer_data = response.content
        else:
            print("Failed to download audio file")
            return

        # Prepare the audio data for Deepgram
        payload: FileSource = {
            "buffer": buffer_data,
        }

        # Configure transcription options
        options = PrerecordedOptions(
            model="nova-2",  # Using Deepgram's Nova-2 model
            smart_format=True,  # Adds punctuation and formatting
        )

        # Send the audio to Deepgram for transcription
        print("Transcribing audio...")
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

        # Save the full JSON response to a file
        with open(TRANSCRIPT_FILE, "w") as transcript_file:
            transcript_file.write(response.to_json(indent=4))

        print(f"Transcript saved successfully to '{TRANSCRIPT_FILE}'.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
```

**Expected Output:**
```
Downloading audio file...
Transcribing audio...
Transcript saved successfully to 'transcript.json'.
```

## Step 3: View the Transcript

After successful transcription, you can extract and print the plain text transcript from the JSON file.

Create a new script or run the following code in your environment.

```python
import json

# Path to the transcript file generated in the previous step
TRANSCRIPT_FILE = "transcript.json"

def print_transcript(transcription_file):
    """Reads the Deepgram JSON and prints the formatted transcript."""
    with open(transcription_file) as file:
        data = json.load(file)
        # Navigate the JSON structure to get the transcript text
        transcript_text = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        
        # Split into sentences for cleaner printing
        sentences = transcript_text.split(".")
        for sentence in sentences:
            if sentence.strip():  # Avoid printing empty strings
                print(sentence.strip() + ".")

print_transcript(TRANSCRIPT_FILE)
```

## Step 4: Generate Interview Questions with Anthropic Claude

Now, use the transcript as context for Claude to generate insightful, open-ended interview questions.

1.  Create a new script (e.g., `generate_questions.py`).
2.  Insert your Anthropic API key.
3.  The script will load the transcript and send it to Claude with a specific system prompt designed to generate high-quality questions.

```python
import json
import anthropic

# Configuration
TRANSCRIPT_FILE = "transcript.json"
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY_HERE"

def get_transcript(transcription_file):
    """Helper function to load the transcript text from the JSON file."""
    with open(transcription_file) as file:
        data = json.load(file)
        transcript_text = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript_text

def main():
    # Load the transcript
    print("Loading transcript...")
    transcript_text = get_transcript(TRANSCRIPT_FILE)

    # Initialize the Anthropic client
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Prepare the message for Claude
    formatted_messages = [{"role": "user", "content": transcript_text}]

    # Define the system prompt to guide Claude's output
    system_prompt = """
    Your task is to generate a series of thoughtful, open-ended questions for an interview based on the given context.
    The questions should be designed to elicit insightful and detailed responses from the interviewee, allowing them to showcase their knowledge, experience, and critical thinking skills.
    Avoid yes/no questions or those with obvious answers. Instead, focus on questions that encourage reflection, self-assessment, and the sharing of specific examples or anecdotes.
    """

    # Call the Claude API
    print("Generating interview questions with Claude...\n")
    response = client.messages.create(
        model="claude-3-opus-20240229", # You can use other Claude 3 models
        max_tokens=1000,
        temperature=0.5, # Balances creativity and focus
        system=system_prompt,
        messages=formatted_messages,
    )

    # Process and print the response
    # Claude's response content is a list of TextBlock objects
    full_content = "".join(block.text for block in response.content)
    
    # Print each question/paragraph with spacing
    questions = full_content.split("\n\n")
    for i, question in enumerate(questions, 1):
        if question.strip():
            print(f"{i}. {question.strip()}\n")

if __name__ == "__main__":
    main()
```

**Expected Output:**
You will see a numbered list of open-ended interview questions derived from the content of your audio transcript. For example:
```
1. Based on your experience described in the transcript, what do you believe are the three most critical skills for succeeding in a high-stakes, collaborative environment like a spacewalk, and why?

2. Reflecting on the communication challenges mentioned, can you share a specific instance where miscommunication occurred during a complex operation and how the team resolved it?
```

## Summary

You have successfully built a pipeline that:
1.  Takes a public audio URL.
2.  Transcribes it accurately using the Deepgram API.
3.  Uses the resulting text as context for Anthropic's Claude model to generate a set of targeted, open-ended interview questions.

This workflow is useful for content creators, journalists, HR professionals, or anyone needing to quickly analyze spoken content and derive meaningful conversation starters.