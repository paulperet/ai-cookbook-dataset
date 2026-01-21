# Voice Memos to Blog Posts: A Gemini AI Workflow

This guide demonstrates how to use the Gemini API to synthesize ideas from voice memos and previous writings into a new blog post draft. You'll upload audio recordings and past articles, then prompt the model to generate content in your unique style.

## Prerequisites & Setup

First, install the required Python client and system tools.

```bash
# Install the Google Generative AI Python SDK
%pip install -U -q "google-genai>=1.0.0"
```

```bash
# Install PDF processing utilities (required for text extraction)
!apt install -q poppler-utils
```

Now, configure your API key and initialize the client. Ensure your key is stored in a Colab Secret named `GOOGLE_API_KEY`.

```python
from google.colab import userdata
from google import genai

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 1: Upload Your Source Files

You'll need an audio file (a voice memo) and at least one previous blog post (in PDF format). For this example, we'll download sample files.

```bash
# Download example files
!wget -q -O Walking_thoughts_3.m4a https://storage.googleapis.com/generativeai-downloads/data/Walking_thoughts_3.m4a
!wget -q -O A_Possible_Future_for_Online_Content.pdf https://storage.googleapis.com/generativeai-downloads/data/A_Possible_Future_for_Online_Content.pdf
!wget -q -O Unanswered_Questions_and_Endless_Possibilities.pdf https://storage.googleapis.com/generativeai-downloads/data/Unanswered_Questions_and_Endless_Possibilities.pdf
```

Upload the audio file to the Gemini API. This prepares it for use in prompts.

```python
audio_file_name = "Walking_thoughts_3.m4a"
audio_file = client.files.upload(file=audio_file_name)
```

## Step 2: Extract Text from PDFs

Convert your PDF blog posts into plain text files, then upload them.

```bash
# Extract text from the PDFs
!pdftotext A_Possible_Future_for_Online_Content.pdf
!pdftotext Unanswered_Questions_and_Endless_Possibilities.pdf
```

```python
# Upload the first blog post text file
blog_file_name = "A_Possible_Future_for_Online_Content.txt"
blog_file = client.files.upload(file=blog_file_name)

# Upload the second blog post text file
blog_file_name2 = "Unanswered_Questions_and_Endless_Possibilities.txt"
blog_file2 = client.files.upload(file=blog_file_name2)
```

## Step 3: Define the System Instructions

Create a detailed system prompt to guide the model. This instructs it to analyze your writing style from the provided examples and synthesize a new draft based on your audio thoughts.

```python
system_instruction = """
Objective: Transform raw thoughts and ideas into polished, engaging blog posts that capture a writers unique style and voice.

Input:
Example Blog Posts (1-5): A user will provide examples of blog posts that resonate with their desired style and tone. These will guide you in understanding the preferences for word choice, sentence structure, and overall voice.
Audio Clips: A user will share a selection of brainstorming thoughts and key points through audio recordings. They will talk freely and openly, as if they were explaining their ideas to a friend.

Output:
Blog Post Draft: A well-structured first draft of the blog post, suitable for platforms like Substack or LinkedIn.
The draft will include:
Clear and engaging writing: you will strive to make the writing clear, concise, and interesting for the target audience.
Tone and style alignment: The language and style will closely match the examples provided, ensuring consistency with the desired voice.
Logical flow and structure: The draft will be organized with clear sections based on the content of the post.
Target word count: Aim for 500-800 words, but this can be adjusted based on user preferences.

Process:
Style Analysis: Carefully analyze the example blog posts provided by the user to identify key elements of their preferred style, including:
Vocabulary and word choice: Formal vs. informal, technical terms, slang, etc.
Sentence structure and length: Short and impactful vs. longer and descriptive sentences.
Tone and voice: Humorous, serious, informative, persuasive, etc.
Audio Transcription and Comprehension: Your audio clips will be transcribed with high accuracy. you will analyze them to extract key ideas, arguments, and supporting points.
Draft Generation: Using the insights from the audio and the style guidelines from the examples, you will generate a first draft of the blog post. This draft will include all relevant sections with supporting arguments or evidence, and a great ending that ties everything together and makes the reader want to invest in future readings.
"""
```

## Step 4: Generate the Blog Post Draft

Now, call the Gemini model with your prompt and uploaded files. We'll use the `gemini-2.5-flash` model for this task.

```python
from google.genai import types

prompt = "Draft my next blog post based on my thoughts in this audio file and these two previous blog posts I wrote."

MODEL_ID = "gemini-2.5-flash"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, blog_file, blog_file2, audio_file],
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=0
        )
    )
)

print(response.text)
```

**Example Output:**
The model will generate a complete blog post draft. Here is a condensed version of what you might receive:

> **The Art of Thinking: Why Every "Discarded" Idea Fuels Growth**
>
> In the fast-paced world of tech and content... I remember literally throwing away large project plans, feeling a deep sense of frustration...
>
> It took me a while to truly appreciate a different perspective... we embrace what we call the "culture of right to think."...
>
> This concept ties directly into the themes I've explored in my previous posts... **the act of creation, in any form, is a continuous learning process.**
>
> The "right to think" encourages us to: *Write more, create more*... *Write earlier, write often*... *Write not with the intention of it needing to be a final product*...
>
> This reframing of "throwaway work" is incredibly liberating... So, the next time you feel frustrated by a project that doesn't go "anywhere," remember: it's not about the destination, but the invaluable journey of thinking, creating, and evolving.

## Next Steps

You now have a first draft synthesized from your audio ideas and past writings. You can refine this prompt, adjust the system instructions, or experiment with different models like `gemini-2.5-pro` for varied results.

To learn more about handling files with the API, see the [File API quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/File_API.ipynb).