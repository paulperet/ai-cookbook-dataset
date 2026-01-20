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

# Gemini API: Audio Quickstart

This notebook provides an example of how to prompt Gemini Flash using an audio file. In this case, you'll use a [sound recording](https://www.jfklibrary.org/asset-viewer/archives/jfkwha-006) of President John F. Kennedy’s 1961 State of the Union address.

## Setup

### Install dependencies


```
%pip install -q -U "google-genai>=1.0.0"
```

[First Entry, ..., Last Entry]

### Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
from google import genai

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

client = genai.Client(api_key=GOOGLE_API_KEY)
```

Now select the model you want to use in this guide, either by selecting one in the list or writing it down. Keep in mind that some models, like the 2.5 ones are thinking models and thus take slightly more time to respond (cf. [thinking notebook](./Get_started_thinking.ipynb) for more details and in particular learn how to switch the thiking off).


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

### Upload an audio file with the File API

To use an audio file in your prompt, you must first upload it using the [File API](../quickstarts/File_API.ipynb).



```
URL = "https://storage.googleapis.com/generativeai-downloads/data/State_of_the_Union_Address_30_January_1961.mp3"
```


```
!wget -q $URL -O sample.mp3
```


```
your_audio_file = client.files.upload(file='sample.mp3')
```

## Use the file in your prompt


```
response = client.models.generate_content(
  model=MODEL_ID,
  contents=[
    'Listen carefully to the following audio file. Provide a brief summary.',
    your_audio_file,
  ]
)

print(response.text)
```

    In his first State of the Union address on January 30, 1961, President John F. Kennedy acknowledged the nation's severe economic challenges, including a recession, high unemployment, and stagnant growth. He outlined a comprehensive domestic agenda to combat these issues, proposing measures to boost employment, stimulate the economy, and address critical social needs in housing, education, healthcare, and civil rights.
    
    Globally, he addressed the balance of payments deficit, affirming the dollar's stability while advocating for stronger military capabilities, a revamped foreign aid program (including the Alliance for Progress for Latin America), and the creation of a Peace Corps to leverage American talent abroad. He also discussed Cold War relations with the Soviet Union and China, emphasizing peaceful competition in areas like space exploration and science, and called for strengthening the United Nations.
    
    Kennedy stressed the need for decisive governmental action and a reinvigorated public service based on merit. He concluded by acknowledging the immense challenges ahead, urging national unity and perseverance to navigate a period of expected difficulty, echoing Franklin D. Roosevelt's call for the nation to be worthy of its opportunities.


## Inline Audio

For small requests you can inline the audio data into the request, like you can with images. Use PyDub to trim the first 10s of the audio:


```
%pip install -Uq pydub
```


```
from pydub import AudioSegment
```


```
sound = AudioSegment.from_mp3("sample.mp3")
```


```
sound[:10000] # slices are in ms
```


Add it to the list of parts in the prompt:


```
from google.genai import types

response = client.models.generate_content(
  model=MODEL_ID,
  contents=[
    'Describe this audio clip',
    types.Part.from_bytes(
      data=sound[:10000].export().read(),
      mime_type='audio/mp3',
    )
  ]
)

print(response.text)
```

    This audio clip features a **male voice** speaking formally, delivering what sounds like an introduction to a political address.
    
    Before the speech begins, there's a constant, low-frequency **hum** or ambient background noise, and then a distinct, sharp **"thud" or "tap" sound**, possibly from a microphone being adjusted or a podium being hit.
    
    The male voice is **clear, well-articulated, and possesses an authoritative, broadcast-like tone**. The speaker states: "The President's State of the Union Address to a Joint Session of the Congress from the rostrum of the House of Representatives."
    
    The overall impression is that of a **formal public event or a news broadcast** related to government proceedings.


Note the following about providing audio as inline data:

- The maximum request size is 100 MB, which includes text prompts, system instructions, and files provided inline. If your file's size will make the total request size exceed 100 MB, then [use the File API](https://ai.google.dev/gemini-api/docs/audio?lang=python#upload-audio) to upload files.
- If you're using an audio sample multiple times, it is more efficient to [use the File API](https://ai.google.dev/gemini-api/docs/audio?lang=python#upload-audio).


## Get a transcript of the audio file
To get a transcript, just ask for it in the prompt. For example:



```
prompt = "Generate a transcript of the speech."
```

### Refer to timestamps in the audio file
A prompt can specify timestamps of the form `MM:SS` to refer to particular sections in an audio file. For example:


```
# Create a prompt containing timestamps.
prompt = "Provide a transcript of the speech between the timestamps 02:30 and 03:29."

response = client.models.generate_content(
  model=MODEL_ID,
  contents=[
    prompt,
    your_audio_file,
  ]
)

print(response.text)
```

    I speak today in an hour of national peril and national opportunity. Before my term has ended, we shall have to test anew whether a nation organized and governed such as ours can endure. The outcome is by no means certain. The answers are by no means clear. All of us together, this administration, this Congress, this nation, must forge those answers. But today, were I to offer, after little more than a week in office, detailed legislation to remedy every national ill, the Congress would rightly wonder whether the desire for speed had replaced the duty of responsibility. My remarks, therefore, will be limited, but they will also be candid. To state the facts frankly is not to despair the future, nor indict the past.


## Use a Youtube video


```
from google.genai import types
from IPython.display import display, Markdown

youtube_url = "https://www.youtube.com/watch?v=RDOMKIw1aF4" # Repalce with the youtube url you want to analyze

prompt = """
    Analyze the following YouTube video content. Provide a concise summary covering:

    1.  **Main Thesis/Claim:** What is the central point the creator is making?
    2.  **Key Topics:** List the main subjects discussed, referencing specific examples or technologies mentioned (e.g., AI models, programming languages, projects).
    3.  **Call to Action:** Identify any explicit requests made to the viewer.
    4.  **Summary:** Provide a concise summary of the video content.

    Use the provided title, chapter timestamps/descriptions, and description text for your analysis.
"""
# Analyze the video
response = client.models.generate_content(
    model=MODEL_ID,
    contents=types.Content(
        parts=[
            types.Part(text=prompt),
            types.Part(
                file_data=types.FileData(file_uri=youtube_url)
            )
        ]
    )
)
display(Markdown(response.text))
```

This YouTube video analyzes the capabilities of Google's recently released Gemini 2.5 Pro (Experimental) AI model, focusing on its performance in various coding tasks compared to other leading AI models.

1.  **Main Thesis/Claim:**
    The central claim is that Google's Gemini 2.5 Pro is the "best coding AI" the creator has ever used, showcasing its advanced capabilities in code generation and refactoring, often outperforming its competitors and prior versions.

2.  **Key Topics:**
    *   **AI Model Comparison & Benchmarks:** The video directly compares Gemini 2.5 Pro against models like OpenAI's o3-mini and GPT-4.5, Claude 3.7 Sonnet, DeepSeek R1, and Grok 3 Beta across different intellectual domains (Reasoning & Knowledge, Science, Mathematics) and specific coding metrics (Code Generation, Code Editing, Agentic Coding). Gemini generally shows strong, often leading, performance, especially in single-attempt tasks and complex problem-solving.
    *   **Game Development:**
        *   **Ultimate Tic-Tac-Toe (Java Swing):** Gemini successfully generated a fully functional "Ultimate Tic-Tac-Toe" game in Java using Swing based on a complex, single prompt, which the creator highlights as a "one-shot" success.
        *   **P5.js Kitten Cannon Clone:** Gemini generated a P5.js game, but it required two rounds of error correction (TypeErrors related to `oncontextmenu` and `sketch.sign`) before becoming fully playable (a "three-shot" process).
    *   **Web Development (Front-end):**
        *   **React Landing Page (Vite, React, Tailwind CSS):** Gemini struggled significantly with creating a landing page from an image mockup, producing incomplete code that poorly replicated the design, which the creator deemed "really bad."
        *   **X (Twitter) UI Recreation (HTML):** Gemini successfully recreated a static HTML/CSS visual representation of the X (Twitter) website's UI, demonstrating its ability to interpret visual layouts for basic rendering.
    *   **Code Refactoring (Rust):** The AI demonstrated impressive refactoring capabilities in Rust, converting complex `for` loops into more idiomatic iterator methods and cleaning up conditional logic, which the creator praised as "clean code."
    *   **AI Data Timeliness/Grounding:** A test on querying the current ReactJS version showed that while Gemini's internal knowledge might not be the absolute latest, it can retrieve up-to-date information accurately when "grounded" with Google Search.

3.  **Call to Action:**
    The creator explicitly asks viewers to:
    *   Leave their thoughts and experiences with Gemini 2.5 Pro Experimental in the comments section.
    *   Subscribe to the channel, like the video, and hit the notification bell.

4.  **Summary:**
    The video evaluates Google's new Gemini 2.5 Pro AI as a powerful and highly effective tool for coding. Through practical demonstrations, the creator showcases its ability to generate complex games (like Ultimate Tic-Tac-Toe in Java Swing in a single attempt) and perform sophisticated code refactoring in Rust with impressive results. While it excelled at recreating existing UI elements (like Twitter's X) and correcting its own errors in game generation, its performance on building a web landing page from a visual mockup was notably weak. Benchmarks are presented to compare its performance across various domains against other major AI models, highlighting its overall strength and competitive edge in many areas. Despite minor shortcomings in some tasks and the need for external "grounding" for real-time data, the creator enthusiastically endorses Gemini 2.5 Pro, suggesting it may become his primary coding AI due to its high quality and cost-effectiveness.


## Count audio tokens

You can count the number of tokens in your audio file using the [count_tokens](https://googleapis.github.io/python-genai/#count-tokens-and-compute-tokens) method.

Audio files have a fixed per second token rate (more details in the dedicated [count token quickstart](./Counting_Tokens.ipynb).


```
count_tokens_response = client.models.count_tokens(
    model=MODEL_ID,
    contents=[your_audio_file],
)

print("Audio file tokens:",count_tokens_response.total_tokens)
```

    Audio file tokens: 83528


## Next Steps
### Useful API references:

More details about Gemini API's [vision capabilities](https://ai.google.dev/gemini-api/docs/vision) in the documentation.

If you want to know about the File API, check its [API reference](https://ai.google.dev/api/files) or the [File API](https://github.com/google-gemini/cookbook/blob/main/quickstarts/File_API.ipynb) quickstart.

### Related examples

Check this example using the audio files to give you more ideas on what the gemini API can do with them:
* Share [Voice memos](https://github.com/google-gemini/cookbook/blob/main/examples/Voice_memos.ipynb) with Gemini API and brainstorm ideas

### Continue your discovery of the Gemini API

Have a look at the [Audio](../quickstarts/Audio.ipynb) quickstart to learn about another type of media file, then learn more about [prompting with media files](https://ai.google.dev/tutorials/prompting_with_media) in the docs, including the supported formats and maximum length for audio files. .