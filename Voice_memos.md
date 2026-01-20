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

# Voice memos

This notebook provides a quick example of how to work with audio and text files in the same prompt. You'll use the Gemini API to help you generate ideas for your next blog post, based on voice memos you recorded on your phone, and previous articles you've written.


```
%pip install -U -q "google-genai>=1.0.0"
```

    [First Entry, ..., Last Entry]

### Setup your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
from google import genai

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

Install PDF processing tools.


```
!apt install -q poppler-utils
```

    [First Entry, ..., Last Entry]

## Upload your audio and text files



```
!wget -q -O Walking_thoughts_3.m4a https://storage.googleapis.com/generativeai-downloads/data/Walking_thoughts_3.m4a
!wget -q -O A_Possible_Future_for_Online_Content.pdf https://storage.googleapis.com/generativeai-downloads/data/A_Possible_Future_for_Online_Content.pdf
!wget -q -O Unanswered_Questions_and_Endless_Possibilities.pdf https://storage.googleapis.com/generativeai-downloads/data/Unanswered_Questions_and_Endless_Possibilities.pdf
```


```
audio_file_name = "Walking_thoughts_3.m4a"
audio_file = client.files.upload(file=audio_file_name)
```

## Extract text from the PDFs


```
!pdftotext A_Possible_Future_for_Online_Content.pdf
!pdftotext Unanswered_Questions_and_Endless_Possibilities.pdf
```


```
blog_file_name = "A_Possible_Future_for_Online_Content.txt"
blog_file = client.files.upload(file=blog_file_name)
```


```
blog_file_name2 = "Unanswered_Questions_and_Endless_Possibilities.txt"
blog_file2 = client.files.upload(file=blog_file_name2)
```

## System instructions

Write a detailed system instruction to configure the model.


```
si="""Objective: Transform raw thoughts and ideas into polished, engaging blog posts that capture a writers unique style and voice.
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

## Generate Content


```
from google.genai import types

prompt = "Draft my next blog post based on my thoughts in this audio file and these two previous blog posts I wrote."

MODEL_ID ="gemini-2.5-flash" # @param ["gemini-2.5-flash", "gemini-2.5-pro"] {"allow-input":true, isTemplate: true}

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

    Here's a draft for your next blog post, incorporating your audio reflections and linking to your previous pieces:
    
    ---
    
    ## The Art of Thinking: Why Every "Discarded" Idea Fuels Growth
    
    In the fast-paced world of tech and content, where new ideas are constantly emerging and old ones quickly fall by the wayside, it’s easy to feel like some of your work is simply "throwaway." I’ve certainly experienced this myself. Early in my career, I spent countless hours crafting visions, roadmaps, and detailed ideas. Often, these ambitious projects ended up being shelved or canceled – sometimes even after significant effort had already been poured into them. I remember literally throwing away large project plans, feeling a deep sense of frustration.
    
    It still happens today, not just to me, but to my team and even entire companies. Priorities change, markets shift, and what seemed like a brilliant idea yesterday might be irrelevant tomorrow. Coming straight out of an academic environment, where every assignment had a clear objective and a defined outcome, this fluidity felt jarring. In school, you're given a task, you produce it, and you're graded on it. There’s no "take-backsies" on an assignment; it’s a linear path from problem to solution. The real world, however, is far from linear.
    
    For a long time, this disconnect felt like a monumental waste of time and effort. How could I reconcile the desire to work hard and produce, with the reality that so much of it might never see the light of day or be used in its original form?
    
    It took me a while to truly appreciate a different perspective, one that became deeply ingrained when I joined my current team. Here, we embrace what we call the "culture of right to think." This isn't just about acknowledging that priorities change and things shift. It’s about recognizing that the act of producing – of creating content, developing ideas, even if they never become a final product – is an essential part of your growth.
    
    The "right to think" transforms those seemingly discarded efforts into invaluable learning experiences. It’s not about the immediate output being perfect or permanent, but about the process itself. Each idea you explore, each piece of content you draft, each plan you outline, hones your skills. It sharpens your ability to think, to iterate, and to adapt.
    
    This concept ties directly into the themes I've explored in my previous posts, "A Possible Future for Online Content" and "Unanswered Questions and Endless Possibilities." When I talked about reimagining content creation with generative AI, or the future of content production blending unstructured thoughts into immersive experiences, the underlying principle is the same: **the act of creation, in any form, is a continuous learning process.**
    
    Imagine a journalist using AI to quickly comb through sources, extracting facts and perspectives. Even if a specific angle or piece of content doesn't make it into the final story, the journalist's ability to identify relevant information and understand context has been reinforced and refined. Or consider the "army of agents" planning a child's birthday party; the process of instructing and refining those agents, even if a few attempts are "scrapped," builds a deeper understanding of efficient task execution.
    
    The "right to think" encourages us to:
    
    *   **Write more, create more:** Don't wait for the perfect idea or the final product. Get your thoughts down, explore possibilities.
    *   **Write earlier, write often:** The earlier and more frequently you engage in the creative process, the more opportunities you have to learn.
    *   **Write not with the intention of it needing to be a final product:** Be willing to scrap it, iterate, and move on once it has served its purpose in your learning journey.
    
    This reframing of "throwaway work" is incredibly liberating. In this model, there is no such thing as wasted effort. Every seemingly failed idea, every discarded draft, every project that gets re-prioritized, contributes to your growth and understanding. It’s all helping you to hone your skills and get better over time.
    
    The "culture of right to think" aligns beautifully with iterative processes and learning by doing. It provides a powerful framework for navigating the dynamic nature of work in an AI-powered world. So, the next time you feel frustrated by a project that doesn't go "anywhere," remember: it's not about the destination, but the invaluable journey of thinking, creating, and evolving.
    
    ---


## Learning more

* Learn more about the [File API](https://github.com/google-gemini/cookbook/blob/main/quickstarts/File_API.ipynb) with the quickstart.