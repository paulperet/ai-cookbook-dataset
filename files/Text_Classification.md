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

# Gemini API: Text Classification

---


You will use the Gemini API to classify what topics are relevant in the text.

## Install dependencies


```
%pip install -U -q "google-genai>=1.0.0"
```

## Set up your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.


```
from google.colab import userdata
from google import genai

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Example


```
from IPython.display import Markdown

MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Generate a 5 paragraph article about Sports, include one other topic",
)
article = response.text
Markdown(article)
```




## The Enduring Appeal of Sports: More Than Just Games

Sports, in their myriad forms, represent a fundamental aspect of the human experience. From the individual pursuit of a personal best to the collective fervor of a national championship, they offer a unique blend of physical exertion, strategic thinking, and emotional engagement. The appeal transcends mere entertainment; sports are a microcosm of life itself, teaching invaluable lessons about teamwork, discipline, perseverance, and the ability to cope with both victory and defeat. They connect us, forging bonds across geographical boundaries and cultural divides, uniting individuals under the banner of shared passion and admiration. This shared experience creates a sense of community and belonging, particularly in an increasingly fragmented world.

The benefits of sports extend far beyond the playing field. Regular participation fosters physical and mental well-being, reducing the risk of chronic diseases and improving cognitive function. Active involvement, whether as a player, coach, or even a dedicated fan, provides a healthy outlet for stress and promotes a positive self-image. Furthermore, sports often act as a catalyst for economic growth, generating revenue through ticket sales, merchandise, advertising, and tourism. Entire industries thrive around the sporting world, providing employment opportunities and contributing significantly to local and national economies.

Beyond the tangible benefits, sports also serve as a powerful platform for social change. Athletes have increasingly used their platform to advocate for issues they believe in, raising awareness about important causes like racial injustice, gender equality, and climate change. Think of Colin Kaepernick's kneeling protest or Megan Rapinoe's advocacy for equal pay; these moments demonstrate the potential of sports figures to spark meaningful conversations and challenge societal norms. By leveraging their influence, athletes can inspire action and contribute to a more just and equitable world.

Interestingly, this social activism mirrors trends in the world of **art**. Just as athletes are increasingly using their platforms to speak out on social issues, artists are using their work to address societal challenges, provoke thought, and inspire change. From protest songs to politically charged paintings, art and sport both serve as powerful mediums for reflecting and shaping the world around us. They both provide a space for expression, fostering dialogue and encouraging individuals to engage with complex issues in a meaningful way.

In conclusion, the enduring appeal of sports lies in its multifaceted nature. It’s a source of entertainment, a pathway to physical and mental well-being, an engine for economic growth, and a platform for social change. Like art, it reflects the human condition, allowing us to express our emotions, celebrate our triumphs, and confront our challenges. Whether we are athletes, fans, or simply observers, sports hold a significant place in our lives, reminding us of the power of human potential and the importance of collective effort. They are more than just games; they are a reflection of who we are, and who we aspire to be.





```
import enum
from typing_extensions import TypedDict  # in python 3.12 replace typing_extensions with typing

from google.genai import types


class Relevance(enum.Enum):
  WEAK = "weak"
  STRONG = "strong"

class Topic(TypedDict):
  topic: str
  relevance: Relevance


sys_int = """
Generate topics from text. Ensure that topics are general e.g. "Health".
Strong relevance is obtained when the topic is a core tenent of the content
and weak relevance reflects one or two mentions.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=article,
    config=types.GenerateContentConfig(
        system_instruction=sys_int,
        response_mime_type="application/json",
        response_schema=list[Topic],
    )
)
```


```
from pprint import pprint

pprint(response.parsed)
```

    [{'relevance': <Relevance.STRONG: 'strong'>, 'topic': 'Sports'},
     {'relevance': <Relevance.STRONG: 'strong'>, 'topic': 'Health'},
     {'relevance': <Relevance.WEAK: 'weak'>, 'topic': 'Economics'},
     {'relevance': <Relevance.STRONG: 'strong'>, 'topic': 'Social Issues'},
     {'relevance': <Relevance.WEAK: 'weak'>, 'topic': 'Art'}]


## Summary
Now, you know how to classify text into different categories. Feel free to experiment with other texts, or provide a specific set of possible topics.

Please see the other notebooks in this directory to learn more about how you can use the Gemini API for other JSON related tasks.