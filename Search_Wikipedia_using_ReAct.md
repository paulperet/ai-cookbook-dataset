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

# Search Wikipedia using ReAct

This notebook is a minimal implementation of [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) with the Google `gemini-2.0-flash` model. You'll use ReAct prompting to configure a model to search Wikipedia to find the answer to a user's question.

In this walkthrough, you will learn how to:

1.   Set up your development environment and API access to use Gemini.
2.   Use a ReAct few-shot prompt.
3.   Use the newly prompted model for multi-turn conversations (chat).
4.   Connect the model to the **Wikipedia API**.
5.  Have conversations with the model (try asking it questions like "how tall is the Eiffel Tower?") and watch it search Wikipedia.

> Note: The non-source code materials on this page are licensed under Creative Commons - Attribution-ShareAlike CC-BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/legalcode.

### Background

[ReAct](https://arxiv.org/abs/2210.03629) is a prompting method which allows language models to create a trace of their thinking processes and the steps required to answer a user's questions. This improves human interpretability and trustworthiness. ReAct prompted models generate Thought-Action-Observation triplets for every iteration, as you'll soon see. Let's get started!

## Setup

```
%pip install -q "google-generativeai>=0.7.2"
```

```
%pip install -q wikipedia
```

Note: The [`wikipedia` package](https://pypi.org/project/wikipedia/) notes that it was "designed for ease of use and simplicity, not for advanced use", and that production or heavy use should instead "use [Pywikipediabot](http://www.mediawiki.org/wiki/Manual:Pywikipediabot) or one of the other more advanced [Python MediaWiki API wrappers](http://en.wikipedia.org/wiki/Wikipedia:Creating_a_bot#Python)".

```
import re
import os

import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

import google.generativeai as genai
```

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.

```
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
```

## The ReAct prompt

The prompts used in the paper are available at [https://github.com/ysymyth/ReAct/tree/master/prompts](https://github.com/ysymyth/ReAct/tree/master/prompts)

Here, you will be working with the following ReAct prompt with a few minor adjustments.

> Note: The prompt and in-context examples used here are borrowed from [https://github.com/ysymyth/ReAct](https://github.com/ysymyth/ReAct) which is published under a [MIT license](https://opensource.org/licenses/MIT).

```
model_instructions = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, Observation is understanding relevant information from an Action's output and Action can be of three types:
(1) <search>entity</search>, which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search and you can try to search the information from those topics.
(2) <lookup>keyword</lookup>, which returns the next sentence containing keyword in the current context. This only does exact matches, so keep your searches short.
(3) <finish>answer</finish>, which returns the answer and finishes the task.
"""
```

### Few-shot prompting to enable in-context learning with Gemini

While large language models show good understanding of the instructions they are prompted with, they still may perform poorly on complex tasks in a zero-shot setting. Hence, you will now provide a few examples along with your prompt to steer the model's output according to your needs. This in-context learning improves the model's performance significantly.

```
examples = """
Here are some examples.

Question
What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?

Thought 1
I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.

Action 1
<search>Colorado orogeny</search>

Observation 1
The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.

Thought 2
It does not mention the eastern sector. So I need to look up eastern sector.

Action 2
<lookup>eastern sector</lookup>

Observation 2
The eastern sector extends into the High Plains and is called the Central Plains orogeny.

Thought 3
The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.

Action 3
<search>High Plains</search>

Observation 3
High Plains refers to one of two distinct land regions

Thought 4
I need to instead search High Plains (United States).

Action 4
<search>High Plains (United States)</search>

Observation 4
The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130m).

Thought 5
High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.

Action 5
<finish>1,800 to 7,000 ft</finish>

Question
Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?

Thought 1
The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.

Action 1
<search>Milhouse</search>

Observation 1
Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.

Thought 2
The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".

Action 2
<lookup>named after</lookup>

Observation 2
Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.

Thought 3
Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.

Action 3
<finish>Richard Nixon</finish>

Question
Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?

Thought 1
I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.

Action 1
<search>Adam Clayton Powell</search>

Observation 1
Could not find [Adam Clayton Powell]. Similar: [’Adam Clayton Powell III’, ’Seventh Avenue (Manhattan)’, ’Adam Clayton Powell Jr. State Office Building’, ’Isabel Washington Powell’, ’Adam Powell’, ’Adam Clayton Powell (film)’, ’Giancarlo Esposito’].

Thought 2
To find the documentary, I can search Adam Clayton Powell (film).

Action 2
<search>Adam Clayton Powell (film)</search>

Observation 2
Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg. The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.

Thought 3
Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.

Action 3
<finish>The Saimaa Gesture</finish>

Question
What profession does Nicholas Ray and Elia Kazan have in common?

Thought 1
I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.

Action 1
<search>Nicholas Ray</search>

Observation 1
Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 - June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.

Thought 2
Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.

Action 2
<search>Elia Kazan</search>

Observation 2
Elia Kazan was an American film and theatre director, producer, screenwriter and actor.

Thought 3
Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.

Action 3
<finish>director, screenwriter, actor</finish>

Question
Which magazine was started first Arthur’s Magazine or First for Women?

Thought 1
I need to search Arthur’s Magazine and First for Women, and find which was started first.

Action 1
<search>Arthur’s Magazine</search>

Observation 1
Arthur’s Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.

Thought 2
Arthur’s Magazine was started in 1844. I need to search First for Women next.

Action 2
<search>First for Women</search>

Observation 2
First for Women is a woman’s magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.

Thought 3
First for Women was started in 1989. 1844 (Arthur’s Magazine) < 1989 (First for Women), so Arthur’s Magazine was started first.

Action 3
<finish>Arthur’s Magazine</finish>

Question
Were Pavel Urysohn and Leonid Levin known for the same type of work?

Thought 1
I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.

Action 1
<search>Pavel Urysohn</search>

Observation 1
Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.

Thought 2
Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.

Action 2
<search>Leonid Levin</search>

Observation 2
Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.

Thought 3
Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.

Action 3
<finish>yes</finish>

Question
{question}"""
```

Copy the instructions along with examples in a file called `model_instructions.txt`

```
ReAct_prompt = model_instructions + examples
with open('model_instructions.txt', 'w') as f:
  f.write(ReAct_prompt)
```

## The Gemini-ReAct pipeline

### Setup

You will now build an end-to-end pipeline to facilitate multi-turn chat with the ReAct-prompted Gemini model.

```
class ReAct:
  def __init__(self, model: str, ReAct_prompt: str | os.PathLike):
    """Prepares Gemini to follow a `Few-shot ReAct prompt` by imitating
    `function calling` technique to generate both reasoning traces and
    task-specific actions in an interleaved manner.

    Args:
        model: name to the model.
        ReAct_prompt: ReAct prompt OR path to the ReAct prompt.
    """
    self.model = genai.GenerativeModel(model)
    self.chat = self.model.start_chat(history=[])
    self.should_continue_prompting = True
    self._search_history: list[str] = []
    self._search_urls: list[str] = []

    try:
      # try to read the file
      with open(ReAct_prompt, 'r') as f:
        self._prompt = f.read()
    except FileNotFoundError:
      # assume that the parameter represents prompt itself rather than path to the prompt file.
      self._prompt = ReAct_prompt

  @property
  def prompt(self):
    return self._prompt

  @classmethod
  def add_method(cls, func):
    setattr(cls, func.__name__, func)

  @staticmethod
  def clean(text: str):
    """Helper function for responses."""
    text = text.replace("\n", " ")
    return text
```

### Define tools

As instructed by the prompt, the model will be generating **Thought-Action-Observation** traces, where every **Action** trace could be one of the following tokens:

1.   </search/> : Perform a Wikipedia search via external API.
2.   </lookup/> : Lookup for specific information on a page with the Wikipedia API.
3.   </finish/> : Stop the execution of the model and return the answer.

If the model encounters any of these tokens, the model should make use of the `tools` made available to the model. This understanding of the model to leverage acquired toolsets to collect information from the external world is often referred to as **function calling**. Therefore, the next goal is to imitate this function calling technique in order to allow ReAct prompted Gemini model to access the external groundtruth.

The Gemini API supports function calling and you could use this feature to set up your tools. However, for this tutorial, you will learn to simulate it using `stop_sequences` parameter.

Define the tools:

#### Search
Define a method to perform Wikipedia searches

```
@ReAct.add_method
def search(self, query: str):
    """Perfoms search on `query` via Wikipedia api and returns its summary.

    Args:
        query: Search parameter to query the Wikipedia API with.

    Returns:
        observation: Summary of Wikipedia search for `query` if found else
        similar search results.
    """
    observation = None
    query = query.strip()
    try:
      # try to get the summary for requested `query` from the Wikipedia
      observation = wikipedia.summary(query, sentences=4, auto_suggest=False)
      wiki_url = wikipedia.page(query, auto_suggest=False).url
      observation = self.clean(observation)

      # if successful, return the first 2-3 sentences from the summary as model's context
      observation = self.model.generate_content(f'Retun the first 2 or 3 \
      sentences from the following text: {observation}')
      observation = observation.text

      # keep track of the model's search history
      self._search_history.append(query)
      self._search_urls.append(wiki_url)
      print(f"Information Source: {wiki_url}")

    # if the page is ambiguous/does not exist, return similar search phrases for model's context
    except (DisambiguationError, PageError) as e:
      observation = f'Could not find ["{query}"].'
      # get a list of similar search topics
      search_results = wikipedia.search(query)
      observation += f' Similar: {search_results}. You should search for one of those instead.'

    return observation
```

#### Lookup
Look for a specific phrase on the Wikipedia page.

```
@ReAct.add_method
def lookup(self, phrase: str, context_length=200):
    """Searches for the `phrase` in the lastest Wikipedia search page
    and returns number of sentences which is controlled by the
    `context_length` parameter.

    Args:
        phrase: Lookup phrase to search for within a page. Generally
        attributes to some specification of any topic.

        context_length: Number of words to consider
        while looking for the answer.

    Returns:
        result: Context related to the `phrase` within the page.
    """
    # get the last searched Wikipedia page and find `phrase` in it.
    page = wikipedia.page(self._search_history[-1], auto_suggest=False)
    page = page.content
    page = self.clean(page)
    start_index = page.find(phrase)

    # extract sentences considering the context length defined
    result = page[max(0, start_index - context_length):start_index+len(phrase)+context_length]
    print(f"Information Source: {self._search_urls[-1]}")
    return result
```

#### Finish
Instruct the pipline to terminate its execution.

```
@ReAct.add_method
def finish(self, _):
  """Finishes the conversation on encountering <finish> token by
  setting the `self.should_continue_prompting` flag to `False`.
  """
  self.should_continue_prompting = False
  print(f"Information Sources: {self._search_urls}")
```

### Stop tokens and function calling imitation

Now that you are all set with function definitions, the next step is to instruct the model to interrupt its execution upon encountering any of the action tokens. You will make use of the `stop_sequences` parameter from [`genai.GenerativeModel.GenerationConfig`](https://ai.google.dev/api/python/google/generativeai/GenerationConfig) class to instruct the model when to stop. Upon encountering an action token, the pipeline will simply extract what specific token from the `stop_sequences` argument terminated the model's execution, and then call the appropriate **tool** (function).

The function's response will be added to model's chat history for continuing the context link.

```
@ReAct.add_method
def __call__(self, user_question, max_calls: int=8, **generation_kwargs):
  """Starts multi-turn conversation