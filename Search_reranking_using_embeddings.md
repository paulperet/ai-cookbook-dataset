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

# Search re-ranking using Gemini embeddings

This notebook demonstrates the use of embeddings to re-rank search results. This walkthrough will focus on the following objectives:

1.   Setting up your development environment and API access to use Gemini.
2.   Using Gemini's function calling support to access the Wikipedia API.
3.   Embedding content via Gemini API.
4.   Re-ranking the search results.

This is how you will implement search re-ranking:

1.   The user will make a search query.
2.   You will use Wikipedia API to return the relevant search results.
3.   The search results will be embedded and their relevance will be evaluated by calculating distance metrics like cosine similarity.
4.   The most relevant search result will be returned as the final answer.

> The non-source code materials in this notebook are licensed under Creative Commons - Attribution-ShareAlike CC-BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/legalcode.

## Setup

First, download and install the Gemini API Python library.

```
!pip install -U -q google-genai
```

Also install the `wikipedia` package that will be used during this tutorial.

```
!pip install -U -q wikipedia
```

Note: The [`wikipedia` package](https://pypi.org/project/wikipedia/) notes that it was "designed for ease of use and simplicity, not for advanced use", and that production or heavy use should instead "use [Pywikipediabot](http://www.mediawiki.org/wiki/Manual:Pywikipediabot) or one of the other more advanced [Python MediaWiki API wrappers](http://en.wikipedia.org/wiki/Wikipedia:Creating_a_bot#Python)".

```
import json
import textwrap

from google import genai
from google.genai import types

import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError

import numpy as np

from IPython.display import Markdown

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
```

### Grab an API Key

Before you can use the Gemini API, you must first obtain an API key. If you don't already have one, create a key with one click in Google AI Studio.

In Colab, add the key to the secrets manager under the "🔑" in the left panel. Give it the name `GEMINI_API_KEY`.

Once you have the API key, pass it to the SDK. You can do this in two ways:

* Put the key in the `GEMINI_API_KEY` environment variable (the SDK will automatically pick it up from there).
* Pass the key to `genai.Client(api_key=...)`

```
from google.colab import userdata

# Or use `os.getenv('GEMINI_API_KEY')` to fetch an environment variable.
GEMINI_API_KEY=userdata.get('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)
```

### Select the model to be used

```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

## Define tools

As stated earlier, this tutorial uses Gemini's function calling support to access the Wikipedia API. Please refer to the [docs](https://ai.google.dev/docs/function_calling) to learn more about function calling.

### Define the search function

To cater to the search engine needs, you will design this function in the following way:

*   For each search query, the search engine will use the `wikipedia.search` method to get relevant topics.
*   From the relevant topics, the engine will choose `n_topics(int)` top candidates and will use `gemini-2.5-flash` to extract relevant information from the page.
*   The engine will avoid duplicate entries by maintaining a search history.

```
def wikipedia_search(search_queries: list[str]) -> list[str]:
  """Search wikipedia for each query and summarize relevant docs."""
  n_topics=3
  search_history = set() # tracking search history
  search_urls = []
  summary_results = []

  for query in search_queries:
    print(f'Searching for "{query}"')
    search_terms = wikipedia.search(query)

    print(f"Related search terms: {search_terms[:n_topics]}")
    for search_term in search_terms[:n_topics]: # select first `n_topics` candidates
      if search_term in search_history: # check if the topic is already covered
        continue

      print(f'Fetching page: "{search_term}"')
      search_history.add(search_term) # add to search history

      try:
        # extract the relevant data by using `gemini-2.0-flash` model
        page = wikipedia.page(search_term, auto_suggest=False)
        url = page.url
        print(f"Information Source: {url}")
        search_urls.append(url)
        page = page.content
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=textwrap.dedent(f"""\
                Extract relevant information
                about user's query: {query}
                From this source:

                {page}

                Note: Do not summarize. Only Extract and return the relevant information
        """))

        urls = [url]
        if response.candidates[0].citation_metadata:
          extra_citations = response.candidates[0].citation_metadata.citation_sources
          extra_urls = [source.url for source in extra_citations]
          urls.extend(extra_urls)
          search_urls.extend(extra_urls)
          print("Additional citations:", response.candidates[0].citation_metadata.citation_sources)
        try:
          text = response.text
        except ValueError:
          pass
        else:
          summary_results.append(text + "\n\nBased on:\n  " + ',\n  '.join(urls))

      except DisambiguationError:
        print(f"""Results when searching for "{search_term}" (originally for "{query}")
        were ambiguous, hence skipping""")
        continue

      except PageError:
        print(f'{search_term} did not match with any page id, hence skipping.')
        continue
        
      except:
        print(f'{search_term} did not match with any page id, hence skipping.')
        continue

  print(f"Information Sources:")
  for url in search_urls:
    print('    ', url)

  return summary_results

```

```
example = wikipedia_search(["What are LLMs?"])
```

[Searching for "What are LLMs?", Related search terms: ['Large language model', 'Retrieval-augmented generation', 'Gemini (chatbot)'], Fetching page: "Large language model", Information Source: https://en.wikipedia.org/wiki/Large_language_model, Fetching page: "Retrieval-augmented generation", Information Source: https://en.wikipedia.org/wiki/Retrieval-augmented_generation, Fetching page: "Gemini (chatbot)", Information Source: https://en.wikipedia.org/wiki/Gemini_(chatbot), Information Sources: ..., https://en.wikipedia.org/wiki/Large_language_model, ..., https://en.wikipedia.org/wiki/Retrieval-augmented_generation, ..., https://en.wikipedia.org/wiki/Gemini_(chatbot)]

Here is what the search results look like:

```
from IPython.display import display

for e in example:
  display(to_markdown(e))
```

> A large language model (LLM) is a language model trained with self-supervised machine learning on a vast amount of text, designed for natural language processing tasks, especially language generation.
> The largest and most capable LLMs are generative pretrained transformers (GPTs).
> LLMs can be fine-tuned for specific tasks or guided by prompt engineering.
> These models acquire predictive power regarding syntax, semantics, and ontologies inherent in human language corpora, but they also inherit inaccuracies and biases present in the data they are trained in.
> An LLM is a type of foundation model (large X model) trained on language.
> LLMs are generally based on the transformer architecture.
> Typically, LLMs are trained with single- or half-precision floating point numbers (float32 and float16).
> The qualifier "large" in "large language model" is inherently vague, as there is no definitive threshold for the number of parameters required to qualify as "large".
> 
> Based on:
>   https://en.wikipedia.org/wiki/Large_language_model

> Large language models (LLMs) are a type of model that rely on static training data. They have pre-existing training data and an internal representation of this data. LLMs can generate responses, output, or synthesize answers to user queries. However, LLMs can provide incorrect information, generate misinformation, or hallucinate. They may also struggle to recognize when they lack sufficient information to provide a reliable response, or misinterpret the context of information they retrieve.
> 
> Based on:
>   https://en.wikipedia.org/wiki/Retrieval-augmented_generation

> *   **Definition/Nature:**
>     *   Generative artificial intelligence chatbots (like Gemini and ChatGPT) are "based on" large language models (LLMs).
>     *   Gemini is described as a "multimodal and more powerful LLM touted as the company's 'largest and most capable AI model'."
> *   **Examples of LLMs mentioned:**
>     *   GPT-3 family
>     *   LaMDA (a prototype LLM)
>     *   PaLM (a newer and more powerful LLM from Google)
>     *   Gemini
> 
> Based on:
>   https://en.wikipedia.org/wiki/Gemini_(chatbot)

### Pass the tools to the model

If you pass a list of functions to the `GenerativeModel`'s `tools` argument,
it will extract a schema from the function's signature and type hints, and then pass schema along to the API calls. In response the model may return a `FunctionCall` object asking to call the function.

Note: This approach only handles annotations of `AllowedTypes = int | float | str | dict | list['AllowedTypes']`

The request to the Gemini model will keep a reference to the function inself, so that it _can_ execute the function locally later.

## Generate supporting search queries

In order to have multiple supporting search queries to the user's original query, you will ask the model to generate more such queries. This would help the engine to cover the asked question on comprehensive levels.

```
instructions = """You have access to the Wikipedia API which you will be using
to answer a user's query. Your job is to generate a list of search queries which
might answer a user's question. Be creative by using various key-phrases from
the user's query. To generate variety of queries, ask questions which are
related to  the user's query that might help to find the answer. The more
queries you generate the better are the odds of you finding the correct answer.
Here is an example:

user: Tell me about Cricket World cup 2023 winners.

function_call: wikipedia_search(['What is the name of the team that
won the Cricket World Cup 2023?', 'Who was the captain of the Cricket World Cup
2023 winning team?', 'Which country hosted the Cricket World Cup 2023?', 'What
was the venue of the Cricket World Cup 2023 final match?', 'Cricket World cup 2023',
'Who lifted the Cricket World Cup 2023 trophy?'])

The search function will return a list of article summaries, use these to
answer the  user's question.

Here is the user's query: {query}
"""
```

In order to yield creative and a more random variety of questions, you will set the model's temperature parameter to a value higher. Values can range from [0.0,1.0], inclusive. A value closer to 1.0 will produce responses that are more varied and creative, while a value closer to 0.0 will typically result in more straightforward responses from the model.

## Enable automatic function calling and call the API

Now start a new chat with `enable_automatic_function_calling=True`. With it enabled, the `genai.ChatSession` will handle the back and forth required to call the function, and return the final response:

```
tools = [wikipedia_search]

config = types.GenerateContentConfig(
    temperature=0.6,
    tools=tools
)
```

```
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=config
)

query = "Explain how deep-sea life survives."

res = chat.send_message(instructions.format(query=query))
```

[Searching for "Deep-sea life survival strategies", ..., Information Sources: ..., https://en.wikipedia.org/wiki/Sea_of_Thieves, ..., https://en.wikipedia.org/wiki/Thalassophobia, ..., https://en.wikipedia.org/wiki/Whalefall_(novel), ..., https://en.wikipedia.org/wiki/Deep-sea_fish, ..., https://en.wikipedia.org/wiki/Deep_sea, ..., https://en.wikipedia.org/wiki/Deep-sea_gigantism, ..., https://en.wikipedia.org/wiki/Deep-sea_community, ..., https://en.wikipedia.org/wiki/Hydrothermal_vent, ..., https://en.wikipedia.org/wiki/Marine_snow, ..., https://en.wikipedia.org/wiki/Life_That_Glows, ..., https://en.wikipedia.org/wiki/Sea_urchin, ..., https://en.wikipedia.org/wiki/Oceanic_trench]

```
to_markdown(res.text)
```

> Deep-sea life has evolved remarkable adaptations to survive the harsh conditions of its environment, characterized by immense pressure, perpetual darkness, extremely cold temperatures, and scarce food resources.
> 
> Here's how deep-sea organisms manage to thrive:
> 
> 1.  **Pressure Adaptations:**
>     *   **Internal Pressure Equalization:** Deep-sea creatures maintain an internal body pressure that is equal to the external hydrostatic pressure, preventing them from being crushed.
>     *   **Flexible Bodies:** Many species have gelatinous, watery flesh with minimal bone structure and reduced tissue density. This allows their bodies to compress without damage.
>     *   **Absence of Gas-Filled Spaces:** Most deep-sea fish lack swim bladders (gas-filled organs used for buoyancy in shallower waters) as these would collapse under pressure. Instead, some use lipid-rich tissues or have hydrofoil-like fins for lift.
>     *   **Molecular Adaptations:** At a cellular level, their proteins and enzymes are specially adapted to function under high pressure, often being more rigid or having modified structures (e.g., increased salt bridges in actin, higher proportion of unsaturated fatty acids in cell membranes to maintain fluidity). Some use osmolytes like Trimethylamine N-oxide (TMAO) to protect proteins.
> 
> 2.  **Food Acquisition:**
>     *   **Marine Snow:** In the vast majority of the deep sea, the primary food source is "marine snow"—a continuous shower of organic detritus (dead organisms, fecal pellets, etc.) falling from the productive upper layers of the ocean. Organisms filter this snow or scavenge larger food falls.
>     *   **Chemosynthesis:** Around hydrothermal vents and cold seeps, life thrives independently of sunlight through chemosynthesis. Specialized bacteria and archaea convert chemical compounds (like hydrogen sulfide and methane) from the Earth's interior into organic matter. These microorganisms form the base of the food web, supporting dense communities of unique organisms, often through symbiotic relationships (e.g., tube worms hosting chemosynthetic bacteria).
>     *   **Efficient Feeding:** Due to food scarcity, many deep-sea fish have slow metabolisms and unspecialized diets, preferring to "sit and wait" for prey. They often possess large, hinged, and extensible jaws with sharp, recurved teeth to engulf prey of their own size or larger.
> 
> 3.  **Light and Sensory Adaptations:**
>     *   **Bioluminescence:** In the absence of sunlight, many deep-sea creatures produce their own light through bioluminescence. This is used for various purposes: attracting prey (like the anglerfish's glowing lure), finding mates, deterring predators (e.g., by startling or counter-illuminating their undersides to blend with faint overhead light), and communication.
>     *   **Enhanced Vision:** While some deep-sea fish are blind, others have exceptionally large, tubular eyes with highly sensitive rod cells that are adapted to detect the faintest flickers of bioluminescence or silhouettes against the dim light from above.
>     *   **Other Senses:** Given the limited utility of sight, deep-sea organisms heavily rely on other senses. They possess highly developed lateral line systems to detect changes in water pressure and vibrations, an acute sense of smell (olfactory system) to locate food or mates, and sensitive inner ears. Many also have long feelers or tentacles to navigate and find prey in the darkness.
> 
> 4.  **Metabolic and Physical Adaptations:**
>     *   **Slow Metabolism:** To conserve energy in a food-scarce environment, deep-sea organisms generally have very slow metabolisms and often grow slowly and live long lives.
>     *   **Body Shape and Movement:** Their bodies are often elongated with weak, watery muscles and minimal skeletal structures, which allows them to remain suspended in water with little energy expenditure. Their body shapes are generally better suited for periodic bursts of swimming rather than continuous movement.
>     *   **Deep-Sea Gigantism:** Some deep-sea species exhibit gigantism, growing much larger than their shallow-water relatives. This is thought to be an adaptation to colder temperatures, food scarcity (larger size improves foraging ability and metabolic efficiency), and reduced predation pressure.
> 
> 5.  **Reproduction:**
>     *   Finding a mate in the vast, dark deep sea can be challenging. Adaptations include hermaphroditism (being both male and female) or unique reproductive strategies, such as the parasitic male anglerfish, which permanently attaches to the female, ensuring a mate is always available.

That looks like it worked. You can go through the chat history to see the details of what was sent and received in the function calls:

```
for content in chat._comprehensive_history:
  part = content.parts[0]

  print