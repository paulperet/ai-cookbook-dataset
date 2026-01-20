

## Gemini API: Getting started with information grounding for Gemini models

In this notebook you will learn how to use information grounding with [Gemini models](https://ai.google.dev/gemini-api/docs/models/).

Information grounding is the process of connecting these models to specific, verifiable information sources to enhance the accuracy, relevance, and factual correctness of their responses. While LLMs are trained on vast amounts of data, this knowledge can be general, outdated, or lack specific context for particular tasks or domains. Grounding helps to bridge this gap by providing the LLM with access to curated, up-to-date information.

Here you will experiment with:
- Grounding information using Google Search grounding
- Grounding real-world information using Google Maps grounding
- Adding YouTube links to gather context information to your prompt
- Using URL context to include website, pdf or image URLs as context to your prompt

## Set up the SDK and the client

### Install SDK

This guide uses the [`google-genai`](https://pypi.org/project/google-genai) Python SDK to connect to the Gemini models.


```
# Grounding with Google Maps was introduced in 1.43
%pip install -q -U "google-genai>=1.43.0"
```

### Set up your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](../quickstarts/Authentication.ipynb) quickstart for an example.


```
from google.colab import userdata

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
```

### Select model and initialize SDK client

Select the model you want to use in this guide, either by selecting one in the list or writing it down. Keep in mind that some models, like the 2.5 ones are thinking models and thus take slightly more time to respond (cf. [thinking notebook](./Get_started_thinking.ipynb) for more details and in particular learn how to switch the thiking off).


```
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

## Use Google Search grounding

Google Search grounding is particularly useful for queries that require current information or external knowledge. Using Google Search, Gemini can access nearly real-time information and better responses.

To enable Google Search, simply add the `google_search` tool in the `generate_content`'s `config` that way:
```
    config={
      "tools": [
        {
          "google_search": {}
        }
      ]
    },
```


```
from IPython.display import HTML, Markdown

response = client.models.generate_content(
    model=MODEL_ID,
    contents="What was the latest Indian Premier League match and who won?",
    config={"tools": [{"google_search": {}}]},
)

# print the response
display(Markdown(f"**Response**:\n {response.text}"))
# print the search details
print(f"Search Query: {response.candidates[0].grounding_metadata.web_search_queries}")
# urls used for grounding
print(f"Search Pages: {', '.join([site.web.title for site in response.candidates[0].grounding_metadata.grounding_chunks])}")

display(HTML(response.candidates[0].grounding_metadata.search_entry_point.rendered_content))
```


**Response**:
 The latest Indian Premier League (IPL) match was the final of the IPL 2025 season, which took place on June 3, 2025. In this match, Royal Challengers Bengaluru defeated Punjab Kings by 6 runs to win their maiden title.


    Search Query: ['latest Indian Premier League match and winner', 'when did IPL 2025 finish', 'IPL 2024 final match and winner']
    Search Pages: olympics.com, wikipedia.org, thehindu.com, olympics.com, skysports.com, wikipedia.org, thehindu.com



You can see that running the same prompt without search grounding gives you outdated information:


```
from IPython.display import Markdown

response = client.models.generate_content(
    model=MODEL_ID,
    contents="What was the latest Indian Premier League match and who won?",
)

# print the response
display(Markdown(response.text))
```


The latest Indian Premier League (IPL) match was the **Final of the IPL 2024 season**.

*   **Match:** Kolkata Knight Riders (KKR) vs. Sunrisers Hyderabad (SRH)
*   **Date:** May 26, 2024
*   **Winner:** **Kolkata Knight Riders (KKR)** won by 8 wickets.


For more examples, please refer to the [dedicated notebook](./Search_Grounding.ipynb).

## Use Google Maps grounding

Google Maps grounding allows you to easily incorporate location-aware functionality into your applications. When a prompt has context related to Maps data, the Gemini model uses Google Maps to provide factually accurate and fresh answers that are relevant to the specified location or general area.

To enable grounding with Google Maps, add the `google_maps` tool in the  `config` argument of `generate_content`, and optionally provide a structured location in the `tool_config`.

```python
client.models.generate_content(
    ...,
    config=types.GenerateContentConfig(
      # Enable the tool.
      tools=[types.Tool(google_maps=types.GoogleMaps())],
      # Provide structured location.
      tool_config=types.ToolConfig(retrieval_config=types.RetrievalConfig(
            lat_lng=types.LatLng(
                latitude=34.050481, longitude=-118.248526))),
    )
)
```


```
from IPython.display import Markdown

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Do any cafes around here do a good flat white? I will walk up to 20 minutes away",
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_maps=types.GoogleMaps())],
        tool_config=types.ToolConfig(
            retrieval_config=types.RetrievalConfig(
                lat_lng=types.LatLng(latitude=40.7680797, longitude=-73.9818957)
            )
        ),
    ),
)

Markdown(f"### Response\n {response.text}")
```


### Response
 Yes, there are several cafes around that do a good flat white within a 20-minute walk.

*   **Tiny Dancer Coffee** specifically mentions serving flat whites, along with espressos and other latte options. It's a cozy subway cafe with a 4.8-star rating and is about a 6.8-minute walk away.
*   **Solid State Coffee** is an easygoing roastery offering thoughtfully sourced brews and has a 4.7-star rating. It's approximately a 4.7-minute walk.
*   **Sote Coffee Roasters** is a warm, laid-back coffee shop serving freshly roasted brews with a 4.9-star rating, about a 5.9-minute walk.
*   **White Noise Coffee - Coffee Shop & Roastery** is an intimate cafe with globally sourced, in-house roasted beans, rated 4.7 stars, and is about a 5.0-minute walk away.
*   **Rex** offers pour-over coffee and espresso drinks and has a 4.6-star rating, located about a 4.8-minute walk from you.
*   **Thē Soirēe** is a cozy cafe featuring espresso drinks, teas, and pastries. It has a 4.7-star rating and is about a 4.4-minute walk away.
*   **Bibble & Sip** is a bakery and coffeehouse serving upscale coffees, rated 4.5 stars, and is about a 4.1-minute walk.


    Search Query: ['cafes with flat white near me']


All grounded outputs require sources to be displayed after the response text. This code snippet will display the sources.


```
def generate_sources(response: types.GenerateContentResponse):
  grounding = response.candidates[0].grounding_metadata
  # You only need to display sources that were part of the grounded response.
  supported_chunk_indices = {i for support in grounding.grounding_supports for i in support.grounding_chunk_indices}

  sources = []
  if supported_chunk_indices:
    sources.append("### Sources from Google Maps")
  for i in supported_chunk_indices:
    ref = grounding.grounding_chunks[i].maps
    sources.append(f"- [{ref.title}]({ref.uri})")

  return "\n".join(sources)


Markdown(generate_sources(response))
```




### Sources from Google Maps
- [Sote Coffee Roasters](https://maps.google.com/?cid=13421224117575076881)
- [Heaven on 7th Marketplace](https://maps.google.com/?cid=13100894621228039586)
- [White Noise Coffee - Coffee Shop & Roastery](https://maps.google.com/?cid=9563404650783060353)
- [Sip + Co.](https://maps.google.com/?cid=4785431035926753688)
- [Weill Café](https://maps.google.com/?cid=16521712104323291061)
- [Down Under Coffee](https://maps.google.com/?cid=3179851379461939943)



The response also includes data you can use to assemble in-line links. See the [Grounding with Google Search docs](https://ai.google.dev/gemini-api/docs/google-search#attributing_sources_with_inline_citations) for an example of this.

### Render the contextual Google Maps widget

If you are building a web-based application, you can add an interactive widget that includes a map view, the contextual location, the places Gemini considered in the query, and review snippets.

To load the widget, perform all of the following steps.
1. [Acquire a Google Maps API key](https://developers.google.com/maps/documentation/javascript/get-api-key), enabled for the Places API and the Maps JavaScript API,
1. Request the widget token in your request (with `GoogleMaps(enable_widget=True)`),
1. [Load the Maps JavaScript API](https://developers.google.com/maps/documentation/javascript/load-maps-js-api) and enable the Places library,
1. Render the [`<gmp-place-contextual/>`](https://developers.google.com/maps/documentation/javascript/reference/places-widget#PlaceContextualElement) element, setting `context-token` to the value of the `google_maps_widget_context_token` returned in the Gemini API response.

Note that generating a widget can add additional latency to the response, so it is recommended that you do not enable the widget if you are not displaying it.

Assuming you have a Google Maps API key with both APIs enabled, the following code shows one way to render the widget.


```
from IPython.display import HTML

# Load or set your Maps API key here.
MAPS_API_KEY = userdata.get("MAPS_API_KEY")

# This is the same request as above, except `enable_widget` is set.
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Do any cafes around here do a good flat white? I will walk up to 20 minutes away",
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_maps=types.GoogleMaps(enable_widget=True))],
        tool_config=types.ToolConfig(
            retrieval_config=types.RetrievalConfig(
                lat_lng=types.LatLng(latitude=40.7680797, longitude=-73.9818957)
            )
        ),
    ),
)

widget_token = response.candidates[0].grounding_metadata.google_maps_widget_context_token

display(Markdown(f"### Response\n {response.text}"))
display(Markdown(generate_sources(response)))
display(HTML(f"""
<!DOCTYPE html>
<html>
  <body>
    <div style="max-width: 500px; margin: 0 auto">
      <script src="https://maps.googleapis.com/maps/api/js?key={MAPS_API_KEY}&loading=async&v=alpha&libraries=places" async></script>
      <gmp-place-contextual context-token="{widget_token}"></gmp-place-contextual>
    </div>
  </body>
</html>
"""))
```


### Response
 There are several highly-rated cafes within a 20-minute walk that serve coffee.

If you're looking for a café open right now, **Heaven on 7th Marketplace** is open 24 hours, has a 4.8-star rating, and is approximately a 2.7-minute walk (576 meters) away. They serve coffee and smoothies along with sandwiches and bagels.

For a café that explicitly mentions flat whites and has a high rating, **Tiny Dancer Coffee** is an excellent option, rated 4.8 stars. They serve espressos and flat whites, as well as oat and matcha latte options. It's about a 6.8-minute walk (1.3 kilometers) away and opens at 7:00 AM local time.

Other well-rated cafes that open soon and are within a short walk include:

*   **Cafe aroma**, with a 4.7-star rating, opens at 6:30 AM and is a 1.6-minute walk (279 meters) away. They offer hot drinks along with bagels, sandwiches, and pastries.
*   **Down Under Coffee**, rated 4.8 stars, opens at 7:30 AM and is a 1.9-minute walk (321 meters) away.
*   **Masseria Caffè**, a 4.6-star rated café, opens at 7:00 AM and is a 2.3-minute walk (472 meters) away. They offer a variety of caffeinated beverages and pastries.
*   **Weill Café**, boasting a 4.9-star rating, opens at 8:00 AM and is a very short 1.7-minute walk (425 meters) away.



### Sources from Google Maps
- [White Noise Coffee - Coffee Shop & Roastery](https://maps.google.com/?cid=9563404650783060353)
- [Tiny Dancer Coffee](https://maps.google.com/?cid=14421445427760414557)
- [Weill Café](https://maps.google.com/?cid=16521712104323291061)
- [maman](https://maps.google.com/?cid=14208928559726348633)
- [Bibble & Sip](https://maps.google.com/?cid=5234372605966457616)



Running and rendering the above code will require a Maps API key. Once you have it working, the widget will look like this.

## Grounding with YouTube links

You can directly include a public YouTube URL in your prompt. The Gemini models will then process the video content to perform tasks like summarization and answering questions about the content.

This capability leverages Gemini's multimodal understanding, allowing it to analyze and interpret video data alongside any text prompts provided.

You do need to explicitly declare the video URL you want the model to process as part of the contents of the request using a `FileData` part. Here a simple interaction where you ask the model to summarize a YouTube video:


```
yt_link = "https://www.youtube.com/watch?v=XV1kOFo1C8M"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=types.Content(
        parts=[
            types.Part(text="Summarize this video."),
            types.Part(file_data=types.FileData(file_uri=yt_link)),
        ]
    ),
)

Markdown(response.text)
```




This video introduces "Gemma Chess," demonstrating how Google's large language model, Gemma, can enhance the game of chess by leveraging its linguistic abilities.

The speaker, Ju-yeong Ji from Google DeepMind, explains that Gemma isn't intended to replace powerful chess engines that excel at calculating moves. Instead, it aims to bring a "new dimension" to chess through understanding and creating text.

The video highlights three key applications:

1.  **Explainer:** Gemma can analyze chess games (e.g., Kasparov vs. Deep Blue) and explain the "most interesting" or strategically significant moves in plain language, detailing their impact, tactical considerations, and psychological aspects, making complex analyses more understandable.
2.  **Storytellers:** Gemma can generate narrative stories about chess games, transforming raw move data into engaging accounts that capture the tension, emotions, and key moments of a match, bringing the game to life beyond just the moves.
3.  **Supporting Chess Learning:** Gemma can act as a personalized chess tutor, explaining concepts like specific openings (e.g., Sicilian Defense) or tactics in an accessible way, even adapting to the user's language and skill level, effectively serving as an always-available, intelligent chess encyclopedia and coach.

By combining the computational strength of traditional chess AI with Gemma's advanced language capabilities, this approach offers a more intuitive and human-friendly way to learn, analyze, and engage with chess.



But you can also use the link as the source of truth for your request. In this example, you will first ask how Gemma models can help on chess games:


```
yt_link = "https://www.youtube.com/watch?v=XV1kOFo1C8M"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=types.Content(
        parts=[
            types.Part(
                text="In 2 paragraph, how Gemma models can help on chess games?"
            ),
            types.Part(file_data=types.FileData(file_uri=yt_link)),
        ]
    ),
)

Markdown(response.text)
```




Gemma models, as large language models (LLMs), can significantly enhance the chess experience by bridging the gap between raw computational power and human understanding. Unlike traditional chess engines that excel at brute-force calculation and generating optimal moves (often in cryptic notation or complex numerical evaluations), Gemma's strength lies in processing