# Guide: Information Grounding with the Gemini API

## Overview
This guide demonstrates how to enhance Gemini model responses by grounding them in specific, verifiable information sources. You will learn to connect the model to real-time data from Google Search, location-based data from Google Maps, video context from YouTube, and general web content via URLs.

Grounding improves the accuracy, relevance, and factual correctness of model outputs by providing curated, up-to-date context beyond the model's static training data.

## Prerequisites

### 1. Install the SDK
Ensure you have the latest `google-genai` Python SDK installed. Grounding with Google Maps requires version 1.43.0 or higher.

```bash
pip install -U "google-genai>=1.43.0"
```

### 2. Set Up Your API Key
Store your Gemini API key in an environment variable named `GOOGLE_API_KEY`. If you're using Google Colab, you can store it as a secret.

```python
import os
from google import genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Or in Colab:
# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
```

### 3. Initialize the Client and Select a Model
Create a client instance and choose a Gemini model. For this guide, we'll use `gemini-2.0-flash-exp`, but you can select any available model.

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.0-flash-exp"  # Or your preferred model
```

---

## Step 1: Grounding with Google Search
Google Search grounding provides the model with near real-time information, ideal for queries about current events or external knowledge.

### How to Enable It
Add the `google_search` tool to the `config` parameter in your `generate_content` call.

### Example: Querying a Recent Sports Event
Let's ask about the latest Indian Premier League (IPL) cricket match.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents="What was the latest Indian Premier League match and who won?",
    config={"tools": [{"google_search": {}}]},
)

print("Response:", response.text)
print("Search Query:", response.candidates[0].grounding_metadata.web_search_queries)
print("Sources:", ', '.join([site.web.title for site in response.candidates[0].grounding_metadata.grounding_chunks]))
```

**Expected Output:**
The response will include the most recent match details, sourced from current web results. The search queries and source URLs are also provided in the metadata.

### Comparison: Without Grounding
Running the same prompt without search grounding may return outdated information, as the model relies on its static training data.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents="What was the latest Indian Premier League match and who won?",
)
print(response.text)
```

**Note:** The ungrounded response might reference an older season, highlighting the value of real-time grounding.

---

## Step 2: Grounding with Google Maps
Google Maps grounding enables location-aware queries, providing accurate, fresh information about places, businesses, and directions.

### How to Enable It
Add the `google_maps` tool to the config and optionally provide a structured location via `tool_config`.

### Example: Finding Nearby Cafes
Let's search for cafes serving a good flat white within a 20-minute walk of a specific location.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Do any cafes around here do a good flat white? I will walk up to 20 minutes away",
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_maps=types.GoogleMaps())],
        tool_config=types.ToolConfig(
            retrieval_config=types.RetrievalConfig(
                lat_lng=types.LatLng(latitude=40.7680797, longitude=-73.9818957)  # Example NYC coordinates
            )
        ),
    ),
)

print(response.text)
```

### Displaying Grounding Sources
It's important to attribute the sources used in the grounded response. The following helper function extracts and formats source links from Google Maps.

```python
def generate_sources(response: types.GenerateContentResponse):
    grounding = response.candidates[0].grounding_metadata
    # Collect indices of chunks actually used in the response
    supported_chunk_indices = {i for support in grounding.grounding_supports for i in support.grounding_chunk_indices}

    sources = []
    if supported_chunk_indices:
        sources.append("### Sources from Google Maps")
    for i in supported_chunk_indices:
        ref = grounding.grounding_chunks[i].maps
        sources.append(f"- [{ref.title}]({ref.uri})")
    return "\n".join(sources)

print(generate_sources(response))
```

### Rendering an Interactive Maps Widget (Web Applications)
For web apps, you can render an interactive Google Maps widget that visualizes the contextual location and places.

**Prerequisites:**
1.  A Google Maps API key with the **Places API** and **Maps JavaScript API** enabled.
2.  Request the widget token by setting `enable_widget=True` in the `GoogleMaps` tool configuration.

**Example Code:**

```python
# Set your Maps API key
MAPS_API_KEY = "YOUR_MAPS_API_KEY"

# Request with widget enabled
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

# HTML to render the widget
html_content = f"""
<!DOCTYPE html>
<html>
  <body>
    <div style="max-width: 500px; margin: 0 auto">
      <script src="https://maps.googleapis.com/maps/api/js?key={MAPS_API_KEY}&loading=async&v=alpha&libraries=places" async></script>
      <gmp-place-contextual context-token="{widget_token}"></gmp-place-contextual>
    </div>
  </body>
</html>
"""
# Display this HTML in your web application
```

**Note:** Enabling the widget adds latency; only use it if you intend to display the interactive map.

---

## Step 3: Grounding with YouTube Links
You can provide a YouTube video URL as context, enabling the model to answer questions or summarize the video's content.

### How to Enable It
Include the YouTube URL as a `FileData` part within the `contents` of your request.

### Example: Summarizing a Video
Let's ask the model to summarize a specific YouTube video.

```python
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

print(response.text)
```

### Example: Using Video as a Knowledge Source
You can also use the video as the primary source for a query.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=types.Content(
        parts=[
            types.Part(text="In 2 paragraphs, how can Gemma models help with chess games?"),
            types.Part(file_data=types.FileData(file_uri=yt_link)),
        ]
    ),
)

print(response.text)
```

The model will generate an answer based solely on the content of the provided video.

---

## Step 4: Grounding with URL Context (Websites, PDFs, Images)
Beyond YouTube, you can ground prompts with content from various web URLs, including websites, PDF documents, and images.

### How to Enable It
Similar to YouTube, pass the URL as a `FileData` part. The model will fetch and process the content from the given URI.

### Example: Summarizing a Web Article
```python
web_url = "https://example.com/article"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=types.Content(
        parts=[
            types.Part(text="What are the key points of this article?"),
            types.Part(file_data=types.FileData(file_uri=web_url)),
        ]
    ),
)

print(response.text)
```

This method works for publicly accessible `https://` URLs pointing to supported content types.

## Summary
You have now learned to ground Gemini model responses using four powerful methods:
1.  **Google Search:** For real-time, factual queries.
2.  **Google Maps:** For location-based information and places.
3.  **YouTube Links:** For video analysis and summarization.
4.  **URL Context:** For grounding in web pages, PDFs, and images.

Grounding ensures your AI applications deliver accurate, relevant, and up-to-date information by connecting the model's reasoning to specific, verifiable data sources.