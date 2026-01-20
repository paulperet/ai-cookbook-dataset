

# Gemini API: Search Grounding

### Authentication

To run this notebook, your API key must be stored it in a Colab Secret named GOOGLE_API_KEY. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](../../quickstarts/Authentication.ipynb) to learn more.

This first cell is in python, just to copy your API key to an environment variable, so you can access it from the shell:


```
import os
from google.colab import userdata
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

## Call the API

Call search grounding.


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```


```bash
%%bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
      "contents": [
          {
              "parts": [
                  {"text": "What is the current Google stock price?"}
              ]
          }
      ],
      "tools": [
          {
              "google_search": {}
          }
      ]
  }' > result.json
```

[% Total % Received % Xferd Average Speed Time Time Time Current Dload Upload Total Spent Left Speed 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0 100 252 0 0 100 252 0 184 0:00:01 0:00:01 --:--:-- 184 100 7793 0 7541 100 252 3904 130 0:00:01 0:00:01 --:--:-- 4033 100 7793 0 7541 100 252 3904 130 0:00:01 0:00:01 --:--:-- 4033]

## Explore the output

Use `jq` to colorize the output, and make it easier to explore.


```
!sudo apt install -q jq
```

Here's all the output:


```
!jq . -r result.json
```

Here is the text response:


```
!jq -r ".candidates[0].content.parts[0].text" result.json
```

    Here are the current prices for Google stock, as of February 12, 2025:
    
    *   **GOOG (Alphabet Inc Class C):** $187.07
    *   **GOOGL (Alphabet Inc Class A):** $185.37
    

Here is the `groundingMetadata`, including links to any supports used:


```
!jq -r ".candidates[0].groundingMetadata" result.json
```

The `rendered_content` is how you link users to the google-search results that helped produce the response:

> Important: If you use search grounding you **must** follow the [requirements outlined here](https://googledevai.devsite.corp.google.com/gemini-api/docs/grounding/search-suggestions?hl=en#requirements), which includes "Display the Search Suggestion exactly as provided" and "Take users directly to the Google Search results page (SRP) when they interact with the Search Suggestion".


```
!jq -r ".candidates[0].groundingMetadata.searchEntryPoint.renderedContent" result.json > rendered_content.html
```


```
# Python so you can display it in this notebook
from IPython.display import HTML
HTML('rendered_content.html')
```