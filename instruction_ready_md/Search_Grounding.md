# Gemini API Cookbook: Search Grounding for Real-Time Information

This guide walks you through using the Gemini API's Search Grounding feature to generate responses backed by real-time web search data. You'll learn how to authenticate, make an API call, and parse the structured results.

## Prerequisites & Setup

Before you begin, ensure you have a valid Google AI API key. This tutorial assumes you are using Google Colab, but the principles apply to any environment.

### 1. Install Required Packages
We'll use `curl` for the API request and `jq` to parse the JSON response.

```bash
# Install jq for JSON parsing
sudo apt install -q jq
```

### 2. Configure Authentication
Store your API key securely. In Colab, you can use a Secret. The following code retrieves the key and sets it as an environment variable.

```python
import os
from google.colab import userdata

# Retrieve your API key from Colab Secrets and set it as an environment variable.
# Replace 'GOOGLE_API_KEY' with your secret's name if different.
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

**Note for Other Environments:** If you're not using Colab, set the `GOOGLE_API_KEY` environment variable directly in your shell or within your application.

---

## Step 1: Make a Search-Grounded API Call

Now, you will call the Gemini API with a query that requires up-to-date information. The `google_search` tool in the request enables Search Grounding.

First, define the model you want to use.

```python
# Define your target Gemini model
MODEL_ID = "gemini-3-flash-preview"
```

Next, construct and send the API request using `curl`. This command asks the model for the current Google stock price and instructs it to use web search.

```bash
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

This command saves the full API response to a file named `result.json`.

---

## Step 2: Explore the API Response

The response is a rich JSON object containing the model's answer and metadata about the sources it used.

### View the Full Response
Use `jq` to pretty-print and explore the entire response structure.

```bash
jq . -r result.json
```

### Extract the Text Answer
The primary answer from the model is contained within the `text` field. Let's extract it.

```bash
jq -r ".candidates[0].content.parts[0].text" result.json
```

**Example Output:**
```
Here are the current prices for Google stock, as of February 12, 2025:

*   **GOOG (Alphabet Inc Class C):** $187.07
*   **GOOGL (Alphabet Inc Class A):** $185.37
```

### Examine the Grounding Metadata
Crucially, the response includes `groundingMetadata`, which details the search results that supported the answer. This is essential for transparency.

```bash
jq -r ".candidates[0].groundingMetadata" result.json
```

This metadata includes citations and a `searchEntryPoint`.

---

## Step 3: Handle the Search Suggestion (Required)

When you use Search Grounding, you **must** comply with Google's [usage requirements](https://googledevai.devsite.corp.google.com/gemini-api/docs/grounding/search-suggestions?hl=en#requirements). A key requirement is to display the provided "Search Suggestion" and link users directly to the Google Search results page.

The `renderedContent` within the `searchEntryPoint` contains the HTML for this suggestion. You must present this to users.

First, extract the HTML content.

```bash
jq -r ".candidates[0].groundingMetadata.searchEntryPoint.renderedContent" result.json > rendered_content.html
```

You can then render this HTML in your application. Here's an example of how to display it in a Python environment.

```python
from IPython.display import HTML

# Read the saved HTML file and display it
with open('rendered_content.html', 'r') as file:
    html_content = file.read()

HTML(html_content)
```

**Important:** The rendered content typically includes a link styled as "Search Google for...". You must ensure this link, when clicked, takes the user to the official Google Search results page for the query.

---

## Summary

You have successfully:
1.  Authenticated with the Gemini API.
2.  Sent a prompt using the `google_search` tool to enable Search Grounding.
3.  Parsed the response to get a factually grounded answer.
4.  Extracted the mandatory Search Suggestion link to comply with usage terms.

This pattern allows you to build applications where Gemini's responses are enhanced and verified by real-time web data, while maintaining proper attribution.