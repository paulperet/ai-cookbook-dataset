# Guide: Summarizing Web Page Content with Claude 3 Haiku

In this tutorial, you will learn how to fetch the content of a web page and use Anthropic's Claude 3 Haiku model to generate a concise summary. This workflow is useful for quickly extracting key information from articles, documentation, or news pages.

## Prerequisites

Before you begin, ensure you have:
- An Anthropic API key (available from the [Anthropic Console](https://console.anthropic.com/))
- Python installed on your system

## Setup

First, install the required Python library and set up your Anthropic client.

1. **Install the Anthropic SDK**  
   Open your terminal or notebook and run:

   ```bash
   pip install anthropic
   ```

2. **Import Libraries and Initialize the Client**  
   Create a new Python script or notebook cell and add the following:

   ```python
   import requests
   from anthropic import Anthropic

   # Initialize the Anthropic client with your API key
   # Replace 'your-api-key-here' with your actual key
   client = Anthropic(api_key="your-api-key-here")
   MODEL_NAME = "claude-3-haiku-20240307"
   ```

   > **Note:** Always store your API key securely using environment variables or a secrets manager in production.

## Step 1: Fetch the Web Page Content

You will use the `requests` library to retrieve the HTML content of a target URL.

```python
# Define the URL you want to summarize
url = "https://en.wikipedia.org/wiki/96th_Academy_Awards"

# Send a GET request with a timeout to avoid hanging
response = requests.get(url, timeout=30)

# Check if the request was successful
if response.status_code == 200:
    page_content = response.text
    print("Page fetched successfully.")
else:
    print(f"Failed to fetch the web page. Status code: {response.status_code}")
    exit(1)
```

**Explanation:**  
This code sends an HTTP GET request to the specified URL. If the server responds with a `200 OK` status, the HTML content is stored in `page_content`. Otherwise, the script exits with an error.

## Step 2: Prepare the Prompt for Claude

Claude models accept structured messages. You will wrap the raw HTML in XML-like tags and provide a clear instruction.

```python
# Construct the prompt with the page content and your request
prompt = f"""<content>
{page_content}
</content>

Please produce a concise summary of the web page content."""

# Format the input as a list of message dictionaries
messages = [
    {"role": "user", "content": prompt}
]
```

**Why this works:**  
Placing the raw HTML inside `<content>` tags helps the model distinguish the data from your instructions. The prompt explicitly asks for a "concise summary," guiding the model toward the desired output format.

## Step 3: Generate the Summary with Claude Haiku

Now, call the Claude API to process the prompt and return the summary.

```python
# Send the request to Claude
response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=1024,
    messages=messages
)

# Extract and display the summary
summary = response.content[0].text
print("Summary:\n")
print(summary)
```

**Parameters explained:**
- `model`: Specifies the Claude 3 Haiku model.
- `max_tokens`: Limits the response length to 1024 tokens (adjust as needed).
- `messages`: Contains the prompt you constructed in Step 2.

## Expected Output

After running the code, you should see a summary similar to the following:

```
The 96th Academy Awards ceremony took place on March 10, 2024 at the Dolby Theatre in Los Angeles. The ceremony, hosted by Jimmy Kimmel, presented Academy Awards (Oscars) in 23 categories honoring films released in 2023.

The big winner of the night was the film "Oppenheimer," which won a leading 7 awards including Best Picture, Best Director for Christopher Nolan, and several technical awards. Other major winners were "Poor Things" with 4 awards and "The Zone of Interest" with 2 awards. Several notable records and milestones were set, including Steven Spielberg receiving his 13th Best Picture nomination, and Billie Eilish and Finneas O'Connell becoming the youngest two-time Oscar winners.

The ceremony featured musical performances, tributes to past winners, and a touching "In Memoriam" segment. However, it also faced some criticism, such as the distracting and hard-to-follow "In Memoriam" presentation and political controversy around a director's comments about the Israel-Gaza conflict.
```

## Next Steps

- **Experiment with different URLs** to summarize news articles, blog posts, or documentation.
- **Refine the prompt** to ask for specific details, like key dates, names, or statistics.
- **Extend the workflow** by saving summaries to a file or integrating this into a larger automation pipeline.

You have now successfully built a web page summarizer using Claude 3 Haiku. This pattern can be adapted for various content-processing tasks with minimal changes.