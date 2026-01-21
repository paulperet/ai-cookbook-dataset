# Browser as a Tool: A Guide to Real-Time Web Data with Gemini

Large Language Models (LLMs) are powerful but often lack a direct connection to live data. While features like Google Search grounding provide fresh information, you can achieve true real-time data access by connecting a browser as a tool. This guide demonstrates how to use a browser with the Gemini API, covering the Live Multimodal API and traditional turn-based conversations.

**Note:** For many use cases, Gemini can directly search content, fetch videos, or retrieve context from URLs using built-in tools. Refer to the [Grounding notebook](https://colab.research.google.com/quickstarts/Grounding.ipynb) for details.

## Prerequisites & Setup

First, install the required Python SDK and helper libraries.

```bash
pip install -U -q 'google-genai' markdownify selenium webdriver-manager
```

```python
import os
import json
import time
import requests
import PIL.Image
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

from google import genai
from google.genai import types
from IPython.display import display, HTML, Markdown
import markdownify
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
```

### Configure API Access

Set up your API key. If you're using Google Colab, store your key in a secret named `GOOGLE_API_KEY`.

```python
from google.colab import userdata
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

Initialize the Gemini client and define the models you'll use.

```python
client = genai.Client(api_key=GOOGLE_API_KEY)

# Choose your preferred models
LIVE_MODEL = 'gemini-2.5-flash-native-audio-preview-09-2025'
MODEL = 'gemini-2.5-flash'
```

### Define Helper Functions

Create utilities to render API responses and check website permissions.

```python
def show_parts(r: types.GenerateContentResponse) -> None:
    """Helper for rendering a GenerateContentResponse object."""
    parts = r.candidates[0].content.parts
    if parts is None:
        finish_reason = r.candidates[0].finish_reason
        print(f'{finish_reason=}')
        return

    for part in parts:
        if part.text:
            display(Markdown(part.text))
        elif part.executable_code:
            display(Markdown(f'```python\n{part.executable_code.code}\n```'))
        else:
            print(json.dumps(part.model_dump(exclude_none=True), indent=2))

    grounding_metadata = r.candidates[0].grounding_metadata
    if grounding_metadata and grounding_metadata.search_entry_point:
        display(HTML(grounding_metadata.search_entry_point.rendered_content))


def can_crawl_url(url: str, user_agent: str = "*") -> bool:
    """Check robots.txt to determine if a URL can be crawled."""
    try:
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        rp = RobotFileParser(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception as e:
        print(f"Error checking robots.txt: {e}")
        return False  # Fail closed to be a good citizen
```

---

## Part 1: Browsing Live with the Multimodal Live API

This section shows how to use the Live API with the Google Search tool, then introduces a custom web browsing tool for real-time data.

### Step 1: Use Google Search as a Tool

The Live API streams responses, requiring you to handle tool calls within the stream. This example streams text, but the technique applies to all supported modalities.

First, define a function to handle the streamed response and tool calls.

```python
async def stream_response(stream, *, tool=None):
    """Handle a live streamed response, printing text and issuing tool calls."""
    all_responses = []

    async for msg in stream.receive():
        all_responses.append(msg)

        if text := msg.text:
            # Print streamed text responses.
            print(text, end='')

        elif tool_call := msg.tool_call:
            # Handle tool calls.
            for fc in tool_call.function_calls:
                print(f'< Tool call', fc.model_dump(exclude_none=True))

                if tool:
                    # Call the actual tool.
                    assert fc.name == tool.__name__, "Unknown tool call encountered"
                    tool_result = tool(**fc.args)
                else:
                    # Mock tool calls by returning 'ok'.
                    tool_result = 'ok'

                tool_response = types.LiveClientToolResponse(
                    function_responses=[types.FunctionResponse(
                        name=fc.name,
                        id=fc.id,
                        response={'result': tool_result},
                    )]
                )
                await stream.send(input=tool_response)

    return all_responses
```

Now, configure the model with the Google Search tool and run a conversation.

```python
config = {
    'response_modalities': ['TEXT'],
    'tools': [
        {'google_search': {}},
    ],
}

async def run():
    async with client.aio.live.connect(model=LIVE_MODEL, config=config) as stream:
        await stream.send(input="What is today's featured article on the English Wikipedia?", end_of_turn=True)
        await stream_response(stream)

await run()
```

**Output:**
```
Today's featured article on the English Wikipedia is about the 2009-10 season of the English football club, Notts County F.C. ...
```

You might notice a discrepancy between Google's indexed data and the live Wikipedia page. To get truly current information, add a custom browser tool.

### Step 2: Add a Live Browser Tool

Define a tool that fetches a webpage, converts its HTML to Markdown, and returns the content.

```python
def load_page(url: str) -> str:
    """Load the page contents as Markdown."""
    if not can_crawl_url(url):
        return f"URL {url} failed a robots.txt check."
    try:
        page = requests.get(url)
        return markdownify.markdownify(page.content)
    except Exception as e:
        return f"Error accessing URL: {e}"
```

Configure the model with this new tool and a system instruction to guide its use.

```python
load_page_def = types.Tool(functionDeclarations=[
    types.FunctionDeclaration.from_callable(client=client, callable=load_page)]).model_dump(exclude_none=True)

config = {
    'response_modalities': ['TEXT'],
    'tools': [
        load_page_def,
    ],
    'system_instruction': """Your job is to answer the users query using the tools available.
First determine the address that will have the information and tell the user. Then immediately
invoke the tool. Then answer the user."""
}

async def run():
    async with client.aio.live.connect(model=LIVE_MODEL, config=config) as stream:
        await stream.send(input="What is today's featured article on the English Wikipedia?", end_of_turn=True)
        await stream_response(stream, tool=load_page)

await run()
```

**Output:**
```
I can find that information for you. I will use the Wikipedia Main Page to find the featured article.
< Tool call {'id': 'function-call-14532754128504730670', 'args': {'url': 'https://en.wikipedia.org/wiki/Main_Page'}, 'name': 'load_page'}
Today's featured article on the English Wikipedia is about John Silva Meehan, an American publisher, printer, and newspaper editor who also served as the Librarian of Congress.
```

The model now uses your custom browser to fetch live data directly from Wikipedia.

---

## Part 2: Browsing Pages Visually

Web pages are multi-modal. A text-only approach loses visual context and cannot handle JavaScript-rendered content. In this section, you'll create a tool that uses a headless browser to capture screenshots.

**Note:** This example uses Selenium with headless Chrome and is designed for a Linux environment like Google Colab.

### Step 1: Install Browser Dependencies

```bash
apt install -y chromium-browser
```

### Step 2: Define a Graphical Browser Tool

Create a function that uses Selenium to load a URL, wait for it to render, and take a screenshot.

```python
SCREENSHOT_FILE = 'screenshot.png'

def browse_url(url: str) -> str:
    """Captures a screenshot of the webpage at the provided URL."""
    if not can_crawl_url(url):
        return f"URL {url} failed a robots.txt check."

    try:
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.headless = True
        driver = webdriver.Chrome(options=chrome_options)

        # Set a large window size to capture most of the page.
        driver.set_window_size(1024, 2048)
        driver.get(url)

        # Wait for the page to fully load.
        time.sleep(5)
        driver.save_screenshot(SCREENSHOT_FILE)

        print(f"Screenshot saved to {SCREENSHOT_FILE}")
        return markdownify.markdownify(driver.page_source)

    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e)

    finally:
        # Close the browser
        if driver:
            driver.quit()
```

Test the function.

```python
url = "https://en.wikipedia.org/wiki/Castle"
browse_url(url)
```

**Output:**
```
Screenshot saved to screenshot.png
```

### Step 3: Connect the Browser to the Model

LLMs are typically trained to state they cannot access the internet. To override this, provide a system instruction that explicitly grants web access via your tool.

```python
sys_int = """You are a system with access to websites via the `browse_url` tool.
Use the `browse_url` tool to browse a URL and generate a screenshot that will be
returned for you to see and inspect, like using a web browser.

When a user requests information, first use your knowledge to determine a specific
page URL, tell the user the URL and then invoke the `browse_tool` with this URL. The
tool will supply the website, at which point you will examine the contents of the
screenshot to answer the user's questions. Do not ask the user to proceed, just act.

You will not be able to inspect the page HTML, so determine the most specific page
URL, rather than starting navigation from a site's homepage.
"""

browse_tool = types.Tool(functionDeclarations=[
    types.FunctionDeclaration.from_callable(client=client, callable=browse_url)])

chat = client.chats.create(
    model=MODEL,
    config={'tools': [browse_tool], 'system_instruction': sys_int})

r = chat.send_message('What is trending on YouTube right now?')
show_parts(r)
```

**Output:**
```
I do not have access to the YouTube API or a general trending topics API. However, I can browse the YouTube trending page for you. I will use the `browse_url` tool to access the YouTube trending page and provide you with a screenshot. The URL for the YouTube trending page is: `https://www.youtube.com/feed/trending`.

{
  "function_call": {
    "args": {
      "url": "https://www.youtube.com/feed/trending"
    },
    "name": "browse_url"
  }
}
```

The model has issued a function call. Execute the tool and prepare the response.

### Step 4: Execute the Tool and Prepare the Response

When the model calls `browse_url`, you must run the function and return both a text response and the generated image.

```python
response_parts = []

# For each function call, generate the response in two parts.
for p in r.candidates[0].content.parts:
    if fn := p.function_call:
        assert fn.name == 'browse_url'

        url = fn.args['url']
        print(url)
        response = browse_url(url)
        print(response)

        img = PIL.Image.open(SCREENSHOT_FILE)

        fr = genai.types.Part(function_response=genai.types.FunctionResponse(
            name=fn.name,
            id=fn.id,
            response={'result': response},
        ))
        response_parts.extend([fr, img])
```

**Output:**
```
https://www.youtube.com/feed/trending
Screenshot saved to screenshot.png
ok
```

You can now inspect the screenshot (`screenshot.png`) before sending it back to the model in the next turn of the conversation.

---

## Summary

You've learned how to:
1. Use the Live Multimodal API with the built-in Google Search tool.
2. Create a custom browser tool to fetch live webpage content as text.
3. Build a graphical browser tool using Selenium to capture screenshots, enabling access to JavaScript-rendered content and visual data.

These techniques allow you to connect Gemini models to real-time, live data sources, overcoming the limitations of static knowledge cutoffs.