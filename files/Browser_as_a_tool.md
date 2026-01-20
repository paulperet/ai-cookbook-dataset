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

# Browser as a tool

LLMs are powerful tools, but are not intrinsically connected to live data sources. Features like Google Search grounding provide fresh information using Google's search index, but to supply truly live information, you can connect a browser to provide up-to-the-minute data and smart exploration.

This notebook will guide you through three examples of using a browser as a tool with the Gemini API, using both the [Live Multimodal API](https://ai.google.dev/api/multimodal-live) and traditional turn-based conversations.

* Requesting live data using a browser tool with the Live API
* Returning images of web pages from function calling
* Connecting to a local network/intranet using a browser tool

**Note**: for most of the use-cases, you can use the tools directly by Gemini to get it to search content using Google Search, grab videos from YouTube or fetch context from URLs without having to set-up anything. Check out the [Grounding](https://colab.research.google.com/quickstarts/Grounding.ipynb) notebook for more details.


## Set up the SDK

This guide uses the [`google-genai`](https://pypi.org/project/google-genai) Python SDK to connect to the Gemini models.


```
%pip install -U -q 'google-genai'

from google import genai
genai.__version__
```

    [First Entry, ..., Last Entry]



    '1.2.0'



### Set up your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](../quickstarts/Authentication.ipynb) quickstart for an example.


```
import os
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

### Create the SDK client

You will use the same `client` instance for both the Live API and the classic REST API interactions, so define models for each.


```
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)

LIVE_MODEL = 'gemini-2.5-flash-native-audio-preview-09-2025'  # @param ['gemini-2.0-flash-live-001', 'gemini-live-2.5-flash-preview', 'gemini-2.5-flash-native-audio-preview-09-2025'] {allow-input: true, isTemplate: true}
MODEL = 'gemini-2.5-flash'  # @param ['gemini-2.5-flash'] {allow-input: true, isTemplate: true}
```

### Define some helpers

The `show_parts` helper renders the deeply nested output that the API returns in an notebook-friendly way; handling text, code and tool calls.

The `can_crawl_url` helper will perform a [`robots.txt`](https://developers.google.com/search/docs/crawling-indexing/robots/intro) check to ensure any automated requests are welcome by the remote service.


```
import json
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

from IPython.display import display, HTML, Markdown


def show_parts(r: types.GenerateContentResponse) -> None:
  """Helper for rendering a GenerateContentResponse object in IPython."""
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
    """Look up robots.txt for a URL and determine if crawling is permissable.

    Args:
        url: The full URL to check.
        user_agent: The user agent to check, defaults to any UA.

    Returns:
        True if the URL can be crawled, False otherwise.
    """
    try:
      parsed_url = urlparse(url)
      robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
      rp = RobotFileParser(robots_url)
      rp.read()

      return rp.can_fetch(user_agent, url)

    except Exception as e:
      print(f"Error checking robots.txt: {e}")
      return False  # Be a good citizen: fail closed.
```

## Browsing live

This example will show you how to use the Multimodal Live API with the Google Search tool, and then comparatively shows a custom web browsing tool to retrieve site contents in real-time.


### Use Google Search as a tool

The streaming nature of the Live API requires that the stream processing and function handling code be written in advance. This allows the stream to continue without timing out.

This example uses text as the input mode, and streams text back out, but the technique applies any mode supported by the Live API, including audio.


```
config = {
    'response_modalities': ['TEXT'],
    'tools': [
        {'google_search': {}},
    ],
}


async def stream_response(stream, *, tool=None):
  """Handle a live streamed response, printing out text and issue tool calls."""
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
          # Call the tool.
          assert fc.name == tool.__name__, "Unknown tool call encountered"
          tool_result = tool(**fc.args)

        else:
          # Return 'ok' as a way to mock tool calls.
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

Now define and run the conversation.


```
async def run():
  async with client.aio.live.connect(model=LIVE_MODEL, config=config) as stream:

    await stream.send(input="What is today's featured article on the English Wikipedia?", end_of_turn=True)
    await stream_response(stream)

await run()
```

    
    Today's featured article on the English Wikipedia is about the 2009-10 season of the English football club, Notts County F.C. The article highlights the club's takeover by Munto Finance, controlled by a convicted fraudster, as part of a scheme to list a fake mining company on the stock exchange, and the subsequent collapse of the scheme, which left Notts County in debt.
    
    Featured articles on Wikipedia are those that have been reviewed for accuracy, neutrality, completeness, and style. They are considered to be among the best articles the platform has to offer. A summary of the featured article is displayed daily on the main page and is viewed by millions of users.


Depending on when you run this, you may note a discrepency between what Google Search has in its index, and what is currently live on Wikipedia. Check out [Wikipedia's featured article](https://en.wikipedia.org/wiki/Main_Page#mp-tfa) yourself. Alternatively, the model may decide not to answer due to the requirement for freshness.

To improve this situation, add a browse tool so the model can acquire this information in real-time.

### Add a live browser

This step defines a "browser" that requests a URL over HTTP(S), converts the response to markdown and returns it.

This technique works for sites that serve content as full HTML, so sites that rely on scripting to serve content, such as a [PWA](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps) without [SSR](https://developer.mozilla.org/en-US/docs/Glossary/SSR), will not work. Check out the visual example later that uses a fully-featured browser.


```
%pip install -q markdownify
```


```
import requests

import markdownify


def load_page(url: str) -> str:
  """
  Load the page contents as Markdown.
  """

  if not can_crawl_url(url):
    return f"URL {url} failed a robots.txt check."

  try:
    page = requests.get(url)
    return markdownify.markdownify(page.content)

  except Exception as e:
    return f"Error accessing URL: {e}"

```

Now define and run the conversation using the new tool. Here an extended system instruction has been added to coerce the model into calling the tool immediately, so that it doesn't engage in an open-ended conversation that's hard to demonstrate in a notebook.


```
load_page_def = types.Tool(functionDeclarations=[
    types.FunctionDeclaration.from_callable(client=client, callable=load_page)]).model_dump(exclude_none=True)

config = {
    'response_modalities': ['TEXT'],
    'tools': [
        load_page_def,
    ],
    'system_instruction': """Your job is to answer the users query using the tools available.

First determine the address that will have the information and tell the user. Then immediately
invoke the tool. Then answer the user.
"""
}


async def run():
  async with client.aio.live.connect(model=LIVE_MODEL, config=config) as stream:

    await stream.send(input="What is today's featured article on the English Wikipedia?", end_of_turn=True)
    await stream_response(stream, tool=load_page)

await run()
```

    I can find that information for you. I will use the Wikipedia Main Page to find the featured article.
    
    
    < Tool call {'id': 'function-call-14532754128504730670', 'args': {'url': 'https://en.wikipedia.org/wiki/Main_Page'}, 'name': 'load_page'}
    Today's featured article on the English Wikipedia is about John Silva Meehan, an American publisher, printer, and newspaper editor who also served as the Librarian of Congress.


## Browse pages visually

In the previous example, you used a tool to retrieve a page's textual content and use it in a live chat context. However, web pages are a rich multi-modal medium, so using text results in some loss of signal. Using a fully-featured web browser also enables websites that use JavaScript to render content, something that is not possible using a simple HTTP request like the earlier example.

In this example, you will define a tool that takes a screenshot of a web page and passes the image back to the model.

Note: This example automates a headless Chromium browser, so the instructions are specific to a Linux environment and will run on Google Colab. Try this example on Colab, or check out the [Selenium documentation](https://www.selenium.dev/documentation/webdriver/browsers/) for setting up specific browsers in your environment.


```
!apt install -y chromium-browser
```


```
%pip install -q selenium webdriver-manager
```

    [First Entry, ..., Last Entry]

### Define a graphical browser

Here you define a `browse_url` function that uses [Selenium](https://selenium-python.readthedocs.io/) to load a headless web browser, navigate to a URL and take a screenshot. This technique takes a single screenshot at a fixed size. There are other tools, such as [`selenium-screenshot`](https://pypi.org/project/selenium_screenshot), that can capture full-length images by repeatedly scrolling and capturing the page. As this tool is intended for use during a live conversation, this example uses the faster single-shot approach.


```
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

SCREENSHOT_FILE = 'screenshot.png'


def browse_url(url: str) -> str:
    """Captures a screenshot of the webpage at the provided URL.

    A graphical browser will be used to connect to the URL provided,
    and generate a screenshot of the rendered web page.

    Args:
        url: The full absolute URL to browse/screenshot.

    Returns:
        "ok" if successfully captured, or any error messages.
    """
    if not can_crawl_url(url):
      return f"URL {url} failed a robots.txt check."

    try:
      chrome_options = webdriver.ChromeOptions()
      chrome_options.add_argument('--headless')
      chrome_options.add_argument('--no-sandbox')
      chrome_options.headless = True
      driver = webdriver.Chrome(options=chrome_options)

      # Take one large image, 2x high as it is wide. This should be enough to
      # capture most of a page's interesting info, and should capture anything
      # designed "above the fold", without going too deep into things like
      # footer links, infinitely scrolling pages, etc.
      # Otherwise multiple images are needed, which requires waiting, scrolling
      # and stitching, and introduces lag that slows down interactions.
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


url = "https://en.wikipedia.org/wiki/Castle"
browse_url(url);
```

    Screenshot saved to screenshot.png





    'ok'



Check out the screenshot to make sure it worked.

### Connect the browser to the model

Add the `browse_url` tool to a model and start a chat session. As LLMs do not directly have internet connectivity, modern models like Gemini are trained to tell users that they can't access the internet, rather than hallucinating results. To override this behaviour, this step adds a system instruction that guides the model to use the tool for internet access.


```
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

# Because `browse_url` generates an image, and images can't be used in function calling
# (but can be used in regular Content/Parts), automatic function calling can't be used and
# the tool must be specified explicitly, and handled manually.
browse_tool = types.Tool(functionDeclarations=[
    types.FunctionDeclaration.from_callable(client=client, callable=browse_url)])

chat = client.chats.create(
    model=MODEL,
    config={'tools': [browse_tool], 'system_instruction': sys_int})

r = chat.send_message('What is trending on YouTube right now?')
show_parts(r)
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


You should see a `function_call` in the response above. Once the model issues a function call, execute the tool and save both the `function_response` and the image for the next turn.

If you do not see a `function_call`, you can either re-run the cell, or continue the chat to answer any questions the model has (e.g. `r = chat.send_message('Yes, please use the tool')`).


```
import PIL

response_parts = []

# For each function call, generate the response in two parts. Once for the
# function response, and one for the image as regular content. This simulates
# the function "returning" an image to the model as part of a function call.
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

    https://www.youtube.com/feed/trending
    Screenshot saved to screenshot.png
    ok


Inspect the image before it is sent back to the model. Depending on where you are running this, you may see localised content. If you are using Google Colab, you can run `!curl ipinfo.io` to see the geolocation of the running kernal.

Note that if you see a semi-bl