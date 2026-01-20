# Have several agents collaborate in a multi-agent hierarchy 🤖🤝🤖
_Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_

> This tutorial is advanced. You should have notions from [this other cookbook](agents) first!

In this notebook we will make a **multi-agent web browser: an agentic system with several agents collaborating to solve problems using the web!**

It will be a simple hierarchy, using a `ManagedAgent` object to wrap the managed web search agent:

```
              +----------------+
              | Manager agent  |
              +----------------+
                       |
        _______________|______________
       |                              |
  Code interpreter   +--------------------------------+
       tool          |         Managed agent          |
                     |      +------------------+      |
                     |      | Web Search agent |      |
                     |      +------------------+      |
                     |         |            |         |
                     |  Web Search tool     |         |
                     |             Visit webpage tool |
                     +--------------------------------+
```
Let's set up this system. 

Run the line below to install the required dependencies:


```python
!pip install markdownify duckduckgo-search smolagents --upgrade -q
```

Let's login in order to call the HF Inference API:


```python
from huggingface_hub import notebook_login

notebook_login()
```

⚡️ Our agent will be powered by [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) using `HfApiEngine` class that uses HF's Inference API: the Inference API allows to quickly and easily run any OS model.

_Note:_ The Inference API hosts models based on various criteria, and deployed models may be updated or replaced without prior notice. Learn more about it [here](https://huggingface.co/docs/api-inference/supported-models).


```python
model_id = "Qwen/Qwen2.5-72B-Instruct"
```

### 🔍 Create a web search tool

For web browsing, we can already use our pre-existing [`DuckDuckGoSearchTool`](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/search.py) tool to provide a Google search equivalent.

But then we will also need to be able to peak into the page found by the `DuckDuckGoSearchTool`.
To do so, we could import the library's built-in `VisitWebpageTool`, but we will build it again to see how it's done.

So let's create our `VisitWebpageTool` tool from scratch using `markdownify`.


```python
import re
import requests
from markdownify import markdownify as md
from requests.exceptions import RequestException
from smolagents import tool


@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = md(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
```

Ok, now let's initialize and test our tool!


```python
print(visit_webpage("https://en.wikipedia.org/wiki/Hugging_Face")[:500])
```

    Hugging Face - Wikipedia
    
    [Jump to content](#bodyContent)
    
    Main menu
    
    Main menu
    move to sidebar
    hide
    
    Navigation
    
    * [Main page](/wiki/Main_Page "Visit the main page [z]")
    * [Contents](/wiki/Wikipedia:Contents "Guides to browsing Wikipedia")
    * [Current events](/wiki/Portal:Current_events "Articles related to current events")
    * [Random article](/wiki/Special:Random "Visit a randomly selected article [x]")
    * [About Wikipedia](/wiki/Wikipedia:About "Learn about Wikipedia and how it works")
    * [Contac


## Build our multi-agent system 🤖🤝🤖

Now that we have all the tools `search` and `visit_webpage`, we can use them to create the web agent.

Which configuration to choose for this agent?
- Web browsing is a single-timeline task that does not require parallel tool calls, so JSON tool calling works well for that. We thus choose a `ReactJsonAgent`.
- Also, since sometimes web search requires exploring many pages before finding the correct answer, we prefer to increase the number of `max_iterations` to 10.


```python
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    ManagedAgent,
    DuckDuckGoSearchTool
)

model = InferenceClientModel(model_id)

web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    model=model,
    max_iterations=10,
)
```

We then wrap this agent into a `ManagedAgent` that will make it callable by its manager agent.


```python
managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="search_agent",
    description="Runs web searches for you. Give it your query as an argument.",
)
```

Finally we create a manager agent, and upon initialization we pass our managed agent to it in its `managed_agents` argument.

Since this agent is the one tasked with the planning and thinking, advanced reasoning will be beneficial, so a `ReactCodeAgent` will be the best choice.

Also, we want to ask a question that involves the current year: so let us add `additional_authorized_imports=["time", "datetime"]`


```python
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_web_agent],
    additional_authorized_imports=["time", "datetime"],
)
```

That's all! Now let's run our system! We select a question that requires some calculation and 


```python
manager_agent.run("How many years ago was Stripe founded?")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #d4b702; text-decoration-color: #d4b702">╭──────────────────────────────────────────────────── </span><span style="color: #d4b702; text-decoration-color: #d4b702; font-weight: bold">New run</span><span style="color: #d4b702; text-decoration-color: #d4b702"> ────────────────────────────────────────────────────╮</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span>                                                                                                                 <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">How many years ago was Stripe founded?</span>                                                                          <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span>                                                                                                                 <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">╰─ InferenceClientModel - Qwen/Qwen2.5-72B-Instruct ────────────────────────────────────────────────────────────────────────╯</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #d4b702; text-decoration-color: #d4b702">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ </span><span style="font-weight: bold">Step </span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span><span style="color: #d4b702; text-decoration-color: #d4b702"> ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">╭─ <span style="font-weight: bold">Executing this code:</span> ──────────────────────────────────────────────────────────────────────────────────────────╮
│ <span style="color: #e3e3dd; text-decoration-color: #e3e3dd; background-color: #272822; font-weight: bold">  </span><span style="color: #656660; text-decoration-color: #656660; background-color: #272822">1 </span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">found_year </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> search_agent(request</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"When was Stripe founded?"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">)</span><span style="background-color: #272822">                                              </span> │
│ <span style="color: #e3e3dd; text-decoration-color: #e3e3dd; background-color: #272822; font-weight: bold">  </span><span style="color: #656660; text-decoration-color: #656660; background-color: #272822">2 </span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">print(found_year)</span><span style="background-color: #272822">                                                                                          </span> │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #d4b702; text-decoration-color: #d4b702">╭──────────────────────────────────────────────────── </span><span style="color: #d4b702; text-decoration-color: #d4b702; font-weight: bold">New run</span><span style="color: #d4b702; text-decoration-color: #d4b702"> ────────────────────────────────────────────────────╮</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span>                                                                                                                 <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">You're a helpful agent named 'search_agent'.</span>                                                                    <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">You have been submitted this task by your manager.</span>                                                              <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">---</span>                                                                                                             <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">Task:</span>                                                                                                           <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">When was Stripe founded?</span>                                                                                        <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">---</span>                                                                                                             <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">You're helping your manager solve a wider task: so make sure to not provide a one-line answer, but give as much</span> <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">information as possible to give them a clear understanding of the answer.</span>                                       <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span>                                                                                                                 <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">Your final_answer WILL HAVE to contain these parts:</span>                                                             <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">### 1. Task outcome (short version):</span>                                                                            <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">### 2. Task outcome (extremely detailed version):</span>                                                               <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">### 3. Additional context (if relevant):</span>                                                                        <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span>                                                                                                                 <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">Put all these in your final_answer tool, everything that you do not pass as an argument to final_answer will be</span> <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">lost.</span>                                                                                                           <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">And even if your task resolution is not successful, please return as much context as possible, so that your </span>    <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">manager can act upon this feedback.</span>                                                                             <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span> <span style="font-weight: bold">{additional_prompting}</span>                                                                                          <span style="color: #d4b702; text-decoration-color: #d4b702">│</span>
<span style="color: #d4b702; text-decoration-color: #d4b702">│</span>                                                                                                                 <span style="color: #d4b702; text-decoration-color: