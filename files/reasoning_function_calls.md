# Managing Function Calls With Reasoning Models
OpenAI now offers function calling using [reasoning models](https://platform.openai.com/docs/guides/reasoning?api-mode=responses). Reasoning models are trained to follow logical chains of thought, making them better suited for complex or multi-step tasks.
> _Reasoning models like o3 and o4-mini are LLMs trained with reinforcement learning to perform reasoning. Reasoning models think before they answer, producing a long internal chain of thought before responding to the user. Reasoning models excel in complex problem solving, coding, scientific reasoning, and multi-step planning for agentic workflows. They're also the best models for Codex CLI, our lightweight coding agent._

For the most part, using these models via the API is very simple and comparable to using familiar 'chat' models. 

However, there are some nuances to bear in mind, particularly when it comes to using features such as function calling. 

All examples in this notebook use the newer [Responses API](https://community.openai.com/t/introducing-the-responses-api/1140929) which provides convenient abstractions for managing conversation state. However the principles here are relevant when using the older chat completions API.

## Making API calls to reasoning models


```python
# pip install openai
# Import libraries 
import json
from openai import OpenAI
from uuid import uuid4
from typing import Callable

client = OpenAI()
MODEL_DEFAULTS = {
    "model": "o4-mini", # 200,000 token context window
    "reasoning": {"effort": "low", "summary": "auto"}, # Automatically summarise the reasoning process. Can also choose "detailed" or "none"
}
```

Let's make a simple call to a reasoning model using the Responses API.
We specify a low reasoning effort and retrieve the response with the helpful `output_text` attribute.
We can ask follow up questions and use the `previous_response_id` to let OpenAI manage the conversation history automatically


```python
response = client.responses.create(
    input="Which of the last four Olympic host cities has the highest average temperature?",
    **MODEL_DEFAULTS
)
print(response.output_text)

response = client.responses.create(
    input="what about the lowest?",
    previous_response_id=response.id,
    **MODEL_DEFAULTS
)
print(response.output_text)
```

    Among the last four Summer Olympic host cities—Tokyo (2020), Rio de Janeiro (2016), London (2012) and Beijing (2008)—Rio de Janeiro has by far the warmest climate. Average annual temperatures are roughly:
    
    • Rio de Janeiro: ≈ 23 °C  
    • Tokyo: ≈ 16 °C  
    • Beijing: ≈ 13 °C  
    • London: ≈ 11 °C  
    
    So Rio de Janeiro has the highest average temperature.
    Among those four, London has the lowest average annual temperature, at about 11 °C.


Nice and easy!

We're asking relatively complex questions that may require the model to reason out a plan and proceed through it in steps, but this reasoning is hidden from us - we simply wait a little longer before being shown the response. 

However, if we inspect the output we can see that the model has made use of a hidden set of 'reasoning' tokens that were included in the model context window, but not exposed to us as end users.
We can see these tokens and a summary of the reasoning (but not the literal tokens used) in the response.


```python
print(next(rx for rx in response.output if rx.type == 'reasoning').summary[0].text)
response.usage.to_dict()
```

    **Determining lowest temperatures**
    
    The user is asking about the lowest average temperatures of the last four Olympic host cities: Tokyo, Rio, London, and Beijing. I see London has the lowest average temperature at around 11°C. If I double-check the annual averages: Rio is about 23°C, Tokyo is around 16°C, and Beijing is approximately 13°C. So, my final answer is London with an average of roughly 11°C. I could provide those approximate values clearly for the user.





    {'input_tokens': 136,
     'input_tokens_details': {'cached_tokens': 0},
     'output_tokens': 89,
     'output_tokens_details': {'reasoning_tokens': 64},
     'total_tokens': 225}



It is important to know about these reasoning tokens, because it means we will consume our available context window more quickly than with traditional chat models.

## Calling custom functions
What happens if we ask the model a complex request that also requires the use of custom tools?
* Let's imagine we have more questions about Olympic Cities, but we also have an internal database that contains IDs for each city.
* It's possible that the model will need to invoke our tool partway through its reasoning process before returning a result.
* Let's make a function that produces a random UUID and ask the model to reason about these UUIDs. 



```python

def get_city_uuid(city: str) -> str:
    """Just a fake tool to return a fake UUID"""
    uuid = str(uuid4())
    return f"{city} ID: {uuid}"

# The tool schema that we will pass to the model
tools = [
    {
        "type": "function",
        "name": "get_city_uuid",
        "description": "Retrieve the internal ID for a city from the internal database. Only invoke this function if the user needs to know the internal ID for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The name of the city to get information about"}
            },
            "required": ["city"]
        }
    }
]

# This is a general practice - we need a mapping of the tool names we tell the model about, and the functions that implement them.
tool_mapping = {
    "get_city_uuid": get_city_uuid
}

# Let's add this to our defaults so we don't have to pass it every time
MODEL_DEFAULTS["tools"] = tools

response = client.responses.create(
    input="What's the internal ID for the lowest-temperature city?",
    previous_response_id=response.id,
    **MODEL_DEFAULTS)
print(response.output_text)

```

    


We didn't get an `output_text` this time. Let's look at the response output


```python
response.output
```




    [ResponseReasoningItem(id='rs_68246219e8288191af051173b1d53b3f0c4fbdb0d4a46f3c', summary=[], type='reasoning', status=None),
     ResponseFunctionToolCall(arguments='{"city":"London"}', call_id='call_Mx6pyTjCkSkmASETsVASogoC', name='get_city_uuid', type='function_call', id='fc_6824621b8f6c8191a8095df7230b611e0c4fbdb0d4a46f3c', status='completed')]



Along with the reasoning step, the model has successfully identified the need for a tool call and passed back instructions to send to our function call. 

Let's invoke the function and send the results to the model so it can continue reasoning.
Function responses are a special kind of message, so we need to structure our next message as a special kind of input:
```json
{
    "type": "function_call_output",
    "call_id": function_call.call_id,
    "output": tool_output
}
```


```python
# Extract the function call(s) from the response
new_conversation_items = []
function_calls = [rx for rx in response.output if rx.type == 'function_call']
for function_call in function_calls:
    target_tool = tool_mapping.get(function_call.name)
    if not target_tool:
        raise ValueError(f"No tool found for function call: {function_call.name}")
    arguments = json.loads(function_call.arguments) # Load the arguments as a dictionary
    tool_output = target_tool(**arguments) # Invoke the tool with the arguments
    new_conversation_items.append({
        "type": "function_call_output",
        "call_id": function_call.call_id, # We map the response back to the original function call
        "output": tool_output
    })
```


```python
response = client.responses.create(
    input=new_conversation_items,
    previous_response_id=response.id,
    **MODEL_DEFAULTS
)
print(response.output_text)

```

    The internal ID for London is 816bed76-b956-46c4-94ec-51d30b022725.


This works great here - as we know that a single function call is all that is required for the model to respond - but we also need to account for situations where multiple tool calls might need to be executed for the reasoning to complete.

Let's add a second call to run a web search.

OpenAI's web search tool is not available out of the box with reasoning models (as of May 2025 - this may soon change) but it's not too hard to create a custom web search function using 4o mini or another web search enabled model.


```python
def web_search(query: str) -> str:
    """Search the web for information and return back a summary of the results"""
    result = client.responses.create(
        model="gpt-4o-mini",
        input=f"Search the web for '{query}' and reply with only the result.",
        tools=[{"type": "web_search_preview"}],
    )
    return result.output_text

tools.append({
        "type": "function",
        "name": "web_search",
        "description": "Search the web for information and return back a summary of the results",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The query to search the web for."}
            },
            "required": ["query"]
        }
    })
tool_mapping["web_search"] = web_search

```

## Executing multiple functions in series

Some OpenAI models support the parameter `parallel_tool_calls` which allows the model to return an array of functions which we can then execute in parallel. However, reasoning models may produce a sequence of function calls that must be made in series, particularly as some steps may depend on the results of previous ones.
As such, we ought to define a general pattern which we can use to handle arbitrarily complex reasoning workflows:
* At each step in the conversation, initialise a loop
* If the response contains function calls, we must assume the reasoning is ongoing and we should feed the function results (and any intermediate reasoning) back into the model for further inference
* If there are no function calls and we instead receive a Reponse.output with a type of 'message', we can safely assume the agent has finished reasoning and we can break out of the loop


```python
# Let's wrap our logic above into a function which we can use to invoke tool calls.
def invoke_functions_from_response(response,
                                   tool_mapping: dict[str, Callable] = tool_mapping
                                   ) -> list[dict]:
    """Extract all function calls from the response, look up the corresponding tool function(s) and execute them.
    (This would be a good place to handle asynchroneous tool calls, or ones that take a while to execute.)
    This returns a list of messages to be added to the conversation history.
    """
    intermediate_messages = []
    for response_item in response.output:
        if response_item.type == 'function_call':
            target_tool = tool_mapping.get(response_item.name)
            if target_tool:
                try:
                    arguments = json.loads(response_item.arguments)
                    print(f"Invoking tool: {response_item.name}({arguments})")
                    tool_output = target_tool(**arguments)
                except Exception as e:
                    msg = f"Error executing function call: {response_item.name}: {e}"
                    tool_output = msg
                    print(msg)
            else:
                msg = f"ERROR - No tool registered for function call: {response_item.name}"
                tool_output = msg
                print(msg)
            intermediate_messages.append({
                "type": "function_call_output",
                "call_id": response_item.call_id,
                "output": tool_output
            })
        elif response_item.type == 'reasoning':
            print(f'Reasoning step: {response_item.summary}')
    return intermediate_messages
```

Now let's demonstrate the loop concept we discussed before.


```python
initial_question = (
    "What are the internal IDs for the cities that have hosted the Olympics in the last 20 years, "
    "and which of those cities have recent news stories (in 2025) about the Olympics? "
    "Use your internal tools to look up the IDs and the web search tool to find the news stories."
)

# We fetch a response and then kick off a loop to handle the response
response = client.responses.create(
    input=initial_question,
    **MODEL_DEFAULTS,
)
while True:   
    function_responses = invoke_functions_from_response(response)
    if len(function_responses) == 0: # We're done reasoning
        print(response.output_text)
        break
    else:
        print("More reasoning required, continuing...")
        response = client.responses.create(
            input=function_responses,
            previous_response_id=response.id,
            **MODEL_DEFAULTS
        )
```

    [Reasoning step: [], ..., Reasoning step: [Summary(text='**Focusing on Olympic News**\n\nI need to clarify that the Invictus Games are not related to the Olympics, so I should exclude them from my search. That leaves me with Olympic-specific news focusing on Paris. I also want to consider past events, like Sochi and Pyeongchang, so I think it makes sense to search for news related to Sochi as well. Let’s focus on gathering relevant Olympic updates to keep things organized.', type='summary_text')]]
    [Invoking tool: get_city_uuid({'city': 'Beijing'}), ..., Invoking tool: web_search({'query': '2025 Pyeongchang Olympics news'})]
    More reasoning required, continuing...
    Here are the internal IDs for all cities that have hosted Olympic Games in the last 20 years (2005–2025), along with those cities that have notable 2025 news stories specifically about the Olympics:
    
    1. Beijing (2008 Summer; 2022 Winter)  
       • UUID: 5b058554-7253-4d9d-a434-5d4ccc87c78b  
       • 2025 Olympic News? No major Olympic-specific news in 2025
    
    2. London (2012 Summer)  
       • UUID: 9a67392d-c319-4598-b69a-adc5ffdaaba2  
       • 2025 Olympic News? No
    
    3. Rio de Janeiro (2016 Summer)  
       • UUID: ad5eaaae-b280-4c1d-9360-3a38b0c348c3  
       • 2025 Olympic News? No
    
    4. Tokyo (2020 Summer)  
       • UUID: 66c3a62a-840c-417a-8fad-ce87b97bb6a3  
       • 2025 Olympic News? No
    
    5. Paris (2024 Summer)  
       • UUID: a2da124e-3fad-402b-8ccf-173f63b4ff68  
       • 2025 Olympic News? Yes  
         – Olympic cauldron balloon to float annually over Paris into 2028 ([AP News])  
         – IOC to replace defective Paris 2024 medals ([NDTV Sports])  
         – IOC elects Kirsty Coventry as president at March 2025 session ([Wikipedia])  
         – MLB cancels its planned 2025 Paris regular-season games ([AP News])
    
    6. Turin (2006 Winter)  
       • UUID: 3674750b-6b76-49dc-adf4-d4393fa7bcfa  
       • 2025 Olympic News? No (Host of Special Olympics World Winter Games, but not mainline Olympics)
    
    7. Vancouver (2010 Winter)  
       • UUID: 22517787-5915-41c8-b9dd-a19aa2953210  
       • 2025 Olympic News? No
    
    8. Sochi (2014 Winter)  
       • UUID: f7efa267-c7da-4cdc-a14f-a4844f47b888  
       • 2025 Olympic News? No
    
    9. Pyeongchang (2018 Winter)  
       • UUID: ffb19c03-5212-42a9-a527-315d35efc5fc  
       • 2025 Olympic News? No
    
    Summary of cities with 2025 Olympic-related news:  
    • Paris (a2da124e-3fad-402b-8ccf-173f63b4ff68)


## Manual conversation orchestration
So far so good! It's really cool to watch the model pause execution to run a function before continuing. 
In practice the example above is quite trivial, and production use cases may be much more complex:
* Our context window may grow too large and we may wish to prune older and less relevant messages, or summarize the conversation so far
* We may wish to allow users to navigate back and forth through the conversation and re-generate answers
* We may wish to store messages in our own database for audit purposes rather than relying on OpenAI's storage and orchestration
* etc.

In these situations we may wish to take full control of the conversation. Rather than using `previous_message_id` we can instead treat the API as 'stateless' and make and maintain an array of conversation items that we send to the model as input each time.

This poses some Reasoning model specific nuances to consider. 
* In particular, it is essential that we preserve any reasoning and function call responses in our conversation history.
* This is how the model keeps track of what chain-of-thought steps it has run through. The API will error if these are not included.

Let's run through the example above again, orchestrating the messages ourselves and tracking token usage.

---
*Note that the code below is structured for readibility - in practice you may wish to consider a more sophisticated workflow to handle edge cases*


```python
# Let's initialise our conversation with the first user message
total_tokens_used = 0
user_messages = [
    (
        "Of those cities that have hosted the summer Olympic games in the last 20 years - "
        "do any of them have IDs beginning with a number and a temperate climate? "
        "Use your available tools to look up the IDs for each city and make sure to search the web to find out about the climate."
    ),
    "Great thanks