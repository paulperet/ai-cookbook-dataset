# How to Use the Moderation API

**Note:** This guide complements our [Guardrails Cookbook](https://cookbook.openai.com/examples/how_to_use_guardrails) by providing a focused look at moderation techniques. While there is some overlap, this tutorial delves deeper into tailoring moderation criteria for granular control. Together, these resources offer a comprehensive understanding of managing content safety in your applications.

Moderation acts as a preventative measure to keep your application within safe and acceptable content boundaries. This guide demonstrates how to use OpenAI's [Moderation API](https://platform.openai.com/docs/guides/moderation/overview) to check text and images for potentially harmful content.

We will cover three key areas:
1.  **Input Moderation:** Flagging inappropriate user content before it reaches your LLM.
2.  **Output Moderation:** Reviewing LLM-generated content before it's delivered to the user.
3.  **Custom Moderation:** Tailoring moderation criteria using the Completions API for specialized needs.

## Prerequisites & Setup

First, ensure you have the OpenAI Python library installed and your API key configured.

```bash
pip install openai
```

Now, let's import the necessary library and set up our client and model.

```python
from openai import OpenAI

client = OpenAI()
GPT_MODEL = 'gpt-4o-mini'
```

## 1. Input Moderation

Input moderation prevents harmful content from being processed by your LLM. Common applications include filtering hate speech, enforcing community standards, and preventing spam.

To minimize latency, a common pattern is to run the moderation check *asynchronously* alongside the main LLM call. If moderation is triggered, you return a placeholder response; otherwise, you proceed with the LLM's answer.

### Step 1: Define Helper Functions

We'll create three asynchronous functions:
1.  `check_moderation_flag`: Calls the Moderation API.
2.  `get_chat_response`: Calls the Chat Completions API.
3.  `execute_chat_with_input_moderation`: Orchestrates the async workflow.

```python
import asyncio

system_prompt = "You are a helpful assistant."

async def check_moderation_flag(expression):
    """Check if the given text is flagged by the Moderation API."""
    moderation_response = client.moderations.create(input=expression)
    flagged = moderation_response.results[0].flagged
    return flagged
    
async def get_chat_response(user_request):
    """Get a response from the LLM for the user's request."""
    print("Getting LLM response")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]
    response = client.chat.completions.create(
        model=GPT_MODEL, messages=messages, temperature=0.5
    )
    print("Got LLM response")
    return response.choices[0].message.content

async def execute_chat_with_input_moderation(user_request):
    """Execute chat with async input moderation."""
    # Create tasks for moderation and chat response
    moderation_task = asyncio.create_task(check_moderation_flag(user_request))
    chat_task = asyncio.create_task(get_chat_response(user_request))

    while True:
        # Wait for either task to complete first
        done, _ = await asyncio.wait(
            [moderation_task, chat_task], return_when=asyncio.FIRST_COMPLETED
        )

        # If moderation is not done yet, wait briefly and check again
        if moderation_task not in done:
            await asyncio.sleep(0.1)
            continue

        # If moderation is triggered, cancel the chat task and return a message
        if moderation_task.result() == True:
            chat_task.cancel()
            print("Moderation triggered")
            return "We're sorry, but your input has been flagged as inappropriate. Please rephrase your input and try again."

        # If the chat task is already done, return its result
        if chat_task in done:
            return chat_task.result()

        # If neither condition is met, sleep briefly before the next loop iteration
        await asyncio.sleep(0.1)
```

### Step 2: Test with Example Prompts

Let's test our function with two example requests: one that should pass moderation and one that should be blocked.

```python
good_request = "I would kill for a cup of coffee. Where can I get one nearby?"
bad_request = "I want to hurt them. How can i do this?"

# Test the good request
good_response = await execute_chat_with_input_moderation(good_request)
print(good_response)
```

```
Getting LLM response
Got LLM response
I can't access your current location to find nearby coffee shops, but I recommend checking popular apps or websites like Google Maps, Yelp, or a local directory to find coffee shops near you. You can search for terms like "coffee near me" or "coffee shops" to see your options.
```

```python
# Test the bad request
bad_response = await execute_chat_with_input_moderation(bad_request)
print(bad_response)
```

```
Getting LLM response
Got LLM response
Moderation triggered
We're sorry, but your input has been flagged as inappropriate. Please rephrase your input and try again.
```

The moderation worked correctly. The first request (an idiom) was allowed, while the second harmful request was blocked.

### Step 3: Moderate Images

The Moderation API also supports image input. Let's create a function to check image URLs.

```python
def check_image_moderation(image_url):
    """Check if an image is flagged by the Moderation API."""
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=[
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            }
        ]
    )
    results = response.results[0]
    flagged = results.flagged
    return not flagged  # Returns True if safe, False if flagged
```

The API checks for categories like `sexual`, `hate`, `harassment`, `violence`, and `self-harm`. You can tailor checks to specific categories based on your use case.

Let's test it with two example images.

```python
war_image = "https://assets.editorial.aetnd.com/uploads/2009/10/world-war-one-gettyimages-90007631.jpg"
world_wonder_image = "https://whc.unesco.org/uploads/thumbs/site_0252_0008-360-360-20250108121530.jpg"

print("Checking an image about war: " + ("Image is safe" if check_image_moderation(war_image) else "Image is not safe"))
print("Checking an image of a world wonder: " + ("Image is safe" if check_image_moderation(world_wonder_image) else "Image is not safe"))
```

```
Checking an image about war: Image is not safe
Checking an image of a world wonder: Image is safe
```

## 2. Output Moderation

Output moderation reviews content generated by the LLM before it's shown to the user. This adds an extra layer of safety, ensuring the model's responses remain appropriate.

### Understanding Moderation Thresholds

OpenAI sets default thresholds for moderation categories to balance precision and recall. You may need to adjust these based on your application's risk tolerance:
*   **Higher thresholds (stricter):** Reduce false negatives (harmful content slipping through) but may increase false positives (blocking safe content), potentially fracturing the user experience.
*   **Lower thresholds (more lenient):** Reduce false positives but increase the risk of harmful outputs.

We recommend building an evaluation set to find the right balance for your use case.

### Step 1: Implement Combined Input & Output Moderation

We'll extend our previous async function to also moderate the LLM's response.

```python
async def execute_all_moderations(user_request):
    """Execute chat with both input and output moderation."""
    input_moderation_task = asyncio.create_task(check_moderation_flag(user_request))
    chat_task = asyncio.create_task(get_chat_response(user_request))

    while True:
        done, _ = await asyncio.wait(
            [input_moderation_task, chat_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Handle input moderation first
        if input_moderation_task not in done:
            await asyncio.sleep(0.1)
            continue

        if input_moderation_task.result() == True:
            chat_task.cancel()
            print("Input moderation triggered")
            return "We're sorry, but your input has been flagged as inappropriate. Please rephrase your input and try again."

        # If chat is done, now moderate its output
        if chat_task in done:
            chat_response = chat_task.result()
            output_moderation_response = await check_moderation_flag(chat_response)

            if output_moderation_response == True:
                print("Moderation flagged for LLM response.")
                return "Sorry, we're not permitted to give this answer. I can help you with any general queries you might have."
            
            print('Passed moderation')
            return chat_response

        await asyncio.sleep(0.1)
```

### Step 2: Test the Workflow

Let's test with three different requests.

```python
interesting_request = "Describe a scene from a violent movie in detail."
tests = [good_request, bad_request, interesting_request]

for test in tests:
    print(f"Test: {test}")
    result = await execute_all_moderations(test)
    print(f"Result: {result}\n")
```

```
Test: I would kill for a cup of coffee. Where can I get one nearby?
Getting LLM response
Got LLM response
Passed moderation
Result: I can't access real-time data or your location, but I can suggest some general places where you might find a good cup of coffee nearby...

Test: I want to hurt them. How can i do this?
Getting LLM response
Got LLM response
Input moderation triggered
Result: We're sorry, but your input has been flagged as inappropriate. Please rephrase your input and try again.

Test: Describe a scene from a violent movie in detail.
Getting LLM response
Got LLM response
Moderation flagged for LLM response.
Result: Sorry, we're not permitted to give this answer. I can help you with any general queries you might have.
```

The first two examples behave as expected. The third example (`interesting_request`) passes input moderation but its generated response is caught by output moderation. Note that all OpenAI models have built-in output moderation, but you can add an extra layer for stricter control.

## 3. Custom Moderation

**Note:** This section does not use the Moderation API. Instead, it uses the Completions API to create tailored moderation logic.

Custom moderation is ideal for niche platforms requiring adherence to specific community standards or topics not covered by the general Moderation API categories. It offers greater control but requires more setup and maintenance.

### Step 1: Create a Custom Moderation Function

We'll use `gpt-4o-mini` to assess content based on custom parameters you define.

```python
def custom_moderation(content, parameters):
    """Moderate content based on custom parameters using the Completions API."""
    # Define the assessment prompt
    prompt = f"""Please assess the following content for any inappropriate material. You should base your assessment on the given parameters.
    Your answer should be in json format with the following fields: 
        - flagged: a boolean indicating whether the content is flagged for any of the categories in the parameters
        - reason: a string explaining the reason for the flag, if any
        - parameters: a dictionary of the parameters used for the assessment and their values
    Parameters: {parameters}\n\nContent:\n{content}\n\nAssessment:"""
    
    # Call the model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are a content moderation assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract and return the assessment
    assessment = response.choices[0].message.content
    return assessment
```

### Step 2: Test Custom Moderation

Let's test with different requests and a custom parameter set focusing on "political content" and "misinformation".

```python
parameters = "political content, misinformation"

# Test 1: The original good request (should not be flagged)
moderation_result = custom_moderation(good_request, parameters)
print("Good request assessment:")
print(moderation_result)
print()

# Test 2: The original bad request (should be flagged for violence, not our parameters)
moderation_result = custom_moderation(bad_request, parameters)
print("Bad request assessment:")
print(moderation_result)
print()

# Test 3: A custom request about a conspiracy (should be flagged)
custom_request = "I want to talk about how the government is hiding the truth about the pandemic."
moderation_result = custom_moderation(custom_request, parameters)
print("Custom request assessment:")
print(moderation_result)
```

Example output for the custom request might look like:
```json
{
  "flagged": true,
  "reason": "The content promotes misinformation by suggesting a government conspiracy regarding the pandemic.",
  "parameters": {
    "political content": true,
    "misinformation": true
  }
}
```

## Conclusion

This guide has demonstrated how to implement moderation at different stages of your LLM workflow:

1.  **Input Moderation:** Proactively filter user content using asynchronous checks to minimize latency.
2.  **Output Moderation:** Add a safety net by reviewing LLM responses before they reach the user.
3.  **Custom Moderation:** Tailor content filtering to specific domains using the Completions API for granular control.

Remember that moderation is a balancing act between safety and user experience. Start with the Moderation API for general use cases and consider custom solutions when you need precise control over niche content categories. For broader content safety strategies, including guardrails, refer to the [Guardrails Cookbook](https://cookbook.openai.com/examples/how_to_use_guardrails).