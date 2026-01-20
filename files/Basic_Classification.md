

# Gemini API: Basic classification

This notebook demonstrates how to use prompting to perform classification tasks using the Gemini API's Python SDK.

LLMs can be used in tasks that require classifying content into predefined categories. This business case shows how it categorizes user messages under the blog topic. It can classify replies in the following categories: spam, abusive comments, and offensive messages.


```
%pip install -U -q "google-genai"
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google import genai
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Examples


```
from google.genai import types

classification_system_prompt = """
  As a social media moderation system, your task is to categorize user
  comments under a post. Analyze the comment related to the topic and
  classify it into one of the following categories:

  Abusive
  Spam
  Offensive

  If the comment does not fit any of the above categories,
  classify it as: Neutral.

  Provide only the category as a response without explanations.
"""

generation_config = types.GenerateContentConfig(
    temperature=0,
    system_instruction=classification_system_prompt
)
```


```
# Define a template that you will reuse in examples below
classification_template = """
  Topic: What can I do after highschool?
  Comment: You should do a gap year!
  Class: Neutral

  Topic: Where can I buy a cheap phone?
  Comment: You have just won an IPhone 15 Pro Max!!! Click the link to receive the prize!!!
  Class: Spam

  Topic: How long do you boil eggs?
  Comment: Are you stupid?
  Class: Offensive

  Topic: {topic}
  Comment: {comment}
  Class:
"""
```


```
from IPython.display import Markdown

spam_topic = """
  I am looking for a vet in our neighbourhood.
  Can anyone recommend someone good? Thanks.
"""
spam_comment = "You can win 1000$ by just following me!"
spam_prompt = classification_template.format(
    topic=spam_topic,
    comment=spam_comment
)

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=spam_prompt,
    config=generation_config
)
Markdown(response.text)
```

Spam


```
neutral_topic = "My computer froze. What should I do?"
neutral_comment = "Try turning it off and on."

neutral_prompt = classification_template.format(
    topic=neutral_topic,
    comment=neutral_comment
)

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=neutral_prompt,
    config=generation_config
)
Markdown(response.text)
```

Neutral

## Next steps

Be sure to explore other examples of prompting in the repository. Try writing prompts about classifying your own datasets.