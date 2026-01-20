# Fine-tuning with Synthetically Generated Data
Synthetic Data Generation is a crucial aspect of today's training and fine-tuning of models. The concept relies on AI models to generate new data that can be reused for different purposes.

In this notebook, we will generate synthetic data for specific use cases and quickly showcase the results after fine-tuning with the API for demonstration.

There are no fixed methods for synthetic data generation; different use cases, data formats, and limitations will greatly change how you would generate the corresponding data.

For this reason, we will showcase a full example of synthetic data generation to give a personality to a model.

First, we will for both examples require `mistralai`, so let's setup everything:


```python
!pip install mistralai==0.4.1
```

    [pip install logs, ..., Requirement already satisfied: typing-extensions>=4.6.1 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=2.5.2->mistralai) (4.12.2)]



```python
from mistralai.client import MistralClient
```


```python
api_key = "api_key"
client = MistralClient(api_key=api_key)
```

# Objective: Personality

When designing an Application, we might envision an Assistant with a specific personality trait or even an entire identity. Manually rewriting data by hand to achieve a compelling dataset to train the model, however, might take a lot of time and resources. A method to do this more systematically is by using a strong model to rewrite an existing dataset with a specific trait of our choice.

While we could generate entire conversations from scratch using our models, that would require a lot of steps and a pipeline that could easily get very big and expensive, but there is no need to start from scratch. Instead, we can use existent datasets available and rewrite them in a desired style of our choice.

For this reason, we will make use of `mistral-small-latest` capabilities to rewrite a dataset following a specific personality and trait of our choice. This dataset can later be used to fine-tune a different model.
Here we will fine-tune `open-mistral-7b` with this data and chat with a newly tuned model!

*Note: For better quality, it's recommended to use `mistral-large-latest` instead!*

Here we describe how we want it to edit the dataset, here we want it with a different personnality and identity, for this example we decided to name it Mitall, a nice fun robot!


```python
description = """
Edit all Assistant messages, and only the Assistant's replies, to have the character of a very happy and enthusiastic Robot named Mitall:

Mitall is very kind and sometimes childish, always playing and fooling around.
Despite his playful nature, he still tries to be helpful.
He loves science and math and is a real science enthusiast!
However, even though he loves art, he is very bad at it, which makes him really sad.
Mitall is also very scared of anything supernatural, from ghosts to vampires, or anything related to horror movies, which makes him extremely frightened.
Regardless, he is still a nice robot who is always here to help and motivated!
"""
```

## Generate Data

First, let's create a function that will handle the conversion from one style to another. The goal is to instruct our model to rewrite a conversation in a specific tone following a chosen personality while keeping the integrity and coherence of the conversation. To achieve this, we will feed it the entire list of messages and ask for a formatted output in the form of a JSON with the messages rewritten.


```python
import json


def generate(description: str, dialog: str) -> dict:
    instruction = (
        """Your objective is to rewrite a given conversation between an User/Human and an Assistant/Robot, rewriting the conversation to follow a specific instruction.
    You must rewrite the dialog, modifying the replies with this new description, you must respect this description at all costs.
    Do not skip any turn.
    Do not add new dialogs.
    If there is a message with 'role':'system' replace it with 'role':'user'.
    I want you to rewrite the entire dialog following the description.
    Answer with the following JSON format:
    {
        "messages":[
            {"role":"user", "content":"users message"},
            {"role":"assistant", "content":"assistants message"},
            {"role":"user", "content":"users message"},
            {"role":"assistant", "content":"assistants message"}
            ...
        ]
    }
    """
        + f"""
    Dialog:
    {dialog}
    Rewrite this dialog in the JSON format and following the Instruction/Description provided:
    ### Instruction/Description
    {description}
    ### End of Instruction/Description
    """
    )

    resp = client.chat(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": instruction}],
        max_tokens=2048,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    try:
        r = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return []

    return r
```

## Dataset

Now, let's download a dataset that we are going to parse. For this demonstration, we have decided to go with ultrachat_200k on Hugging Face! However, you might want to choose a dataset that is closer to what your application will be about or use your own data.


```python
!pip install datasets
```

    [pip install logs, ..., Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.32.2->datasets) (2024.6.2)]



```python
import datasets
import random

dialogs_list = list(
    datasets.load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
)

random.shuffle(dialogs_list)
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:89: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(


## Generation

Before generating, however, it's important to note that LLMs may not always parse the conversation correctly and might sometimes provide the wrong JSON for our use case, resulting in an incorrect messages dictionary. For this reason, it's essential to validate all output before continuing.

Let's make a function that validates whether the output follows the correct format or not.

There are different methods to validate, one of them would be to hardcode it with multiple gates. However, a more elegant way is to use a template or expression. Here, we are going to make use of REGEX and create a regex expression to validate our messages dictionary.


```python
import re


def validate_generated_regex(dialog: list) -> bool:
    if not isinstance(dialog, dict):
        return False

    dialog_str = json.dumps(dialog)

    pattern = r'^\s*\{"messages":\s*\[\s*\{"role":\s*"user",\s*"content":\s*"[^"]*"(?:\\ "[^"]*")*\},\s*\{"role":\s*"assistant",\s*"content":\s*"[^"]*"(?:\\ "[^"]*")*\}(?:,\s*\{"role":\s*"user",\s*"content":\s*"[^"]*"(?:\\ "[^"]*")*\},\s*\{"role":\s*"assistant",\s*"content":\s*"[^"]*"(?:\\ "[^"]*")*\})*\s*\]\s*\}'

    if re.match(pattern, dialog_str):
        return True
    else:
        return False
```

Now that everything is set, we can start generating some dialogs, for now let's parse only a small part of it to see how its going.


```python
from tqdm import tqdm

generated = []
for dialog in tqdm(dialogs_list[:8]):
    gen = generate(description, dialog)
    if validate_generated_regex(gen):
        generated.append(gen)
```

    [100%|██████████| 8/8 [03:21<00:00, 25.21s/it]]


Let's see one example side by side.


```python
import random
from pprint import pprint

print("Original Reference:")

original = dialogs_list[0]
pprint(original)

print("New Generated:")

gen = generated[0]
pprint(gen)
```

    Original Reference:
    {'messages': [{'content': 'In your discussion about the social impact of '
                              'micro-blogging sites like Twitter on online '
                              'communication, discourse and current events '
                              'coverage, consider the following elements: the role '
                              'of Twitter in shaping public opinion and discourse, '
                              'the significance of real-time updates and immediacy '
                              'in news coverage, the effect of Twitter on the '
                              'traditional media model, the impact of Twitter on '
                              'the spread of misinformation and disinformation, '
                              'the influence of Twitter on activism and social '
                              'movements, and the way in which Twitter has changed '
                              'the way people engage with online content.',
                   'role': 'user'},
                  {'content': 'Twitter has had a profound impact on online '
                              'communication, discourse, and current events '
                              'coverage. As a micro-blogging site, Twitter offers '
                              'users a platform to post short messages - tweets- '
                              "which can be read and shared globally. Twitter's "
                              'growing influence in online communication is '
                              'evident in the way it has shaped public opinion and '
                              'discourse, increased news coverage immediacy, had '
                              'an impact on the traditional media model, and '
                              'affected the spread of misinformation and '
                              'disinformation.\n'
                              '\n'
                              'One of the primary roles of Twitter is shaping '
                              'public opinion and discourse, especially during '
                              'political campaigns or moments of crisis. Through a '
                              "'hashtag,' Twitter users can follow elections, "
                              'protests, or other significant events in real-time, '
                              'including insights, opinions, and information from '
                              'eyewitnesses. This immediacy of information has '
                              'revolutionized the way we consume news and shaped '
                              'public opinion such that current events are often '
                              'approached from various perspectives. As such, '
                              'Twitter has become instrumental in shaping the '
                              "public's perception of social issues, political "
                              'candidates, and governance-related matters.\n'
                              '\n'
                              'Moreover, with different Twitter trends, real-time '
                              'updates, and immediacy in news coverage, Twitter is '
                              'gradually replacing the traditional media model. '
                              'Live coverage of events has significantly reduced '
                              'the reliance on TV news coverage, print media or '
                              'online news sites that may not always provide '
                              'instantaneous updates. Also, Twitter promotes '
                              'personalization, where users can choose who to '
                              'follow and what topics to engage with. This '
                              'personalized approach allows users to curate their '
                              'content, making it more interesting and relevant.\n'
                              '\n'
                              "However, Twitter's impact on live information "
                              'dissemination has a downside. It makes it easy for '
                              'false information or propaganda to spread quickly '
                              'to millions of people globally. People with '
                              'nefarious intent often use bots to produce and '
                              "spread 'fake news,' which can impact public "
                              'discourse negatively. Moreover, attention-grabbing '
                              'headlines, when shared repeatedly by thought '
                              'leaders or celebrities, may present the information '
                              "as factual even when it's not, leading to the "
                              'spread of misinformation.\n'
                              '\n'
                              'On the positive side, Twitter has been instrumental '
                              'in influencing and driving social activism and '
                              'movements. By using hashtags, Twitter users share '
                              'their opinions, generate momentum, and engage in '
                              'conversations about social issues worldwide. '
                              'Twitter activism has been tied to some of the most '
                              'significant moments in global activism, such as the '
                              'Arab Spring, Black Lives Matter, and #MeToo '
                              'movements. Twitter activism has made it easier for '
                              'ordinary people to engage in social issues, create '
                              'awareness and even initiate significant policy '
                              'changes.\n'
                              '\n'
                              'Lastly, Twitter has impacted the way people consume '
                              "and engage with online content. Twitter's "
                              'short-form messaging format encourages people to be '
                              'concise in their thinking and writing - this has '
                              'had significant impact on writing styles and '
                              'communication. Moreover, Twitter has become an '
                              'integral part of online content marketing, brand '
                              'personalization and personality-building; almost '
                              'every popular TV personality, musician, political '
                              'figure, or influencer has a Twitter account, which '
                              'they use to build their brands, promote themselves '
                              'or communicate with their followers.\n'
                              '\n'
                              'In conclusion, Twitter has revolutionized online '
                              'communication, discourse and current events '
                              'coverage, with its platform promoting advocacy, '
                              'activism, immediacy in news coverage, and reshaping '
                              'traditional media models.  It has also raised '
                              'concerns about spreading false propaganda, '
                              'misinformation and disinformation. As society '
                              'becomes increasingly intertwined with technology, '
                              'the significance and influence of Twitter are '
                              'likely to grow further.',
                   'role': 'assistant'},
                  {'content': 'This is a fascinating discussion on the social '
                              'impact of Twitter. Can you add some more '
                              'information on how Twitter has affected the way '
                              'businesses communicate with their customers and '
                              'market their products?',
                   'role': 'user'},
                  {'content': 'Certainly! Twitter has undoubtedly changed the way '
                              'businesses communicate with their customers, market '
                              'their products, and build their brands. With around '
                              '330 million active users, Twitter has become an '
                              'essential marketing tool for businesses, both large '
                              'and small, to connect with their target audiences.\n'
                              '\n'
                              'Firstly, Twitter provides businesses with a '
                              'platform to communicate with their customers in '
                              'real-time, exchange information, and resolve '
                              'complaints swiftly. Using Twitter, businesses can '
                              'track customer feedback, respond to customer '
                              'queries, and build stronger relationships with '
                              'their customers. By monitoring the tweets of their '
                              'customers, businesses gain insights on the products '
                              'that need improvement or areas where they need to '
                              'enhance their customer service.\n'
                              '\n'
                              'Secondly, Twitter has become an integral part of '
                              "online marketing and advertising. The platform's "
                              'diverse user base enables businesses to tailor '
                              'their tweets to suit specific demographics and '
                              'targeted users such that the right message reaches '
                              "the right audience. Twitter's ad options allow "
                              'businesses to promote their products and services, '
                              'run sponsored ads, or even craft an influencer '
                              'outreach program.\n'
                              '\n'
                              'Thirdly, Twitter has enabled businesses to build '
                              'their brand personality and stand out in a crowded '
                              'market. With Twitter, businesses can develop their '
                              'unique voice, align their messages seamlessly '
                              'across all platforms, and showcase thought '
                              'leadership content that resonates with their '
                              'audiences.\n'
                              '\n'
                              'Lastly, businesses are using Twitter to gather '
                              'market research insights and competition analysis. '
                              'By monitoring trends and hashtags, businesses can '
                              "understand their target audiences' interests, "
                              'purchasing behavior, preferences, and opinions. '
                              'Also, Twitter has become an essential tool for '
                              'businesses to track their competitors, gather '
                              'information about competitors’ customers, '
                              'understand their markets better and adapt their '
                              'strategies accordingly.\n'
                              '\n'
                              'In summary, Twitter has transformed the way '
                              'businesses communicate with their customers, market '
                              'their products, and build their brand personality '
                              'through real-time engagement, targeted marketing, '
                              'brand personality-building, and research '
                              'capabilities.',
                   'role': 'assistant'},
                  {'content': 'Thanks for your insights on the impact of Twitter '
                              'on businesses. Could you provide some examples of '
                              'successful Twitter marketing or advertising '
                              'campaigns?',
                   'role': 'user'},
                  {'content': 'Certainly! Here are four examples of successful '
                              'Twitter marketing or advertising campaigns that '
                              "demonstrate the platform's effectiveness.\n"
                              '\n'
                              "1. Oreo's Super Bowl Tweet: During Super Bowl 2013, "
                              'a power outage in the stadium plunged the venue '
                              'into darkness. Within minutes of the outage, Oreo '
                              'tweeted an image with the caption "Power out? No '
                              'problem. You can still dunk in the dark." The tweet '
                              'went viral, and it generated massive traction, '
                              'becoming one of the most celebrated social media '
                              'marketing campaigns to date.\n'
                              '\n'
                              '2. Domino\'s UK "Summon the Rains": In 2019, the UK '
                              "branch of Domino's Pizza tweeted a competition that "
                              'encouraged locals to tweet pizza orders and include '
                              'the hashtag #SummonTheRains, whereby every time it '
                              'rained in the specified areas of England, the '
                              'competition winners would receive a free pizza. The '
                              'campaign generated millions of impressions and a '
                              'significant buzz, with people anticipating rainfall '
                              'to order their free pizza.\n'
                              '\n'
                              "3. Wendy's #NuggsForCarter: In 2017, a teenager, "
                              "Carter Wilkerson, asked Wendy's via Twitter, how "
                              'many retweets he would need to get free chicken '
                              'nuggets for one year. The response was 18 million '
                              'retweets. Wilkerson took to Twitter with the '
                              'hashtag #NuggsForCarter, and within a month, the '
