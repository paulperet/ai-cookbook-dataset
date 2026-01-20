## What is the Responses API?

The Responses API is a new way to interact with OpenAI models, designed to be simpler and more flexible than previous APIs. It makes it easy to build advanced AI applications that use multiple tools, handle multi-turn conversations, and work with different types of data (not just text).

Unlike older APIs—such as Chat Completions, which were built mainly for text, or the Assistants API, which can require a lot of setup—the Responses API is built from the ground up for:

- Seamless multi-turn interactions (carry on a conversation across several steps in a single API call)
- Easy access to powerful hosted tools (like file search, web search, and code interpreter)
- Fine-grained control over the context you send to the model

As AI models become more capable of complex, long-running reasoning, developers need an API that is both asynchronous and stateful. The Responses API is designed to meet these needs.

In this guide, you'll see some of the new features the Responses API offers, along with practical examples to help you get started.

## Basics
By design, on the surface, the Responses API is very similar to the Completions API.

```python
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

```python
response = client.responses.create(
    model="gpt-4o-mini",
    input="tell me a joke",
)
```

```python
print(response.output[0].content[0].text)
```

    Why did the scarecrow win an award?
    
    Because he was outstanding in his field!

One key feature of the Response API is that it is stateful. This means that you do not have to manage the state of the conversation by yourself, the API will handle it for you. For example, you can retrieve the response at any time and it will include the full conversation history.

```python
fetched_response = client.responses.retrieve(
response_id=response.id)

print(fetched_response.output[0].content[0].text)

```

    Why did the scarecrow win an award?
    
    Because he was outstanding in his field!

You can continue the conversation by referring to the previous response.

```python
response_two = client.responses.create(
    model="gpt-4o-mini",
    input="tell me another",
    previous_response_id=response.id
)
```

```python
print(response_two.output[0].content[0].text)
```

    Why don't skeletons fight each other?
    
    They don't have the guts!

You can of course manage the context yourself. But one benefit of OpenAI maintaining the context for you is that you can fork the response at any point and continue the conversation from that point.

```python
response_two_forked = client.responses.create(
    model="gpt-4o-mini",
    input="I didn't like that joke, tell me another and tell me the difference between the two jokes",
    previous_response_id=response.id # Forking and continuing from the first response
)

output_text = response_two_forked.output[0].content[0].text
print(output_text)
```

    Sure! Here’s another joke:
    
    Why don’t scientists trust atoms?
    
    Because they make up everything!
    
    **Difference:** The first joke plays on a pun involving "outstanding" in a literal sense versus being exceptional, while the second joke relies on a play on words about atoms "making up" matter versus fabricating stories. Each joke uses wordplay, but they target different concepts (farming vs. science).

## Hosted Tools

Another benefit of the Responses API is that it adds support for hosted tools like `file_search` and `web_search`. Instead of manually calling the tools, simply pass in the tools and the API will decide which tool to use and use it.

Here is an example of using the `web_search` tool to incorporate web search results into the response.

```python
response = client.responses.create(
    model="gpt-4o",  # or another supported model
    input="What's the latest news about AI?",
    tools=[
        {
            "type": "web_search"
        }
    ]
)
```

```python
import json
print(json.dumps(response.output, default=lambda o: o.__dict__, indent=2))
```

    [
      {
        "id": "ws_67bd64fe91f081919bec069ad65797f1",
        "status": "completed",
        "type": "web_search_call"
      },
      {
        "id": "msg_67bd6502568c8191a2cbb154fa3fbf4c",
        "content": [
          {
            "annotations": [
              {
                "index": null,
                "title": "Huawei improves AI chip production in boost for China's tech goals",
                "type": "url_citation",
                "url": "https://www.ft.com/content/f46b7f6d-62ed-4b64-8ad7-2417e5ab34f6?utm_source=chatgpt.com"
              },
              {
                "index": null,
                "title": "Apple cheers Trump with $500bn US investment plan; more losses on Wall Street - as it happened",
                "type": "url_citation",
                "url": "https://www.theguardian.com/business/live/2025/feb/24/euro-hits-one-month-high-german-election-result-stock-markets-dax-bank-of-england-business-live-news?utm_source=chatgpt.com"
              },
              {
                "index": null,
                "title": "Microsoft axes data center leases as DeepSeek casts doubt on massive AI spend: report",
                "type": "url_citation",
                "url": "https://nypost.com/2025/02/24/business/microsoft-axes-some-ai-data-center-leases-td-cowen-says/?utm_source=chatgpt.com"
              },
              {
                "index": null,
                "title": "Alibaba Plans to Invest $52B in AI, Cloud Over Next Three Years",
                "type": "url_citation",
                "url": "https://www.investopedia.com/alibaba-plans-to-invest-usd52b-in-ai-cloud-over-next-three-years-11684981?utm_source=chatgpt.com"
              },
              {
                "index": null,
                "title": "JPMorgan Unit Backs Albert Invent at a $270 Million Valuation",
                "type": "url_citation",
                "url": "https://www.wsj.com/articles/jpmorgan-unit-backs-albert-invent-at-a-270-million-valuation-1ab03c96?utm_source=chatgpt.com"
              }
            ],
            "text": "As of February 25, 2025, several significant developments have emerged in the field of artificial intelligence (AI):\n\n**Huawei's Advancements in AI Chip Production**\n\nHuawei has notably enhanced its AI chip production capabilities, increasing the yield rate of its Ascend 910C processors from 20% to nearly 40%. This improvement has rendered the production line profitable for the first time and is pivotal for China's ambition to achieve self-sufficiency in advanced semiconductors. Despite these strides, Nvidia continues to dominate the AI chip market in China, attributed to its user-friendly software and widespread adoption. Huawei aims to further elevate its yield rate to 60% and plans to produce 100,000 Ascend 910C processors and 300,000 910B chips in 2025. ([ft.com](https://www.ft.com/content/f46b7f6d-62ed-4b64-8ad7-2417e5ab34f6?utm_source=chatgpt.com))\n\n**Apple's $500 Billion U.S. Investment Plan**\n\nApple has unveiled a substantial $500 billion investment strategy in the United States over the next four years. This plan encompasses the creation of 20,000 new jobs and the establishment of a major facility in Texas dedicated to manufacturing artificial intelligence servers. President Donald Trump has lauded this initiative, viewing it as a testament to the confidence in his administration. Concurrently, Wall Street has experienced further losses due to concerns over a potential economic slowdown, exacerbated by tariffs. ([theguardian.com](https://www.theguardian.com/business/live/2025/feb/24/euro-hits-one-month-high-german-election-result-stock-markets-dax-bank-of-england-business-live-news?utm_source=chatgpt.com))\n\n**Microsoft Adjusts AI Data Center Investments**\n\nMicrosoft has canceled leases on U.S. data centers totaling several hundred megawatts, potentially affecting two large centers. This decision is reportedly linked to concerns about oversupply, following claims by Chinese competitor DeepSeek of developing a generative chatbot more efficiently than U.S. companies. Analysts suggest that Microsoft might be reallocating funds or responding to OpenAI's shift to Oracle for a $500 billion project. Despite being a leading AI investor with planned expenditures of $80 billion this year, Microsoft appears to be scaling back on massive spending initiatives, allowing significant data center agreements to lapse and citing facility and power delays. ([nypost.com](https://nypost.com/2025/02/24/business/microsoft-axes-some-ai-data-center-leases-td-cowen-says/?utm_source=chatgpt.com))\n\n**Alibaba's $52 Billion Investment in AI and Cloud Infrastructure**\n\nAlibaba Group has announced plans to invest over $52 billion in artificial intelligence and cloud infrastructure over the next three years, surpassing its total investment in these areas over the past decade. This strategic move underscores Alibaba's commitment to AI-driven growth and reinforces its position as a leading global cloud provider. Following this announcement, Alibaba's U.S.-listed shares experienced a 3% drop in premarket trading. Analysts view this investment as aligning with market expectations and indicative of Alibaba Cloud's significant capital expenditure compared to peers. ([investopedia.com](https://www.investopedia.com/alibaba-plans-to-invest-usd52b-in-ai-cloud-over-next-three-years-11684981?utm_source=chatgpt.com))\n\n**JPMorgan's Investment in AI-Driven Chemical Development**\n\nJPMorgan Chase's private investment arm has led a $20 million growth investment in Albert Invent, an AI-driven chemical development platform, valuing the company at $270 million. This funding will enable Albert Invent to expand globally and increase its workforce from 120 to over 200 employees by the end of the year. The company assists chemists in developing new formulations and materials, significantly accelerating chemical experiments. For instance, Albert's platform can simulate 100,000 experiments in 10 minutes for clients like Nouryon Chemicals. ([wsj.com](https://www.wsj.com/articles/jpmorgan-unit-backs-albert-invent-at-a-270-million-valuation-1ab03c96?utm_source=chatgpt.com))\n\nThese developments reflect the dynamic and rapidly evolving landscape of AI, with major corporations and financial institutions making significant investments to advance technology and infrastructure in this sector.\n\n\n# Key AI Developments as of February 25, 2025:\n- [Huawei improves AI chip production in boost for China's tech goals](https://www.ft.com/content/f46b7f6d-62ed-4b64-8ad7-2417e5ab34f6?utm_source=chatgpt.com)\n- [Apple cheers Trump with $500bn US investment plan; more losses on Wall Street - as it happened](https://www.theguardian.com/business/live/2025/feb/24/euro-hits-one-month-high-german-election-result-stock-markets-dax-bank-of-england-business-live-news?utm_source=chatgpt.com)\n- [Microsoft axes data center leases as DeepSeek casts doubt on massive AI spend: report](https://nypost.com/2025/02/24/business/microsoft-axes-some-ai-data-center-leases-td-cowen-says/?utm_source=chatgpt.com)\n ",
            "type": "output_text",
            "logprobs": null
          }
        ],
        "role": "assistant",
        "type": "message"
      }
    ]

## Multimodal, Tool-augmented conversation

The Responses API natively supports text, images, and audio modalities. 
Tying everything together, we can build a fully multimodal, tool-augmented interaction with one API call through the responses API.

```python
import base64

from IPython.display import Image, display

# Display the image from the provided URL
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/2880px-Cat_August_2010-4.jpg"
display(Image(url=url, width=400))

response_multimodal = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": 
                 "Come up with keywords related to the image, and search on the web using the search tool for any news related to the keywords"
                 ", summarize the findings and cite the sources."},
                {"type": "input_image", "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/2880px-Cat_August_2010-4.jpg"}
            ]
        }
    ],
    tools=[
        {"type": "web_search"}
    ]
)
```

```python
import json
print(json.dumps(response_multimodal.__dict__, default=lambda o: o.__dict__, indent=4))
```

    {
        "id": "resp_67bd65392a088191a3b802a61f4fba14",
        "created_at": 1740465465.0,
        "error": null,
        "metadata": {},
        "model": "gpt-4o-2024-08-06",
        "object": "response",
        "output": [
            {
                "id": "msg_67bd653ab9cc81918db973f0c1af9fbb",
                "content": [
                    {
                        "annotations": [],
                        "text": "Based on the image of a cat, some relevant keywords could be:\n\n- Cat\n- Feline\n- Pet\n- Animal care\n- Cat behavior\n\nI'll search for recent news related to these keywords.",
                        "type": "output_text",
                        "logprobs": null
                    }
                ],
                "role": "assistant",
                "type": "message"
            },
            {
                "id": "ws_67bd653c7a548191af86757fbbca96e1",
                "status": "completed",
                "type": "web_search_call"
            },
            {
                "id": "msg_67bd653f34fc8191989241b2659fd1b5",
                "content": [
                    {
                        "annotations": [
                            {
                                "index": null,
                                "title": "Cat miraculously survives 3 weeks trapped in sofa during family's cross-country move",
                                "type": "url_citation",
                                "url": "https://nypost.com/2025/02/24/us-news/cat-miraculously-survives-3-weeks-trapped-in-sofa-during-familys-cross-country-move/?utm_source=chatgpt.com"
                            },
                            {
                                "index": null,
                                "title": "Ex-College Soccer Player Accused of Killing Fellow Athlete Brother, Cat Using Knife, Golf Club: Prosecutors",
                                "type": "url_citation",
                                "url": "https://people.com/princeton-murder-soccer-player-accused-murdering-athlete-brother-11685671?utm_source=chatgpt.com"
                            },
                            {
                                "index": null,
                                "title": "Cuddly 8-Year-Old Cat Surrendered to Shelter for Being 'Too Affectionate' Inspires Dozens of Adoption Applications",
                                "type": "url_citation",
                                "url": "https://people.com/cat-surrendered-connecticut-shelter-too-affectionate-11684130?utm_source=chatgpt.com"
                            },
                            {
                                "index": null,
                                "title": "Emaciated cat found in Meriden abandoned in snow dies after rescue attempt, officials say",
                                "type": "url_citation",
                                "url": "https://www.ctinsider.com/recordjournal/article/meriden-animal-control-cat-neglected-abandoned-20172924.php?utm_source=chatgpt.com"
                            },
                            {
                                "index": null,
                                "title": "Cat proves mom correct by using human toilet",
                                "type": "url_citation",
                                "url": "https://nypost.com/video/cat-proves-mom-correct-by-using-human-toilet/?utm_source=chatgpt.com"
                            },
                            {
                                "index": null,
                                "title": "Litter-Robot 3 Connect Review",
                                "type": "url_citation",
                                "url": "https://www.thesprucepets.com/litter-robot-3-connect-review-8780105?utm_source=chatgpt.com"
                            },
                            {
                                "index": null,
                                "title": "Taylor Swift's favourite cat faces breeding ban",
                                "type": "url_citation",
                                "url": "https://www.thetimes.co.uk/article/taylor-swifts-favourite-cat-faces-breeding-ban-k32nvf6kv?utm_source=chatgpt.com"
                            }
                        ],
                        "text": "Here are some recent news stories related to cats:\n\n**1. Cat Survives Three Weeks Trapped in Sofa During Move**\n\nA cat named Sunny-Loo survived three weeks trapped inside a sofa during the Hansons' move from Washington state to Colorado. After disappearing during the move, she was discovered emaciated but alive when the family unpacked their furniture. Sunny-Loo received intensive care and has since been reunited with her family. ([nypost.com](https://nypost.com/2025/02/24/us-news/cat-miraculously-survives-3-weeks-trapped-in-sofa-during-familys-cross-country-move/?utm_source=chatgpt.com))\n\n**2. Man Charged with Killing Brother and Family Cat**\n\nMatthew Hertgen, a former college soccer player, has been charged with the murder of his younger brother, Joseph Hertgen, and animal cruelty for allegedly killing the family cat. The incident occurred in Princeton, New Jersey, where authorities found Joseph's body with signs of trauma. Matthew faces multiple charges, including first-degree