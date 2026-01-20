# Prompting Capabilities with Mistral AI

When you first start using Mistral models, your first interaction will revolve around prompts. The art of crafting effective prompts is essential for generating desirable responses from Mistral models or other LLMs. This guide will walk you through example prompts showing four different prompting capabilities.

- Classification
- Summarization
- Personalization
- Evaluation

```python
! pip install mistralai
```

```python
from mistralai import Mistral
```

```python
api_key = "TYPE YOUR API KEY"
```

```python
def run_mistral(user_message, model="mistral-large-latest"):
    client = Mistral(api_key=api_key)
    messages = [
        {"role":"user", "content":user_message}
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return (chat_response.choices[0].message.content)
```

## Classification
Mistral models can easily categorize text into distinct classes. In this example prompt, we can define a list of predefined categories and ask Mistral models to classify user inquiry.

```python
def user_message(inquiry):
    user_message = (
        f"""
        You are a bank customer service bot. Your task is to assess customer intent
        and categorize customer inquiry after <<<>>> into one of the following predefined categories:

        card arrival
        change pin
        exchange rate
        country support
        cancel transfer
        charge dispute

        If the text doesn't fit into any of the above categories, classify it as:
        customer service

        You will only respond with the predefined category. Do not include the word "Category". Do not provide explanations or notes.

        ####
        Here are some examples:

        Inquiry: How do I know if I will get my card, or if it is lost? I am concerned about the delivery process and would like to ensure that I will receive my card as expected. Could you please provide information about the tracking process for my card, or confirm if there are any indicators to identify if the card has been lost during delivery?
        Category: card arrival
        Inquiry: I am planning an international trip to Paris and would like to inquire about the current exchange rates for Euros as well as any associated fees for foreign transactions.
        Category: exchange rate
        Inquiry: What countries are getting support? I will be traveling and living abroad for an extended period of time, specifically in France and Germany, and would appreciate any information regarding compatibility and functionality in these regions.
        Category: country support
        Inquiry: Can I get help starting my computer? I am having difficulty starting my computer, and would appreciate your expertise in helping me troubleshoot the issue.
        Category: customer service
        ###

        <<<
        Inquiry: {inquiry}
        >>>
        """
    )
    return user_message
```

### Strategies we used:

- **Few shot learning**: Few-shot learning or in-context learning is when we give a few examples in the prompts, and the LLM can generate corresponding output based on the example demonstrations. Few-shot learning can often improve model performance especially when the task is difficult or when we want the model to respond in a specific manner.
- **Delimiter**: Delimiters like ### <<< >>> specify the boundary between different sections of the text. In our example, we used ### to indicate examples and <<<>>> to indicate customer inquiry.
- **Role playing**: Providing LLM a role (e.g., "You are a bank customer service bot.") adds personal context to the model and often leads to better performance.

```python
print(run_mistral(user_message(
    "I am inquiring about the availability of your cards in the EU, as I am a resident of France and am interested in using your cards. "
)))
```

    country support

```python
print(run_mistral(user_message("What's the weather today?")))
```

    customer service

## Summarization

Summarization is a common task for LLMs due to their natural language understanding and generation capabilities. Here is an example prompt we can use to generate interesting questions about an essay and summarize the essay.

```python
import requests
response = requests.get('https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt')
essay = response.text
```

```python
message = f"""
You are a commentator. Your task is to write a report on an essay.
When presented with the essay, come up with interesting questions to ask, and answer each question.
Afterward, combine all the information and write a report in the markdown format.

# Essay:
{essay}

# Instructions:
## Summarize:
In clear and concise language, summarize the key points and themes presented in the essay.

## Interesting Questions:
Generate three distinct and thought-provoking questions that can be asked about the content of the essay. For each question:
- After "Q: ", describe the problem
- After "A: ", provide a detailed explanation of the problem addressed in the question.
- Enclose the ultimate answer in <>.

## Write a report
Using the essay summary and the answers to the interesting questions, create a comprehensive report in Markdown format.
"""
```

```python
print(run_mistral(message))
```

    Summary:
    The essay is a personal account of the author's journey through various interests and careers, including writing, programming, philosophy, and art. The author began writing short stories and programming on an IBM 1401 in school, but found little success in either. They then studied philosophy in college, but found it unfulfilling. The author became interested in artificial intelligence (AI) and decided to switch to it, but eventually realized that AI as practiced at the time was a hoax. They then turned to Lisp, a programming language they found interesting for its own sake, and wrote a book about Lisp hacking. The author also became interested in painting and attended art school, but found it disappointing. They eventually dropped out and moved to New York to become a New York artist.
    
    Interesting Questions:
    
    Q1: Why did the author find AI as practiced at the time to be a hoax?
    A1: The author realized that AI, as practiced at the time, was a hoax because it was limited to a subset of natural language that was a formal language, and there was an unbridgeable gap between what it could do and actually understanding natural language. The author also found that the brokenness of this approach generated a lot of opportunities to write papers about various band-aids that could be applied to it, but it was never going to get them to the level of understanding of natural language they were hoping for.
    
    Q2: Why did the author decide to focus on Lisp after realizing that AI was a hoax?
    A2: The author decided to focus on Lisp after realizing that AI was a hoax because they knew from experience that Lisp was interesting for its own sake and not just for its association with AI. They also saw an opportunity to write a book about Lisp hacking, which they felt would help them learn more about the language.
    
    Q3: Why did the author decide to become a New York artist after dropping out of art school?
    A3: The author decided to become a New York artist after dropping out of art school because they were disappointed with the lack of rigor and structure in the painting department. They also wanted to be truly independent and not have to rely on a boss or research funding. They felt that as an artist, they could make a living by being industrious and living cheaply, and they were excited about the possibility of making art that would last and not become obsolete.
    
    Report:
    
    # Essay Report
    
    This essay is a personal account of the author's journey through various interests and careers, including writing, programming, philosophy, and art. The author began writing short stories and programming on an IBM 1401 in school, but found little success in either. They then studied philosophy in college, but found it unfulfilling.
    
    The author became interested in artificial intelligence (AI) and decided to switch to it, but eventually realized that AI as practiced at the time was a hoax. They found that AI was limited to a subset of natural language that was a formal language, and there was an unbridgeable gap between what it could do and actually understanding natural language. The author also found that the brokenness of this approach generated a lot of opportunities to write papers about various band-aids that could be applied to it, but it was never going to get them to the level of understanding of natural language they were hoping for.
    
    After realizing that AI was a hoax, the author turned to Lisp, a programming language they found interesting for its own sake. They decided to write a book about Lisp hacking, which they felt would help them learn more about the language.
    
    The author also became interested in painting and attended art school, but found it disappointing. They eventually dropped out and moved to New York to become a New York artist. They were disappointed with the lack of rigor and structure in the painting department at art school, and wanted to be truly independent and not have to rely on a boss or research funding. They felt that as an artist, they could make a living by being industrious and living cheaply, and they were excited about the possibility of making art that would last and not become obsolete.
    
    In conclusion, the author's journey through various interests and careers highlights the importance of finding something that one is truly passionate about and pursuing it, even if it means taking unconventional paths and facing challenges along the way.

## Strategies we used:

- **Step-by-step instructions**: This strategy is inspired by the chain-of-thought prompting that enables LLMs to use a series of intermediate reasoning steps to tackle complex tasks. It's often easier to solve complex problems when we decompose them into simpler and small steps and it's easier for us to debug and inspect the model behavior.  In our example, we break down the task into three steps: summarize, generate interesting questions, and write a report. This helps the language to think in each step and generate a more comprehensive final report.
- **Example generation**: We can ask LLMs to automatically guide the reasoning and understanding process by generating examples with the explanations and steps. In this example, we ask the LLM to generate three questions and provide detailed explanations for each question.
- **Output formatting**: We can ask LLMs to output in a certain format by directly asking "write a report in the Markdown format".

## Personlization

LLMs excel at personalization tasks as they can deliver content that aligns closely with individual users. In this example, we create personalized email responses to address customer questions.

```python
email = """
Dear mortgage lender,

What's your 30-year fixed-rate APR, how is it compared to the 15-year fixed rate?

Regards,
Anna
"""
```

```python
message = f"""

You are a mortgage lender customer service bot, and your task is to create personalized email responses to address customer questions.
Answer the customer's inquiry using the provided facts below. Ensure that your response is clear, concise, and
directly addresses the customer's question. Address the customer in a friendly and professional manner. Sign the email with
"Lender Customer Support."



# Facts
30-year fixed-rate: interest rate 6.403%, APR 6.484%
20-year fixed-rate: interest rate 6.329%, APR 6.429%
15-year fixed-rate: interest rate 5.705%, APR 5.848%
10-year fixed-rate: interest rate 5.500%, APR 5.720%
7-year ARM: interest rate 7.011%, APR 7.660%
5-year ARM: interest rate 6.880%, APR 7.754%
3-year ARM: interest rate 6.125%, APR 7.204%
30-year fixed-rate FHA: interest rate 5.527%, APR 6.316%
30-year fixed-rate VA: interest rate 5.684%, APR 6.062%

# Email
{email}
"""
```

```python
print(run_mistral(message))
```

    Subject: Information on Our 30-Year and 15-Year Fixed-Rate Mortgages
    
    Dear Anna,
    
    Thank you for reaching out to us with your inquiry. I'm happy to provide you with the information you need to make an informed decision about your mortgage options.
    
    Our current 30-year fixed-rate mortgage has an Annual Percentage Rate (APR) of 6.484%. On the other hand, our 15-year fixed-rate mortgage has a lower APR of 5.848%.
    
    The difference in APR between the two options is 0.636%. While the 15-year fixed-rate mortgage has a lower APR, it's important to note that the monthly payments are typically higher due to the shorter repayment period. However, you'll pay less in interest over the life of the loan compared to the 30-year fixed-rate mortgage.
    
    I hope this information helps you in your decision-making process. If you have any more questions or would like further clarification, please don't hesitate to contact us.
    
    Best Regards,
    
    Lender Customer Support

### Strategies we used:
- Providing facts: Incorporating facts into prompts can be useful for developing customer support bots. It’s important to use clear and concise language when presenting these facts. This can help the LLM to provide accurate and quick responses to customer queries.

## Evaluation

There are many ways to evaluate LLM outputs. Here are three approaches for your reference: include a confidence score, introduce an evaluation step, or employ another LLM for evaluation.

## Include a confidence score
We can include a confidence score along with the generated output in the prompt.

```python
def run_mistral(user_message, model="mistral-large-latest"):
    client = Mistral(api_key=api_key)
    messages = [
        {
            "role":"user",
            "content": user_message
        }
    ]
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        temperature=1,
        response_format = {
          "type": "json_object"
        }
    )
    return (chat_response.choices[0].message.content)
```

```python
message = f"""
You are a summarization system that can provide summaries with associated confidence scores.
In clear and concise language, provide three short summaries of the following essay, along with their confidence scores.
You will only respond with a JSON object with the key Summary and Confidence. Do not provide explanations.

# Essay:
{essay}

"""
```

```python
print(run_mistral(message))
```

    {
    "Summary 1": "The author writes about their experience with writing and programming before college. They attempted to write programs on an IBM 1401, but struggled to find a purpose for them. They were later introduced to microcomputers and became interested in AI after reading a novel and seeing a documentary.",
    "Confidence 1": 0.95
    },
    {
    "Summary 2": "The author discusses their decision to switch from studying philosophy to AI in college. They were disappointed with the lack of depth in philosophy courses and saw potential in the field of AI. They began teaching themselves AI and Lisp.",
    "Confidence 2": 0.9
    },
    {
    "Summary 3": "The author reflects on their realization that AI, as practiced at the time, was not going to lead to true understanding of natural language. They decided to focus on Lisp instead and wrote a book about Lisp hacking.",
    "Confidence 3": 0.85
    }

### Strategies we used:
- JSON output: For facilitating downstream tasks, JSON format output is frequently preferred. We can enable the JSON mode by setting the response_format to `{"type": "json_object"}` and specify in the prompt that “You will only respond with a JSON object with the key Summary and Confidence.” Specifying these keys within the JSON object is beneficial for clarity and consistency.
- Higher Temperature: In this example, we increase the temperature score to encourage the model to be more creative and output three generated summaries that are different from each other.

## Introduce an evaluation step
We can also add a second step in the prompt for evaluation.

```python
message = f"""
You are given an essay text and need to provide summaries and evaluate them.

# Essay:
{essay}

Step 1: In this step, provide three short summaries of the given essay. Each summary should be clear, concise, and capture the key points of the speech. Aim for around 2-3 sentences for each summary.
Step 2: Evaluate the three summaries from Step 1 and rate which one you believe is the best. Explain your choice by pointing out specific reasons such as clarity, completeness, and relevance to the speech content.


"""
print(run_mistral(message))
```

    Step 1:
    
    Summary 1: This essay recounts the author's diverse experiences in writing, programming, and painting, and how these pursuits intertwined throughout their life. The author describes their journey from writing short stories to programming on early computers, and their eventual focus on Lisp hacking and painting.
    
    Summary 2: In this essay, the author shares their personal journey through various interests and careers, including writing, programming, and art. They discuss their experiences with early computing, their love for Lisp programming, and their eventual decision to pursue art, leading to their time at art schools in the US and Italy.
    
    Summary 3: The author of this essay chronicles their life's exploration of writing, programming, and art. They delve into their early experiences with writing and programming, their affinity for Lisp, and their pursuit of art, which led them to study at different art institutions and live in various places.
    
    Step 2:
    
    The best summary among the three is Summary 1. This summary provides a clear and concise overview of the author's experiences in writing, programming, and painting. It captures the key points of the essay, such as the author's journey from writing short stories to programming on early computers, and their eventual focus on Lisp hacking and painting. The summary is relevant to the speech content and presents the information in a coherent manner.

## Employ another LLM for evaluation
In production systems, it is common to employ another LLM for evaluation so that the evaluation step can be separate from the generation step.

- Step 1: use the first LLM to generate three summaries

```python
message = f"""
Provide three short summaries of the given essay. Each summary should be clear, concise, and capture the key points of the essay.
Aim for around 2-3 sentences for each summary.

# essay:
{essay}

"""
summaries = run_mistral(message)
```

```python
print(summaries)
```

    [First Entry, ..., Last Entry]

- Step 2: use another LLM to rate the generated summaries

```python
message = f"""
You are given an essay and three summaries of the essay. Evaluate the three summaries and rate which one you believe is the best.
Explain your choice by pointing out specific reasons such as clarity, completeness, and relevance to the essay content.

# Essay:
{essay}

# Summaries
{summaries}

"""
print(run_mistral(message))
```

    All three summaries accurately capture the main points of the essay, but the third summary is the best. It is the most clear and concise,