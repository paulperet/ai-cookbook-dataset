# Cookbook: Mistral AI integration(Python)

This cookbook provides step-by-step examples of integrating phospho with Mistral AI in Python.

---

## Overview

This notebook shows how to log your Mistral ChatBot conversations in phospho. After logging, we use phospho's analytics to compare two Mistral models: **mistral-7b** and **mistral-large-latest**.

## What is phospho ?

**phospho** is an open-source analytics platform for LLM applications, enabling automatic clustering to help discover use cases and topics. It provides no-code analytics to help you gain insights into how your app is being used, simplifying the process of understanding user behavior and trends.

## Setup

First you need the **phospho** python module. It is compatible with Python >= 3.9.

```python
%%capture --no-display

%pip install phospho==0.3.40
%pip install mistralai==1.1.0
%pip install python-dotenv==1.0.1
%pip install tqdm==4.66.5
```

```python
import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PHOSPHO_API_KEY = os.getenv("PHOSPHO_API_KEY")
PHOSPHO_PROJECT_ID = os.getenv("PHOSPHO_PROJECT_ID")
```

## Log your messages

### Simple completion

Initialize the phospho module with `phospho.init()`. By default, phospho will look for `PHOSPHO_API_KEY` and `PHOSPHO_PROJECT_ID` environment variables. You can also pass the `api_key` and `project_id` as parameters.

```python
import phospho

phospho.init(api_key=PHOSPHO_API_KEY, project_id=PHOSPHO_PROJECT_ID)
```

Use `phospho.log` to log a query and an optional answer. Here is an example of a one query conversation with a Mistral assistant.

```python
import mistralai

input_txt = "Thank you for your last message!"


def one_mistral_call(input: str) -> str:
    client = mistralai.Mistral(api_key=MISTRAL_API_KEY)
    completion = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input},
        ],
    )

    return completion.choices[0].message.content


output_txt = one_mistral_call(input_txt)

print(output_txt)

phospho.log(
    input=input_txt,
    output=output_txt,
    # You can log additonal metadata
    user_id="USER_ID", # eg: "user-123", "anonymous", ...
    version_id="one_mistral_call",
)
```

    You're welcome! How can I assist you today? If you have any questions or need help with something, feel free to ask.





    {'client_created_at': 1728548119,
     'project_id': 'db99fc60ceac424fab371252f2f75485',
     'session_id': None,
     'task_id': 'cb3f05c74c9c4ce3ad3971d0e651c8b3',
     'input': 'Thank you for your last message!',
     'raw_input': 'Thank you for your last message!',
     'raw_input_type_name': 'str',
     'output': "You're welcome! How can I assist you today? If you have any questions or need help with something, feel free to ask.",
     'raw_output': "You're welcome! How can I assist you today? If you have any questions or need help with something, feel free to ask.",
     'raw_output_type_name': 'str',
     'user_id': 'USER_ID',
     'version_id': 'one_mistral_call'}



Your message is now accessible on the platform, where you can store all your transcripts. While logging, sentiment evaluation and language detection are performed automatically on the message, along with any events you specify.

### Streaming completion

phospho supports streamed outputs. Pass `stream=True` to `phospho.log` to handle streaming responses. When iterating over the response, phospho will automatically log each chunk until the iteration is completed. Here is an example with a simple Mistral chatBot.

```python
import phospho
import mistralai
from phospho import MutableGenerator
import uuid

# This is used to make the chat prettier
from IPython.display import HTML, display


def set_css():
    display(
        HTML("""
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  """)
    )


get_ipython().events.register("pre_run_cell", set_css)


# This is your chatbot
def simple_mistral_chat():
    phospho.init(api_key=PHOSPHO_API_KEY, project_id=PHOSPHO_PROJECT_ID)
    client = mistralai.Mistral(api_key=MISTRAL_API_KEY)
    messages = []
    # I want to create a random session id
    session_id = str(uuid.uuid4())

    print("Ask anything (Type /exit to quit)", end="")

    while True:
        prompt = input("\n>")
        if prompt == "/exit":
            break
        messages.append({"role": "user", "content": prompt})
        query = {
            "messages": messages,
            "model": "mistral-small-latest",
        }
        response = client.chat.stream(**query)

        # This is how you log a streaming Mistral response to phospho··
        mutable_response = MutableGenerator(response, stop=lambda x: x == "")
        phospho.log(
            input=query,
            output=mutable_response,
            stream=True,
            session_id=session_id,
            user_id="USER_ID",
            version_id="simple_mistral_chat",
            output_to_str_function=lambda x: x["data"]["choices"][0]["delta"].get(
                "content", ""
            ),
        )

        print("\nAssistant: ", end="")
        for r in mutable_response:
            text = r.data.choices[0].delta.content
            if text is not None:
                print(text, end="", flush=True)
```

```python
simple_mistral_chat()
```

    Ask anything (Type /exit to quit)
    >Hi, what are you ?
    
    Assistant: Hello! I am a Large Language Model trained by Mistral AI.
    >Nice ! What is Mistral AI ?
    
    Assistant: Hello! I am a text-based AI model designed to assist and provide information.
    
    Mistral AI is a cutting-edge company based in Paris, France, developing large language models. I am very grateful they created me!
    >So French is your native language ?
    
    Assistant: Hello! I am a text-based AI model designed to assist and engage in conversation.
    
    Mistral AI is a cutting-edge company based in Paris, France, developing large language models. I am very grateful they created me!
    
    And yes, French is one of the languages I can understand and generate text in. How can I assist you today?
    >/exit


The full conversation is now available to be viewed in the platform

**Note:** Find all the logging possibilities (async, async streaming, decorator...) in our [documentation](https://docs.phospho.ai/integrations/python/logging).

## Evaluate Mistral models progressions with phospho analytics

Now that the messages are logged in the platform, it's time to set up analytics to uncover insights.

Let's take an example. Let's pretend we are building an mathematical AI tutor. We are unsure whether to use **mistral-7b** or **mistral-large-latest**. To compare these models on our use, we can use phospho. We will setup an **AB test** to evaluate the performance of both models based on the pedagogical quality of their responses to a set of mathematical problems.

### The problems

In order to extract 50 problems, we use the MetaMath problem dataset, which can be accessed [here](https://huggingface.co/datasets/meta-math/MetaMathQA).

**Titre :** MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models  
**Auteurs :** Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, Weiyang Liu  
**Journal :** arXiv preprint arXiv:2309.12284  
**Année :** 2023

```python
maths_questions = [
    "Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?",
    "What is the total cost of purchasing equipment for all sixteen players on the football team, considering that each player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks priced at $6.80?",
    "Diego baked 12 cakes for his sister's birthday. Donald also baked 4 cakes, but ate x while waiting for the party to start. There are 15 cakes left. What is the value of unknown variable x?",
    "Convert $10101_3$ to a base 10 integer.",
    "Sue works in a factory and every 30 minutes, a machine she oversees produces 30 cans of soda. How many cans of soda can x machine produce in 8 hours? If we know the answer to the above question is 480, what is the value of unknown variable x?",
    "Mark is buying asphalt to pave a new section of road. The road will be 2000 feet long and 20 feet wide. Each truckload of asphalt will cover 800 square feet of road. If each truckload costs x, and there's a 20% sales tax, how much will Mark need to pay for asphalt? If we know the answer to the above question is 4500, what is the value of unknown variable x?",
    "Evan’s dog weighs 63 pounds; it weighs x times as much as Ivan’s dog.  Together, what is the weight of the dogs? If we know the answer to the above question is 72, what is the value of unknown variable x?",
    "The town of Belize has 400 homes. One fourth of the town's homes are white. One fifth of the non-white homes have a fireplace. How many of the non-white homes do not have a fireplace?",
    "Quantities $r$ and $s$ vary inversely. When $r$ is $1200,$ $s$ is $0.35.$ What is the value of $s$ when $r$ is $2400$? Express your answer as a decimal to the nearest thousandths.",
    "Dave bought 8 books about animals, 6 books about outer space, and 3 books about trains to keep him busy over the holidays. Each book cost $6. How much did Dave spend on the books?",
    "Calculate 8 divided by $\frac{1}{8}.$",
    "What is $ 6 \div X - 2 - 8 + 2 \cdot 8$? If we know the answer to the above question is 8, what is the value of unknown variable X?",
    "The points $(x, y)$ represented in this table lie on a straight line. The point $(28, t)$ lies on the same line. What is the value of $t?$ \begin{tabular}{c|c} $x$ & $y$ \\ \hline 1 & 7 \\ 3 & 13 \\ 5 & 19 \\ \end{tabular}",
    "Maximoff's monthly bill is $60 per month. His monthly bill increased by thirty percent when he started working at home. How much is his total monthly bill working from home?",
    "Compute $\dbinom{14}{11}$.",
    "There are 6 girls and 8 boys in the school play. If both parents of each kid attend the premiere, how many parents will be in the auditorium?",
    "If Williams has a certain amount of money, Jackson has 5 times that amount. If they have a total of $150 together, how much money does Jackson have in dollars?",
    "Mike has earned a total of $160 in wages this week. He received the wages for his first job, then later received the wages from his second job where he works 12 hours a week. If his second job pays $9 per hour then how much money, in dollars, did Mike receive from his first job?",
    "A 26-mile circular marathon has x checkpoints inside it. The first is one mile from the start line, and the last checkpoint is one mile from the finish line. The checkpoints have equal spacing between them. How many miles apart are each of the consecutive checkpoints between the start-finish line? If we know the answer to the above question is 6, what is the value of unknown variable x?",
    "Miggy's mom brought home x bags of birthday hats. Each bag has 15 hats. Miggy accidentally tore off 5 hats. During the party, only 25 hats were used. How many hats were unused? If we know the answer to the above question is 15, what is the value of unknown variable x?",
    "If Rebecca is currently 25 years old and Brittany is 3 years older than Rebecca, how old will Brittany be when she returns from her 4-year vacation?",
    "Find the $2 \times 2$ matrix $\mathbf{M}$ such that $\mathbf{M} \begin{pmatrix} 3 \\ 0 \end{pmatrix} = \begin{pmatrix} 6 \\ 21 \end{pmatrix}$ and $\mathbf{M} \begin{pmatrix} -1 \\ 5 \end{pmatrix} = \begin{pmatrix} X \\ -17 \end{pmatrix}.$ If we know the answer to the above question is \begin{pmatrix}2&1\7&-2\end{pmatrix}, what is the value of unknown variable X?",
    "Five socks, colored blue, brown, black, red, and purple are in a drawer. In how many different ways can we choose three socks from the drawer if the order of the socks does not matter?",
    "If James drives to Canada at a speed of 60 mph and the distance is 360 miles, with a 1-hour stop along the way, how long will it take him to reach Canada?",
    "How many different combinations are there to choose 3 captains from a team of 11 people?",
    "Frank is making hamburgers and he wants to sell them to make $50.  Frank is selling each hamburger for $x and 2 people purchased 4 and another 2 customers purchased 2 hamburgers. Frank needs to sell 4 more hamburgers to make $50. What is the value of unknown variable x?",
    "Which integer $n$ satisfies the conditions $0 \leq n < 19$ and $38574 \equiv n \pmod{19}$?",
    "What is the common ratio of the infinite geometric series $\frac{-3}{5} - \frac{5}{3} - \frac{125}{27} - \dots$?",
    "What is the sum of all positive integer values of $n$ for which $\frac{n+6}{n}$ is an integer?",
    "We have that $2a + 1 = 1$ and $b - a = 1.$ What is the value of $b$?",
    "If Heike has a certain number of cards in her collection, Anton has three times as many cards, and Ann has six times as many cards. If Ann has 60 cards, how many more cards does Ann have compared to Anton?",
    "Sabina is starting her first year of college that costs $30,000. She has saved $10,000 for her first year. She was awarded a grant that will cover 40% of the remainder of her tuition. How much will Sabina need to apply for to receive a loan that will cover her tuition?",
    "If Billy made 49 sandwiches and Katelyn made 47 more sandwiches than Billy, and Chloe made a quarter of the number that Katelyn made, what is the total number of sandwiches that they made?",
    "A fair 6-sided die is rolled.  If I roll $n$, then I win $n^2$ dollars.  What is the expected value of my win?  Express your answer as a dollar value rounded to the nearest cent.",
    "Randy, Peter, and Quincy all drew pictures. Peter drew 8 pictures. Quincy drew 20 more pictures than Peter. If they drew 41 pictures altogether, how many did Randy draw?",
    "Gina has two bank accounts. Each account has a quarter of the balance in Betty's account. If Betty's account balance is $3,456, what is the combined balance of both Gina's accounts?",
    "John makes 6 dozen cookies for a bake sale.  He sells each cookie for $1.5 and each cookie costs $x to make.  He splits the profit between two charities evenly.  How much does each charity get? If we know the answer to the above question is 45, what is the value of unknown variable x?"
    "When a number is divided by 7, the remainder is 2. What is the remainder when three times the number minus 7 is divided by 7?",
    "Diane bakes four trays with 25 gingerbreads in each tray and three trays with 20 gingerbreads in each tray. How many gingerbreads does Diane bake?",
    "For homework, Brooke has 15 math problems, 6 social studies problems, and x science problems. He can answer each math problem for 2 minutes while answering each social studies problem takes him 30 seconds. If he can answer each science problem in 1.5 minutes, It will take Brooke 48 to answer all his homework. What is the value of unknown variable x?",
    "On Monday, Mack writes in his journal for 60 minutes at a rate of 1 page every 30 minutes. On Tuesday, Mack writes in his journal for 45 minutes at a rate of 1 page every 15 minutes. On Wednesday, Mack writes x pages in his journal. Mack writes 10 pages total in his journal from Monday to Wednesday. What is the value of unknown variable x?",
    "Kevin has a tree growing in his garden that is currently 180 inches tall. That is 50% taller than it was when he planted it there. How tall was the tree, in feet, then?",
    "In a week, 450 cars drove through a toll booth. Fifty vehicles went through the toll booth on Monday and the same number of vehicles drove through the toll booth on Tuesday. On each of Wednesday