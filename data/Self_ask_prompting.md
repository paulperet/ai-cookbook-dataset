# Guide: Implementing Self-Ask Prompting with the Gemini API

Self-ask prompting is a reasoning technique where a language model explicitly asks itself follow-up questions to break down a complex query. This step-by-step approach helps the model think analytically, similar to chain-of-thought prompting, but with a more structured question-and-answer format.

## Prerequisites

Before you begin, ensure you have the following:

1.  A **Google AI API Key**. If you don't have one, you can [create it here](https://aistudio.google.com/app/apikey).
2.  The API key stored securely. In this guide, we'll assume it's stored in a Colab Secret named `GOOGLE_API_KEY`.

## Step 1: Install and Import Required Libraries

First, install the official Google Generative AI Python SDK.

```bash
pip install -U -q "google-genai>=1.0.0"
```

Next, import the necessary modules.

```python
from google.colab import userdata
from google import genai
from IPython.display import Markdown
```

## Step 2: Configure the Gemini Client

Initialize the client with your API key. This client will be used to interact with the Gemini models.

```python
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 3: Select a Model

Choose a Gemini model for this task. For self-ask prompting, which involves reasoning, a capable model like `gemini-2.5-pro` or `gemini-3-pro-preview` is recommended.

```python
MODEL_ID = "gemini-2.5-pro"  # You can change this to any available model
```

## Step 4: Construct a Self-Ask Prompt

The key to self-ask prompting is providing a clear template in your prompt. This template should show the model the expected format: a main question, a series of follow-up questions with their intermediate answers, and a final synthesis.

Let's construct a prompt that demonstrates this format with one example, then asks a new question.

```python
prompt = """
Question: Who was the president of the united states when Mozart died?
Are follow up questions needed?: yes.
Follow up: When did Mozart died?
Intermediate answer: 1791.
Follow up: Who was the president of the united states in 1791?
Intermediate answer: George Washington.
Final answer: When Mozart died George Washington was the president of the USA.

Question: Where did the Emperor of Japan, who ruled the year Maria Skłodowska was born, die?
"""
```

## Step 5: Generate a Response

Send the constructed prompt to the Gemini model using the client.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)
```

## Step 6: Display the Result

View the model's response. It should follow the self-ask pattern you demonstrated.

```python
Markdown(response.text)
```

**Expected Output:**
The model should generate a response structured like the example in the prompt. For the question about Maria Skłodowska (Marie Curie), a correct chain might look like this:

```
Are follow up questions needed?: Yes.
Follow up: When was Maria Skłodowska born?
Intermediate answer: 1867.
Follow up: Who was the Emperor of Japan in 1867?
Intermediate answer: Emperor Kōmei.
Follow up: When did Emperor Kōmei die?
Intermediate answer: 1867.
Final answer: Emperor Kōmei died in Kyoto.
```

## How It Works & Advanced Use

The model uses the initial example as a guide. For the new question, it:
1.  Recognizes the need for follow-up questions.
2.  Asks the first logical sub-question (When was she born?).
3.  Provides a hypothetical "Intermediate answer."
4.  Uses that answer to ask the next question (Who was Emperor then?).
5.  Continues until it has all necessary information to synthesize the **Final answer**.

**Advanced Application: Self-Ask with Function Calling**
This technique is powerful when combined with tools. The "Follow up" questions can be used as inputs to external functions—like a web search or database query. The function's real answer is then fed back to the model, which can decide to ask another question or provide the final answer. This creates a robust, fact-grounded reasoning loop.

For a practical example of using Gemini with external data retrieval, see the [Search re-ranking using Gemini embeddings](https://github.com/google-gemini/cookbook/blob/main/examples/Search_reranking_using_embeddings.ipynb) cookbook example.