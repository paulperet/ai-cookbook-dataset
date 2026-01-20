# Question Answering with Langchain, Qdrant and OpenAI

This notebook presents how to implement a Question Answering system with Langchain, Qdrant as a knowledge based and OpenAI embeddings. If you are not familiar with Qdrant, it's better to check out the [Getting_started_with_Qdrant_and_OpenAI.ipynb](Getting_started_with_Qdrant_and_OpenAI.ipynb) notebook.

This notebook presents an end-to-end process of:
1. Calculating the embeddings with OpenAI API.
2. Storing the embeddings in a local instance of Qdrant to build a knowledge base.
3. Converting raw text query to an embedding with OpenAI API.
4. Using Qdrant to perform the nearest neighbour search in the created collection to find some context.
5. Asking LLM to find the answer in a given context.

All the steps will be simplified to calling some corresponding Langchain methods.

## Prerequisites

For the purposes of this exercise we need to prepare a couple of things:

1. Qdrant server instance. In our case a local Docker container.
2. The [qdrant-client](https://github.com/qdrant/qdrant_client) library to interact with the vector database.
3. [Langchain](https://github.com/hwchase17/langchain) as a framework.
3. An [OpenAI API key](https://beta.openai.com/account/api-keys).

### Start Qdrant server

We're going to use a local Qdrant instance running in a Docker container. The easiest way to launch it is to use the attached [docker-compose.yaml] file and run the following command:

```python
! docker-compose up -d
```

We might validate if the server was launched successfully by running a simple curl command:

```python
! curl http://localhost:6333
```

### Install requirements

This notebook obviously requires the `openai`, `langchain` and `qdrant-client` packages.

```python
! pip install openai qdrant-client "langchain==0.0.100" wget
```

### Prepare your OpenAI API key

The OpenAI API key is used for vectorization of the documents and queries.

If you don't have an OpenAI API key, you can get one from [https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys).

Once you get your key, please add it to your environment variables as `OPENAI_API_KEY` by running following command:

```python
! export OPENAI_API_KEY="your API key"
```

```python
# Test that your OpenAI API key is correctly set as an environment variable
# Note. if you run this notebook locally, you will need to reload your terminal and the notebook for the env variables to be live.
import os

# Note. alternatively you can set a temporary env variable like this:
# os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

if os.getenv("OPENAI_API_KEY") is not None:
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
```

## Load data

In this section we are going to load the data containing some natural questions and answers to them. All the data will be used to create a Langchain application with Qdrant being the knowledge base.

```python
import wget

# All the examples come from https://ai.google.com/research/NaturalQuestions
# This is a sample of the training set that we download and extract for some
# further processing.
wget.download("https://storage.googleapis.com/dataset-natural-questions/questions.json")
wget.download("https://storage.googleapis.com/dataset-natural-questions/answers.json")
```

```python
import json

with open("questions.json", "r") as fp:
    questions = json.load(fp)

with open("answers.json", "r") as fp:
    answers = json.load(fp)
```

```python
print(questions[0])
```

```python
print(answers[0])
```

## Chain definition

Langchain is already integrated with Qdrant and performs all the indexing for given list of documents. In our case we are going to store the set of answers we have.

```python
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain import VectorDBQA, OpenAI

embeddings = OpenAIEmbeddings()
doc_store = Qdrant.from_texts(
    answers, embeddings, host="localhost" 
)
```

At this stage all the possible answers are already stored in Qdrant, so we can define the whole QA chain.

```python
llm = OpenAI()
qa = VectorDBQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    vectorstore=doc_store,
    return_source_documents=False,
)
```

## Search data

Once the data is put into Qdrant we can start asking some questions. A question will be automatically vectorized by OpenAI model, and the created vector will be used to find some possibly matching answers in Qdrant. Once retrieved, the most similar answers will be incorporated into the prompt sent to OpenAI Large Language Model. The communication between all the services is shown on a graph:

```python
import random

random.seed(52)
selected_questions = random.choices(questions, k=5)
```

```python
for question in selected_questions:
    print(">", question)
    print(qa.run(question), end="\n\n")
```

### Custom prompt templates

The `stuff` chain type in Langchain uses a specific prompt with question and context documents incorporated. This is what the default prompt looks like:

```text
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:
```

We can, however, provide our prompt template and change the behaviour of the OpenAI LLM, while still using the `stuff` chain type. It is important to keep `{context}` and `{question}` as placeholders.

#### Experimenting with custom prompts

We can try using a different prompt template, so the model:
1. Responds with a single-sentence answer if it knows it.
2. Suggests a random song title if it doesn't know the answer to our question.

```python
from langchain.prompts import PromptTemplate
```

```python
custom_prompt = """
Use the following pieces of context to answer the question at the end. Please provide
a short single-sentence summary answer only. If you don't know the answer or if it's 
not present in given context, don't try to make up an answer, but suggest me a random 
unrelated song title I could listen to. 
Context: {context}
Question: {question}
Helpful Answer:
"""
```

```python
custom_prompt_template = PromptTemplate(
    template=custom_prompt, input_variables=["context", "question"]
)
```

```python
custom_qa = VectorDBQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    vectorstore=doc_store,
    return_source_documents=False,
    chain_type_kwargs={"prompt": custom_prompt_template},
)
```

```python
random.seed(41)
for question in random.choices(questions, k=5):
    print(">", question)
    print(custom_qa.run(question), end="\n\n")
```