# Fine Tuning MistralAI models using Finetuning API and LlamaIndex

In this notebook, we walk through an example of fine-tuning `open-mistral-7b` using MistralAI finetuning API.

Specifically, we attempt to distill `mistral-large-latest`'s knowledge, by generating training data with `mistral-large-latest` to then fine-tune `open-mistral-7b`.

All training data is generated using two different sections of our index data, creating both a training and evalution set.

We will use `mistral-small-largest` to create synthetic training and evaluation questions to avoid any biases towards `open-mistral-7b` and `mistral-large-latest`.

We then finetune with our `MistraAIFinetuneEngine` wrapper abstraction.

Evaluation is done using the `ragas` library, which we will detail later on.

We can monitor the metrics on `Weights & Biases`


```python
%pip install llama-index-finetuning
%pip install llama-index-finetuning-callbacks
%pip install llama-index-llms-mistralai
%pip install llama-index-embeddings-mistralai
```


```python
# !pip install llama-index pypdf sentence-transformers ragas
```


```python
# NESTED ASYNCIO LOOP NEEDED TO RUN ASYNC IN A NOTEBOOK
import nest_asyncio

nest_asyncio.apply()
```

## Set API Key


```python
import os

os.environ["MISTRAL_API_KEY"] = "<YOUR MISTRALAI API KEY>"
```

## Download Data

Here, we first down load the PDF that we will use to generate training data.


```python
!curl https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter03.pdf --output IPCC_AR6_WGII_Chapter03.pdf
```

[First Entry, ..., Last Entry]


The next step is generating a training and eval dataset.

We will generate 40 training and 40 evaluation questions on different sections of the PDF we downloaded.

We can use `open-mistral-7b` on the eval questions to get our baseline performance.

Then, we will use `mistral-large-latest` on the train questions to generate our training data. 


## Load Data


```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.evaluation import DatasetGenerator

documents = SimpleDirectoryReader(
    input_files=["IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()
```

## Setup LLM and Embedding Model


```python
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding

open_mistral = MistralAI(
    model="open-mistral-7b", temperature=0.1
)  # model to be finetuning
mistral_small = MistralAI(
    model="mistral-small-latest", temperature=0.1
)  # model for question generation
embed_model = MistralAIEmbedding()
```

## Training and Evaluation Data Generation


```python
question_gen_query = (
    "You are a Teacher/ Professor. Your task is to setup "
    "a quiz/examination. Using the provided context, formulate "
    "a single question that captures an important fact from the "
    "context. Restrict the question to the context information provided."
    "You should generate only question and nothing else."
)

dataset_generator = DatasetGenerator.from_documents(
    documents[:80],
    question_gen_query=question_gen_query,
    llm=mistral_small,
)
```

    /Users/ravithejad/Desktop/llamaindex/lib/python3.9/site-packages/llama_index/core/evaluation/dataset_generation.py:215: DeprecationWarning: Call to deprecated class DatasetGenerator. (Deprecated in favor of `RagDatasetGenerator` which should be used instead.)
      return cls(


We will generate 40 training and 40 evaluation questions


```python
# Note: This might take sometime.
questions = dataset_generator.generate_questions_from_nodes(num=40)
print("Generated ", len(questions), " questions")
```

    Generated  40  questions


    /Users/ravithejad/Desktop/llamaindex/lib/python3.9/site-packages/llama_index/core/evaluation/dataset_generation.py:312: DeprecationWarning: Call to deprecated class QueryResponseDataset. (Deprecated in favor of `LabelledRagDataset` which should be used instead.)
      return QueryResponseDataset(queries=queries, responses=responses_dict)



```python
questions[10:15]
```




    ['What is the estimated relative human dependence on marine ecosystems for coastal protection, nutrition, fisheries economic benefits, and overall, as depicted in Figure 3.1?',
     'What are the limitations of the overall index mentioned in the context, and how were values for reference regions computed?',
     'What are the primary non-climate drivers that alter marine ecosystems and their services, as mentioned in the context?',
     'What are the main challenges in detecting and attributing climate impacts on marine-dependent human systems, according to the provided context?',
     'What new insights have been gained from experimental evidence about evolutionary adaptation, particularly in relation to eukaryotic organisms and their limited adaptation options to climate change, as mentioned in Section 3.3.4 of the IPCC AR6 WGII Chapter 03?']




```python
with open("train_questions.txt", "w") as f:
    for question in questions:
        f.write(question + "\n")
```

Now, lets generate questions on a completely different set of documents, in order to create our eval dataset.


```python
dataset_generator = DatasetGenerator.from_documents(
    documents[80:],
    question_gen_query=question_gen_query,
    llm=mistral_small,
)
```

    /Users/ravithejad/Desktop/llamaindex/lib/python3.9/site-packages/llama_index/core/evaluation/dataset_generation.py:215: DeprecationWarning: Call to deprecated class DatasetGenerator. (Deprecated in favor of `RagDatasetGenerator` which should be used instead.)
      return cls(



```python
# Note: This might take sometime.
questions = dataset_generator.generate_questions_from_nodes(num=40)
print("Generated ", len(questions), " questions")
```

    Generated  40  questions


    /Users/ravithejad/Desktop/llamaindex/lib/python3.9/site-packages/llama_index/core/evaluation/dataset_generation.py:312: DeprecationWarning: Call to deprecated class QueryResponseDataset. (Deprecated in favor of `LabelledRagDataset` which should be used instead.)
      return QueryResponseDataset(queries=queries, responses=responses_dict)



```python
with open("eval_questions.txt", "w") as f:
    for question in questions:
        f.write(question + "\n")
```

## Initial Eval with `open-mistral-7b` Query Engine

For this eval, we will be using the [`ragas` evaluation library](https://github.com/explodinggradients/ragas).

Ragas has a ton of evaluation metrics for RAG pipelines, and you can read about them [here](https://github.com/explodinggradients/ragas/blob/main/docs/metrics.md).

For this notebook, we will be using the following two metrics

- `answer_relevancy` - This measures how relevant is the generated answer to the prompt. If the generated answer is incomplete or contains redundant information the score will be low. This is quantified by working out the chance of an LLM generating the given question using the generated answer. Values range (0,1), higher the better.
- `faithfulness` - This measures the factual consistency of the generated answer against the given context. This is done using a multi step paradigm that includes creation of statements from the generated answer followed by verifying each of these statements against the context. The answer is scaled to (0,1) range. Higher the better.


```python
questions = []
with open("eval_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())
```


```python
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.mistralai import MistralAIEmbedding

# limit the context window to 2048 tokens so that refine is used
from llama_index.core import Settings

Settings.context_window = 2048
Settings.llm = open_mistral
Settings.embed_model = MistralAIEmbedding()

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(similarity_top_k=2)
```


```python
contexts = []
answers = []

for question in questions:
    response = query_engine.query(question)
    contexts.append([x.node.get_content() for x in response.source_nodes])
    answers.append(str(response))
```


```python
# We will use OpenAI LLM for evaluation using RAGAS

os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI API KEY>"
```


```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

ds = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
)

result = evaluate(ds, [answer_relevancy, faithfulness])
```

    Evaluating: 100%|██████████| 80/80 [01:12<00:00,  1.10it/s]


Let's check the results before finetuning.


```python
print(result)
```

    {'answer_relevancy': 0.8151, 'faithfulness': 0.8360}


## `mistral-large-latest` to Collect Training Data

Here, we use `mistral-large-latest` to collect data that we want `open-mistral-7b` to finetune on.


```python
from llama_index.llms.mistralai import MistralAI
from llama_index.finetuning.callbacks import MistralAIFineTuningHandler
from llama_index.core.callbacks import CallbackManager

finetuning_handler = MistralAIFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])

llm = MistralAI(model="mistral-large-latest", temperature=0.1)
llm.callback_manager = callback_manager
```


```python
questions = []
with open("train_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())
```


```python
from llama_index.core import VectorStoreIndex

Settings.embed_model = MistralAIEmbedding()
Settings.llm = llm

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(similarity_top_k=2)
```

[HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK", ..., HTTP Request: POST https://api.mistral.ai/v1/embeddings "HTTP/1.1 200 OK"]



```python
from tqdm import tqdm

contexts = []
answers = []

for question in tqdm(questions, desc="Processing questions"):
    response = query_engine.query(question)
    contexts.append(
        "\n".join([x.node.get_content() for x in response.source_nodes])
    )
    answers.append(str(response))
```

[Processing questions:   0%|          | 0/40 [00:00<?, ?it/s], ..., Processing questions: 100%|██████████| 40/40 [03:30<00:00,  5.25s/it]]



```python
import json
from typing import List


def convert_data_jsonl_format(
    questions: List[str],
    contexts: List[str],
    answers: List[str],
    output_file: str,
) -> None:
    with open(output_file, "w") as outfile:
        for context, question, answer in zip(contexts, questions, answers):
            message_dict = {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "You are a helpful assistant to answer user queries based on provided context.",
                    },
                    {
                        "role": "user",
                        "content": f"context: {context} \n\n question: {question}",
                    },
                    {"role": "assistant", "content": answer},
                ]
            }
            # Write the JSON object in a single line
            json.dump(message_dict, outfile)
            # Add a newline character after each JSON object
            outfile.write("\n")
```


```python
convert_data_jsonl_format(questions, contexts, answers, "training.jsonl")
```

## Create `MistralAIFinetuneEngine`

We create an `MistralAIFinetuneEngine`: the finetune engine will take care of launching a finetuning job, and returning an LLM model that you can directly plugin to the rest of LlamaIndex workflows.

We use the default constructor, but we can also directly pass in our finetuning_handler into this engine with the `from_finetuning_handler` class method.


```python
from llama_index.llms.mistralai import MistralAI
from llama_index.finetuning.callbacks import MistralAIFineTuningHandler
from llama_index.core.callbacks import CallbackManager
from llama_index.finetuning.mistralai import MistralAIFinetuneEngine

# Wandb for monitorning the training logs
wandb_integration_dict = {
    "project": "mistralai",
    "run_name": "finetuning",
    "api_key": "3a486c0d4066396b07b0ebf6826026c24e981f37",
}

finetuning_engine = MistralAIFinetuneEngine(
    base_model="open-mistral-7b",
    training_path="training.jsonl",
    # validation_path="<validation file>", # validation file is optional
    verbose=True,
    training_steps=5,
    learning_rate=0.0001,
    wandb_integration_dict=wandb_integration_dict,
)
```


```python
# starts the finetuning of open-mistral-7b

finetuning_engine.finetune()
```


```python
# This will show the current status of the job - 'RUNNING'
finetuning_engine.get_current_job()
```

    INFO:httpx:HTTP Request: GET https://api.mistral.ai/v1/fine_tuning/jobs/19f5943a-3000-4568-b227-e45c36ac15f1 "HTTP/1.1 200 OK"
    HTTP Request: GET https://api.mistral.ai/v1/fine_tuning/jobs/19f5943a-3000-4568-b227-e45c36ac15f1 "HTTP/1.1 200 OK"





    DetailedJob(id='19f5943a-3000-4568-b227-e45c36ac15f1', hyperparameters=TrainingParameters(training_steps=5, learning_rate=0.0001), fine_tuned_model=None, model='open-mistral-7b', status='RUNNING', job_type='FT', created_at=1718613228, modified_at=1718613229, training_files=['07270085-65e6-441e-b99d-ef6f75dd5a30'], validation_files=[], object='job', integrations=[WandbIntegration(type='wandb', project='mistralai', name=None, run_name='finetuning')], events=[Event(name='status-updated', data={'status': 'RUNNING'}, created_at=1718613229), Event(name='status-updated', data={'status': 'QUEUED'}, created_at=1718613228)], checkpoints=[], estimated_start_time=None)




```python
# This will show the current status of the job - 'SUCCESS'
finetuning_engine.get_current_job()
```

    INFO:httpx:HTTP Request: GET https://api.mistral.ai/v1/fine_tuning/jobs/19f5943a-3000-4568-b227-e45c36ac15f1 "HTTP/1.1 200 OK"
    HTTP Request: GET https://api.mistral.ai/v1/fine_tuning/jobs/19f5943a-3000-4568-b227-e45c36ac15f1 "HTTP/1.1 200 OK"





    DetailedJob(id='19f5943a-3000-4568-b227-e45c36ac15f1', hyperparameters=TrainingParameters(training_steps=5, learning_rate=0.0001), fine_tuned_model='ft:open-mistral-7b:35c6fa92:20240617:19f5943a', model='open-mistral-7b', status='SUCCESS', job_type='FT', created_at=1718613228, modified_at=1718613306, training_files=['07270085-65e6-441e-b99d-ef6f75dd5a30'], validation_files=[], object='job', integrations=[WandbIntegration(type='wandb', project='mistralai', name=None, run_name='finetuning')], events=[Event(name='status-updated', data={'status': 'SUCCESS'}, created_at=1718613306), Event(name='status-updated', data={'status': 'RUNNING'}, created_at=1718613229), Event(name='status-updated', data={'status': 'QUEUED'}, created_at=1718613228)], checkpoints=[], estimated_start_time=None)




```python
ft_llm = finetuning_engine.get_finetuned_model(temperature=0.1)
```

    INFO:httpx:HTTP Request: GET https://api.mistral.ai/v1/fine_tuning/jobs/19f5943a-3000-4568-b227-e45c36ac15f1 "HTTP/1.1 200 OK"
    HTTP Request: GET https://api.mistral.ai/v1/fine_tuning/jobs/19f5943a-3000-4568-b227-e45c36ac15f1 "HTTP/1.1 200 OK"
    INFO:llama_index.finetuning.mistralai.base:status of the job_id: 19f5943a-3000-4568-b227-e45c36ac15f1 is SUCCESS
    status of the job_id: 19f5943a-3000-4568-b227-e45c36ac