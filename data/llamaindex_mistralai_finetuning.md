# Fine-Tuning MistralAI Models Using the Finetuning API and LlamaIndex

This guide demonstrates how to fine-tune the `open-mistral-7b` model using MistralAI's finetuning API via LlamaIndex. The goal is to distill knowledge from the larger `mistral-large-latest` model by generating synthetic training data with it, then using that data to fine-tune the smaller `open-mistral-7b` model.

We will generate training and evaluation datasets from different sections of a document, fine-tune the model, and evaluate its performance before and after fine-tuning using the `ragas` evaluation library.

## Prerequisites

First, install the required packages.

```bash
pip install llama-index-finetuning
pip install llama-index-finetuning-callbacks
pip install llama-index-llms-mistralai
pip install llama-index-embeddings-mistralai
pip install llama-index pypdf sentence-transformers ragas
```

Since this workflow uses async operations within a notebook environment, apply `nest_asyncio`.

```python
import nest_asyncio
nest_asyncio.apply()
```

## Step 1: Set Up API Keys

Set your MistralAI and OpenAI API keys as environment variables. The OpenAI key is required for evaluation with `ragas`.

```python
import os

os.environ["MISTRAL_API_KEY"] = "<YOUR MISTRALAI API KEY>"
os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI API KEY>"
```

## Step 2: Download and Load the Data

Download the PDF document that will serve as the source for generating questions and answers.

```bash
curl https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter03.pdf --output IPCC_AR6_WGII_Chapter03.pdf
```

Load the document using LlamaIndex.

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()
```

## Step 3: Initialize LLMs and Embedding Model

Set up the three MistralAI models we'll use:
- `open-mistral-7b`: The base model to be fine-tuned.
- `mistral-small-latest`: Used to generate unbiased synthetic questions.
- `mistral-large-latest`: Used to generate high-quality training answers (knowledge distillation).
- `MistralAIEmbedding`: For creating embeddings.

```python
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding

open_mistral = MistralAI(model="open-mistral-7b", temperature=0.1)
mistral_small = MistralAI(model="mistral-small-latest", temperature=0.1)
embed_model = MistralAIEmbedding()
```

## Step 4: Generate Training and Evaluation Questions

We'll generate 40 training questions from the first 80 document nodes and 40 evaluation questions from the remaining nodes. This separation ensures the evaluation set is distinct from the training data.

Define the prompt for question generation.

```python
question_gen_query = (
    "You are a Teacher/ Professor. Your task is to setup "
    "a quiz/examination. Using the provided context, formulate "
    "a single question that captures an important fact from the "
    "context. Restrict the question to the context information provided."
    "You should generate only question and nothing else."
)
```

### 4.1 Generate Training Questions

```python
from llama_index.core.evaluation import DatasetGenerator

dataset_generator = DatasetGenerator.from_documents(
    documents[:80],
    question_gen_query=question_gen_query,
    llm=mistral_small,
)

# This might take some time
train_questions = dataset_generator.generate_questions_from_nodes(num=40)
print(f"Generated {len(train_questions)} training questions")

# Save the training questions
with open("train_questions.txt", "w") as f:
    for question in train_questions:
        f.write(question + "\n")
```

### 4.2 Generate Evaluation Questions

```python
dataset_generator = DatasetGenerator.from_documents(
    documents[80:],
    question_gen_query=question_gen_query,
    llm=mistral_small,
)

# This might take some time
eval_questions = dataset_generator.generate_questions_from_nodes(num=40)
print(f"Generated {len(eval_questions)} evaluation questions")

# Save the evaluation questions
with open("eval_questions.txt", "w") as f:
    for question in eval_questions:
        f.write(question + "\n")
```

## Step 5: Establish a Baseline Evaluation with `open-mistral-7b`

Before fine-tuning, let's evaluate the base `open-mistral-7b` model's performance on the evaluation questions using the `ragas` library. We'll measure **answer relevancy** and **faithfulness**.

First, load the evaluation questions.

```python
questions = []
with open("eval_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())
```

Set up a query engine using the base model.

```python
from llama_index.core import VectorStoreIndex, Settings

# Configure settings
Settings.context_window = 2048  # Limit context to force refine usage
Settings.llm = open_mistral
Settings.embed_model = MistralAIEmbedding()

# Build index and query engine
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=2)
```

Run the evaluation questions through the query engine to collect answers and source contexts.

```python
contexts = []
answers = []

for question in questions:
    response = query_engine.query(question)
    contexts.append([x.node.get_content() for x in response.source_nodes])
    answers.append(str(response))
```

Evaluate using `ragas`.

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
print(result)
```

**Expected output (example):**
```
{'answer_relevancy': 0.8151, 'faithfulness': 0.8360}
```

Record these baseline scores for later comparison.

## Step 6: Generate Training Data with `mistral-large-latest`

Now, we'll use the more powerful `mistral-large-latest` model to generate high-quality answers for the training questions. This creates the synthetic dataset for knowledge distillation.

Set up the LLM with a fine-tuning handler to capture the prompt-completion pairs.

```python
from llama_index.finetuning.callbacks import MistralAIFineTuningHandler
from llama_index.core.callbacks import CallbackManager

finetuning_handler = MistralAIFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])

llm = MistralAI(model="mistral-large-latest", temperature=0.1)
llm.callback_manager = callback_manager
```

Load the training questions.

```python
questions = []
with open("train_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())
```

Configure the query engine to use `mistral-large-latest`.

```python
Settings.llm = llm
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=2)
```

Generate answers and their source contexts.

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

Format the generated data into the JSONL format required by MistralAI's fine-tuning API.

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
            json.dump(message_dict, outfile)
            outfile.write("\n")

convert_data_jsonl_format(questions, contexts, answers, "training.jsonl")
```

## Step 7: Fine-Tune `open-mistral-7b`

Create a `MistralAIFinetuneEngine` to manage the fine-tuning job. Optionally, integrate with Weights & Biases for monitoring.

```python
from llama_index.finetuning.mistralai import MistralAIFinetuneEngine

# Optional WandB integration
wandb_integration_dict = {
    "project": "mistralai",
    "run_name": "finetuning",
    "api_key": "<YOUR_WANDB_API_KEY>",  # Replace with your key
}

finetuning_engine = MistralAIFinetuneEngine(
    base_model="open-mistral-7b",
    training_path="training.jsonl",
    # validation_path="<validation file>", # Optional validation file
    verbose=True,
    training_steps=5,
    learning_rate=0.0001,
    wandb_integration_dict=wandb_integration_dict,
)
```

Start the fine-tuning job.

```python
finetuning_engine.finetune()
```

Check the job status. It will initially show `RUNNING` and eventually transition to `SUCCESS`.

```python
job_info = finetuning_engine.get_current_job()
print(job_info.status)
```

Once the job status is `SUCCESS`, retrieve the fine-tuned model.

```python
ft_llm = finetuning_engine.get_finetuned_model(temperature=0.1)
```

## Step 8: Evaluate the Fine-Tuned Model

Now, evaluate the fine-tuned model on the same evaluation questions to measure improvement.

Update the query engine to use the fine-tuned model.

```python
Settings.llm = ft_llm
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=2)
```

Run the evaluation again.

```python
contexts_ft = []
answers_ft = []

for question in questions:
    response = query_engine.query(question)
    contexts_ft.append([x.node.get_content() for x in response.source_nodes])
    answers_ft.append(str(response))

ds_ft = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers_ft,
        "contexts": contexts_ft,
    }
)

result_ft = evaluate(ds_ft, [answer_relevancy, faithfulness])
print("Fine-tuned model results:", result_ft)
```

Compare the scores with the baseline.

```python
print("Baseline scores:", result)
print("Fine-tuned scores:", result_ft)
```

## Conclusion

You have successfully fine-tuned the `open-mistral-7b` model using synthetic data generated by `mistral-large-latest`. By comparing the `ragas` evaluation metrics before and after fine-tuning, you can quantify the improvement in answer relevancy and faithfulness.

This workflow demonstrates a practical knowledge distillation pipeline using LlamaIndex and MistralAI's fine-tuning API, enabling you to create smaller, more efficient models that retain the knowledge of larger counterparts.