# Agentic RAG: Turbocharge Your RAG with Query Reformulation and Self-Query

_Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_

> **Prerequisite:** This is an advanced tutorial. It is recommended to have prior knowledge from [this other cookbook](advanced_rag) first.

## Introduction

Retrieval-Augmented Generation (RAG) enhances an LLM's ability to answer questions by grounding its responses in information retrieved from a knowledge base. This reduces confabulations and allows for domain-specific knowledge integration.

However, standard RAG has two primary limitations:
1. It performs only a single retrieval step—poor results lead to poor generation.
2. Semantic similarity is computed using the raw user query, which may be suboptimal (e.g., a question vs. an affirmative statement in the documents).

We can overcome these by building a **RAG agent**—an agent equipped with a retriever tool. This agent will:
- **Reformulate the query** to better match document phrasing (similar to [HyDE](https://huggingface.co/papers/2212.10496)).
- **Critique and re-retrieve** information if needed (similar to [Self-Query](https://docs.llamaindex.ai/en/stable/examples/evaluation/RetryQuery/)).

Let's build this system step by step.

## Setup and Installation

First, install the required dependencies.

```bash
pip install pandas langchain langchain-community sentence-transformers faiss-cpu smolagents --upgrade -q
```

Next, log in to access the Hugging Face Inference API.

```python
from huggingface_hub import notebook_login

notebook_login()
```

## Step 1: Load and Prepare the Knowledge Base

We'll use a dataset containing Hugging Face documentation pages in markdown format.

```python
import datasets

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
```

## Step 2: Process Documents and Create a Vector Database

We'll use LangChain to split the documents and create a FAISS vector store. The embedding model `thenlper/gte-small` is chosen for its strong performance.

```python
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# Convert dataset to LangChain Documents
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained("thenlper/gte-small"),
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Split documents and keep only unique chunks
print("Splitting documents...")
docs_processed = []
unique_texts = {}
for doc in tqdm(source_docs):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)

# Create vector store
print("Embedding documents... This may take a few minutes.")
embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
vectordb = FAISS.from_documents(
    documents=docs_processed,
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)
```

## Step 3: Build a Custom Retriever Tool

We need a tool that the agent can call to retrieve documents. We'll subclass the `Tool` class from `smolagents` to integrate our vector database.

```python
from smolagents import Tool
from langchain_core.vectorstores import VectorStore

class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )

        return "\nRetrieved documents:\n" + "".join(
            [
                f"===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )
```

## Step 4: Initialize the Agent

We'll create an agent powered by `meta-llama/Llama-3.1-70B-Instruct` via the Hugging Face Inference API. The agent uses the `ToolCallingAgent` class from `smolagents`.

```python
from smolagents import InferenceClientModel, ToolCallingAgent

model = InferenceClientModel("meta-llama/Llama-3.1-70B-Instruct")
retriever_tool = RetrieverTool(vectordb)
agent = ToolCallingAgent(tools=[retriever_tool], model=model)
```

The agent uses a default system prompt that instructs the LLM to reason step-by-step and generate tool calls in JSON format. When `agent.run()` is called, it handles the LLM calls, tool execution, and looping until a final answer is produced.

## Step 5: Test the Agent

Let's ask a question to see the agent in action.

```python
agent_output = agent.run("How can I push a model to the Hub?")
print("Final output:")
print(agent_output)
```

**Example Output:**
```
To push a model to the Hub, you can use the push_to_hub() method after training. You can also use the PushToHubCallback to upload checkpoints regularly during a longer training run. Additionally, you can push the model up to the hub using the api.upload_folder() method.
```

## Step 6: Evaluate Agentic RAG vs. Standard RAG

Now, let's compare our agentic RAG system against a standard RAG setup using an LLM judge.

### 6.1 Load Evaluation Dataset

```python
eval_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval", split="train")
```

### 6.2 Reduce Agent Verbosity

```python
import logging
agent.logger.setLevel(logging.WARNING)
```

### 6.3 Run Agentic RAG Evaluation

We'll run the agent on each evaluation question with an enhanced prompt that encourages multiple retrievals and query reformulation.

```python
outputs_agentic_rag = []

for example in tqdm(eval_dataset):
    question = example["question"]

    enhanced_question = f"""Using the information contained in your knowledge base, which you can access with the 'retriever' tool,
give a comprehensive answer to the question below.
Respond only to the question asked, response should be concise and relevant to the question.
If you cannot find information, do not give up and try calling your retriever again with different arguments!
Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries.
Your queries should not be questions but affirmative form sentences: e.g. rather than "How do I load a model from the Hub in bf16?", query should be "load a model from the Hub bf16 weights".

Question:
{question}"""
    answer = agent.run(enhanced_question)
    print("=======================================================")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f'True answer: {example["answer"]}')

    results_agentic = {
        "question": question,
        "true_answer": example["answer"],
        "source_doc": example["source_doc"],
        "generated_answer": answer,
    }
    outputs_agentic_rag.append(results_agentic)
```

### 6.4 Run Standard RAG Evaluation

For the standard RAG baseline, we'll retrieve documents using the raw user query and then generate an answer with a separate LLM.

```python
from huggingface_hub import InferenceClient

reader_llm = InferenceClient("Qwen/Qwen2.5-72B-Instruct")
outputs_standard_rag = []

for example in tqdm(eval_dataset):
    question = example["question"]
    context = retriever_tool(question)

    prompt = f"""Given the question and supporting documents below, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If you cannot find information, do not give up and try calling your retriever again with different arguments!

Question:
{question}

{context}
"""
    messages = [{"role": "user", "content": prompt}]
    answer = reader_llm.chat_completion(messages).choices[0].message.content

    print("=======================================================")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f'True answer: {example["answer"]}')

    results_agentic = {
        "question": question,
        "true_answer": example["answer"],
        "source_doc": example["source_doc"],
        "generated_answer": answer,
    }
    outputs_standard_rag.append(results_agentic)
```

### 6.5 Define the Evaluation Prompt

We'll use a structured prompt for the LLM judge, based on best practices from evaluation cookbooks.

```python
EVALUATION_PROMPT = """You are a fair evaluator language model.

You will be given an instruction, a response to evaluate, a reference answer that gets a score of 3, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 3. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 3}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.
5. Do not score conciseness: a correct answer that covers the question should receive max score, even if it contains additional useless information.

The instruction to evaluate:
{instruction}

Response to evaluate:
{response}

Reference Answer (Score 3):
{reference_answer}

Score Rubrics:
[Is the response complete, accurate, and factual based on the reference answer?]
Score 1: The response is completely incomplete, inaccurate, and/or not factual.
Score 2: The response is somewhat complete, accurate, and/or factual.
Score 3: The response is completely complete, accurate, and/or factual.

Feedback:"""
```

### 6.6 Perform LLM Judge Evaluation

We'll use `meta-llama/Llama-3.1-70B-Instruct` as the judge to score both sets of answers.

```python
from huggingface_hub import InferenceClient
import pandas as pd

evaluation_client = InferenceClient("meta-llama/Llama-3.1-70B-Instruct")
results = {}

for system_type, outputs in [
    ("agentic", outputs_agentic_rag),
    ("standard", outputs_standard_rag),
]:
    for experiment in tqdm(outputs):
        eval_prompt = EVALUATION_PROMPT.format(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        messages = [
            {"role": "system", "content": "You are a fair evaluator language model."},
            {"role": "user", "content": eval_prompt},
        ]

        eval_result = evaluation_client.text_generation(
            eval_prompt, max_new_tokens=1000
        )
        try:
            feedback, score = [item.strip() for item in eval_result.split("[RESULT]")]
            experiment["eval_score_LLM_judge"] = score
            experiment["eval_feedback_LLM_judge"] = feedback
        except:
            print(f"Parsing failed - output was: {eval_result}")

    results[system_type] = pd.DataFrame.from_dict(outputs)
    results[system_type] = results[system_type].loc[~results[system_type]["generated_answer"].str.contains("Error")]
```

### 6.7 Calculate and Compare Scores

Finally, we compute the average scores for both systems.

```python
DEFAULT_SCORE = 2  # Assign average score if parsing fails

def fill_score(x):
    try:
        return int(x)
    except:
        return DEFAULT_SCORE

for system_type, outputs in [
    ("agentic", outputs_agentic_rag),
    ("standard", outputs_standard_rag),
]:
    results[system_type]["eval_score_LLM_judge_int"] = (
        results[system_type]["eval_score_LLM_judge"].fillna(DEFAULT_SCORE).apply(fill_score)
    )
    # Normalize score to 0-1 range
    results[system_type]["eval_score_LLM_judge_int"] = (results[system_type]["eval_score_LLM_judge_int"] - 1) / 2

    print(
        f"Average score for {system_type} RAG: {results[system_type]['eval_score_LLM_judge_int'].mean()*100:.1f}%"
    )
```

**Results:**
```
Average score for agentic RAG: 86.9%
Average score for standard RAG: 73.1%
```

## Conclusion

The agentic RAG system achieves an **86.9%** score, a **14% improvement** over the standard RAG baseline (73.1%). For reference, using Llama-3-70B without any knowledge base scored only 36%.

This demonstrates that a simple agent setup—empowered with query reformulation and self-query capabilities—can significantly enhance RAG performance. The agent intelligently reformulates queries and performs multiple retrievals, leading to more accurate and complete answers.

🚀 **You've successfully built and evaluated an advanced Agentic RAG system!**