# Scaling Test-Time Compute for Longer Thinking in LLMs

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

🚨 **WARNING**: This tutorial is **resource-intensive** and requires substantial computational power. If you’re running this in **Colab**, it will utilize an **A100 GPU**.

---

In this guide, you'll learn how to extend the inference time for an **Instruct LLM system** using **test-time compute** to solve more challenging problems, such as **complex math problems**. This approach, inspired by [**OpenAI o1-o3 models**](https://openai.com/index/learning-to-reason-with-llms/), demonstrates that **longer reasoning time** during inference can enhance model performance.

This technique builds on experiments shared in [this **blog post**](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute), which show that smaller models, like the **1B** and **3B Llama Instruct models**, can outperform much larger ones on the **MATH-500 benchmark** when given enough **"time to think"**. Recent research from [DeepMind](https://arxiv.org/abs/2408.03314) suggests that **test-time compute** can be scaled optimally through strategies like iterative self-refinement or using a reward model.

The blog introduces a [**new repository**](https://github.com/huggingface/search-and-learn) for running these experiments. In this recipe, we'll focus on building a **small chatbot** that engages in **longer reasoning** to tackle **harder problems** using small open models.

## Prerequisites

Let’s start by installing the [search-and-learn](https://github.com/huggingface/search-and-learn) repository! 🚀  
This repo is designed to replicate the experimental results and is not a Python pip package. However, we can still use it to generate our system. To do so, we’ll need to install it from source.

```bash
!git clone https://github.com/huggingface/search-and-learn
%cd search-and-learn
!pip install -e '.[dev]'
```

Log in to Hugging Face to access [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), as it is a gated model! 🗝️  
If you haven't previously requested access, you'll need to submit a request before proceeding.

```python
from huggingface_hub import notebook_login

notebook_login()
```

## Step 1: Setup the Large Language Model (LLM) and the Process Reward Model (PRM)

As illustrated in the diagram, the system consists of an LLM that generates intermediate answers based on user input, a [PRM model](https://huggingface.co/papers/2211/14275) that evaluates and scores these answers, and a search strategy that uses the PRM feedback to guide the subsequent steps in the search process until reaching the final answer.

Let’s begin by initializing each model. For the LLM, we’ll use the [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) model, and for the PRM, we’ll use the [RLHFlow/Llama3.1-8B-PRM-Deepseek-Data](https://huggingface.co/RLHFlow/Llama3.1-8B-PRM-Deepseek-Data) model.

```python
import torch
from vllm import LLM
from sal.models.reward_models import RLHFFlow

model_path="meta-llama/Llama-3.2-1B-Instruct"
prm_path="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"

llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.5,  # Utilize 50% of GPU memory
    enable_prefix_caching=True,  # Optimize repeated prefix computations
    seed=42,                     # Set seed for reproducibility
)

prm = RLHFFlow(prm_path)
```

## Step 2: Define the Question and Search Strategy

Now that we've set up the LLM and PRM, let's proceed by defining the question, selecting a search strategy to retrieve relevant information, and calling the pipeline to process the question through the models.

1.  **Instantiate the Question**: In this step, we define the input question that the system will answer.
2.  **Search Strategy**: The system currently supports the following search strategies: `best_of_n`, `beam_search`, and `dvts`. For this example, we'll use `best_of_n`, but you can easily switch to any of the other strategies based on your needs. We need to define some configuration parameters for the configuration of the search strategy. You can check the full list [here](https://github.com/huggingface/search-and-learn/blob/main/src/sal/config.py).
3.  **Call the Pipeline**: With the question and search strategy in place, we’ll call the inference pipeline, processing the inputs through both the LLM and PRM to generate the final answer.

The first step is to clearly define the question that the system will answer. This ensures that we have a precise task for the model to tackle.

```python
question_text = 'Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$'
input_batch = {"problem": [question_text]}
```

Next, we define the configuration, including parameters like the number of candidate answers `(N)`, and choose the search strategy that will be used. The search strategy dictates how we explore the potential answers. In this case, we'll use `best_of_n`.

With the question and configuration in place, we use the selected search strategy to generate multiple candidate answers. These candidates are evaluated based on their relevance and quality and the final answer is returned.

```python
from sal.config import Config
from sal.search import beam_search, best_of_n, dvts

config = Config()
config.n=32 # Number of answers to generate during the search

search_result = best_of_n(x=input_batch, config=config, llm=llm, prm=prm)
```

## Step 3: Display and Format the Final Result

Once the pipeline has processed the question through the LLM and PRM, we can display the final result. This result will be the model's output after considering the intermediate answers and scoring them using the PRM.

Here's how to display the final answer:

```python
search_result['pred'][0]
```

The model’s output might include special tokens, such as `<|start_header_id|>` or `<|end_header_id|>`. To make the answer more readable, we can safely remove them before displaying it to the end user.

```python
formatted_output = search_result['pred'][0].replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
formatted_output
```

After removing any special tokens, we can display the final answer to the user. Since the answer is based on markdown, it can be rendered properly by displaying it as markdown.

```python
from IPython.display import display, Markdown

display(Markdown(formatted_output))
```

## Step 4: Create a Reusable Pipeline Method

Now, let's create a method that encapsulates the entire pipeline. This will allow us to easily reuse the process in future applications, making it efficient and modular.

By combining the LLM, PRM, search strategy, and result display, we can simplify the workflow and ensure that it’s reusable for other tasks or questions.

We simplify the workflow, ensuring that it’s reusable for different tasks or questions. Additionally, we’ll track the time spent on each method so that we can **understand the practical implications** of using each strategy and configuration.

Here’s how we can structure the method:

```python
import time

def generate_with_search_and_learn(question, config, llm, prm, method='best_of_n'):
    """
    Generate an answer for a given question using the search-and-learn pipeline.

    Args:
    - question (str): The input question to generate an answer for.
    - config (Config): Configuration object containing parameters for search strategy.
    - llm (LLM): Pretrained large language model used for generating answers.
    - prm (RLHFFlow): Process reward model used for evaluating answers.
    - method (str): Search strategy to use. Options are 'best_of_n', 'beam_search', 'dvts'. Default is 'best_of_n'.

    Returns:
    - str: The formatted output after processing the question.
    """
    batch = {"problem": [question]}

    start_time = time.time()
    if method == 'best_of_n':
      result = best_of_n(x=batch, config=config, llm=llm, prm=prm)
    elif method == 'beam_search':
      result = beam_search(examples=batch, config=config, llm=llm, prm=prm)
    elif method == 'dvts':
      result = dvts(examples=batch, config=config, llm=llm, prm=prm)

    elapsed_time = time.time() - start_time
    print(f"\nFinished in {elapsed_time:.2f} seconds\n")

    tokenizer = llm.get_tokenizer()
    total_tokens = 0
    for completion in result['completions']:
        for comp in  completion:
            output_tokens = tokenizer.encode(comp)
            total_tokens += len(output_tokens)

    print(f"Total tokens in all completions: {total_tokens}")

    formatted_output = result['pred'][0].replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "").strip()
    return formatted_output
```

## Step 5: Compare Thinking Time for Each Strategy

Let’s compare the **thinking time** of three methods: `best_of_n`, `beam_search`, and `dvts`. Each method is evaluated using the same number of answers during the search process, measuring the time spent thinking in seconds and the number of generated tokens.

In the results below, the `best_of_n` method shows the least thinking time, while the `dvts` method takes the most time. However, `best_of_n` generates more tokens due to its simpler search strategy.

| **Method**      | **Number of Answers During Search** | **Thinking Time (Seconds)** | **Generated Tokens** |
|------------------|-------------------------------------|-----------------------------|-----------------------|
| **best_of_n**    | 8                                   | 3.54                        | 3087                  |
| **beam_search**  | 8                                   | 10.06                       | 2049                  |
| **dvts**         | 8                                   | 8.46                        | 2544                  |

This comparison illustrates the trade-offs between the strategies, balancing time spent thinking and the complexity of the search process.

### 5.1 Test the `best_of_n` Strategy

We’ll begin by using the `best_of_n` strategy. Here’s how to track the thinking time for this method:

```python
question = 'Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$'

config.n=8

formatted_output = generate_with_search_and_learn(question=question, config=config, llm=llm, prm=prm, method='best_of_n')
display(Markdown(formatted_output))
```

### 5.2 Test the `beam_search` Strategy

Now, let's try using the `beam_search` strategy.

```python
config.n=8
# beam search specific
config.sort_completed=True
config.filter_duplicates=True

formatted_output = generate_with_search_and_learn(question=question, config=config, llm=llm, prm=prm, method='beam_search')
display(Markdown(formatted_output))
```