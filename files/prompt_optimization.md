# Automated Prompt Optimization

- ❌ Prompt engineering... sucks. It's a non-standard process, heavily relying on trial and error and difficult to standardize
- 🤩 Luckily, we can automate it using ✨prompt optimization✨, investigated in recent works such as [_Self-Supervised Prompt Optimization_](https://arxiv.org/pdf/2502.06855)
- 🎯 In its essence, Prompt Optimization (PO) consists in the process of taking a prompt aiming at performing a certain task and iteratively refining it to make it better for the specific problem tackled.
- ✅ This notebook gives an overview of how to use PO with Mistral models

# Problem setting

- You have put up a form, and collected many more answers than the ones you can read.
- Your survey got popular---very popular, 😅---and need to sift through the answers. To keep things accessibly, we allowed (and will continue to!) responses using plain text.
- Filtering is therefore _impossible_. Still, you need some strategies to sift through the applications received to identify the most promising profiles.
- Let's define a few prompts to process answers and output answers we can filter on effectively.

### Task prompts

- Let's define a few prompts to process answers
- These prompts are purposely not optimized, and rather serve as an example of something quick and dirty we wish to work with.
- For this example, we will consider answers collected as part of the applications for our [Ambassadorship Program](https://docs.mistral.ai/guides/contribute/ambassador/)


```python
# overarching prompt, giving context
context = (
    "I am working on recruiting people to advocate about the products of an AI company. "
    "The position in in close contact with the DevRel team, and we are looking at having people "
    "share on their own personal social media more about the company and its products. "
    "The company I work at produces Large Language Models and is very followed, "
    "therefore I got a sheer amount of applications that I need to process "
    "very soon. I won't be able to process them by hand, and there is little structure in the "
    "form that we sent out to applicants. Therefore, I am expecting you to assist me into processing the "
    "information these people gave to make it much more structured. This means that you do read "
    "what applicants declared and extract key information based on the context of the question asked."
)

# classifying job titles
job_prompt = lambda job_title: (
    "Your task is to provide me with a direct classification of the person's job title into one of 4 categories. "
    "The categories you can decide are always: 'RESEARCH', 'ENGINEERING', 'BUSINESS', 'FOUNDER'. "
    "There is no possibility for mixed assignments. You always assign one and one only category to each subject. "
    "When in doubt, assign to 'OTHER'. You must strictly adhere to the categories I have mentioned, and nothing more. "
    "This means that you cannot use any other output apart from 'RESEARCH', 'ENGINEERING', 'BUSINESS', 'FOUNDER', 'OTHER'. "
    "Keep your answer very, very concise. Don't give context on your answer. As a matter of fact, only answer with one word "
    "based on the category you deem the most appropriate. Absolutely don't change this. You will be penalized if "
    "(1) you use a category outside of the ones I have mentioned and (2) you use more than 1 word in your output. "
    f"# INPUT declared title: the person job title is {job_title}"
)

# getting the location in an easy way
location_prompt = lambda location: (
    "Your task is basic. Your task is to disambiguate the respondent's answer in terms of the location used. "
    "Your output is always CITY, COUNTRY. Use always the English name of a city. Also, always use the international "
    "country code. Nothing else. For instance, if a user answered with 'Rome', you would output 'Rome, IT'. "
    "In the rare case when someone puts down multiple locations, make sure you always select the first one. Nothing more"
    f" #INPUT declared location: the respondent declared being located in {location}"
)
```

### Installing dependancies

To use SPO via MetaGPT you need to clone the repository, and move this notebook inside of it. Dependancies are not easily usable, but hacking around it is fairly straightforward 😉 

Just run:


```python
# clone the repo
!git clone https://github.com/geekan/MetaGPT

# install dependancies
!pip install -qUr MetaGPT/requirements.txt

# move inside the directory, kernel-wise
%cd MetaGPT
```

    [Cloning into 'MetaGPT'..., remote: Enumerating objects: 48797, done., remote: Counting objects: 100% (287/287), done., remote: Compressing objects: 100% (136/136), done., remote: Total 48797 (delta 195), reused 151 (delta 151), pack-reused 48510 (from 3), Receiving objects: 100% (48797/48797), 179.81 MiB | 45.07 MiB/s, done., Resolving deltas: 100% (36800/36800), done.]
    /Users/francescocapuano/Desktop/prompt-optimization/third_party/MetaGPT/MetaGPT


## Create instruction files

After having installed `metagpt`, we can perform prompt optimization creating a yaml file specifying the task tackled.

From `metagpt` [documentation](https://github.com/geekan/MetaGPT/tree/main/examples/spo), this yaml file needs the following structure:

```bash
prompt: |
  Please solve the following problem.

requirements: |
  ...

count: None

qa:
  - question: |
      ...
    answer: |
      ...

  - question: |
      ...
    answer: |
      ...
```

We will need to generate one of these template files **for each** of the prompts we are seeking to optimize. Luckily, we can do so automatically. 

Also, as the tasks we're dealing with are fairly straightforward we can spare us providing few shot examples in the form Q&As 🤩

Still, these template files offer a very straightforward way to provide real-world few-shot examples so definitely worth looking into those.


```python
from typing import Optional

def prompt_to_dict(
        prompt: str,
        requirements: Optional[str],
        questions: list[str],
        answers: list[str],
        count: Optional[int] = None,
)->dict:
    return {
        "prompt": prompt if isinstance(prompt, str) else prompt(""),
        "requirements": requirements,
        "count": count,
        "qa": [
            {
                "question": question,
                "answer": answer
            } for question, answer in zip(questions, answers)
        ]
    }
```


```python
import yaml

prompts = {
    "job": job_prompt,
    "location": location_prompt
}

requirements = [
    "The job title, categorized",
    "The location, disambiguated"
]
path = "metagpt/ext/spo/settings"  # this is the path where the template files needs to be saved

for (name, prompt), requirement in zip(prompts.items(), requirements):
    # creating template files for each prompt
    with open(f"{path}/{name}.yaml", "w") as f:
        yaml.dump(
            prompt_to_dict(
                prompt, 
                requirement,
                [""], 
                [""]
            ),
            f,
        )
```

## Creating model files

Once you created template files for the different prompts, you need to specify which models you need to use as (1) executors (2) evaluators and (3) optimizers for the different prompts.

metagpt's SPO requires you to provide these models within a specific `.yaml` file---you can use the following snippet to create these files using your own Mistral API key ([get one!](https://console.mistral.ai/api-keys)).


```python
def models_dict(
        mistral_api_key: str
    )->dict:
    return {
        "llm": {
            "api_type": "openai",
            "model": "mistral-small-latest",
            "base_url": "https://api.mistral.ai/v1/",
            "api_key": mistral_api_key,
            "temperature": 0
        },
        "models": {
            "mistral-small-latest": {
                "api_type": "openai",
                "base_url": "https://api.mistral.ai/v1/",
                "api_key": mistral_api_key,
                "temperature": 0
            },
            "mistral-large-latest": {
                "api_type": "openai",
                "base_url": "https://api.mistral.ai/v1/",
                "api_key": mistral_api_key,
                "temperature": 0
            }
        }
    }
```


```python
path = "config/config2.yaml" # saving the models file here

MISTRAL_API_KEY = "ADD YOU KEY HERE"  # your api key

with open(path, "w") as f:
    yaml.dump(models_dict(MISTRAL_API_KEY), f)
```

**We're good! 🎉** 

Once you have (1) template files for your candidate prompts and (2) a `models.yaml` file to identify the different models you wish to use, we can get start running rounds and optimizing the prompts 😊

### A little hack: jupyter notebooks don't really work with `asyncio` 🫠

...if only jupyter notebooks worked well with `asyncio` 😂 The little hack here is to export the code you need to run prompt optimization to a `.py` file and then run that one using CLI-like instructions.

Here we are only creating one file for the job title extraction prompt. Exporting these prompt optimization processes to different files also allows for parallel execution (💨, right?). For the sake of demonstration, we are only showing how to optimize one prompt (job extraction), but you can easily switch this to other prompts yourself.


```python
%%writefile spo.py

from metagpt.ext.spo.components.optimizer import PromptOptimizer
from metagpt.ext.spo.utils.llm_client import SPO_LLM

# Initialize LLM settings
SPO_LLM.initialize(
    # same temperature settings as metagpt's default!
    optimize_kwargs={
        "model": "mistral-large-latest", 
        "temperature": 0.6
    },
    evaluate_kwargs={
        "model": "mistral-small-latest", 
        "temperature": 0.3
    },
    execute_kwargs={
        "model": "mistral-small-latest", 
        "temperature": 0
    }
)

template_name = "job.yaml"  # change this for each prompt!

# Create and run optimizer
optimizer = PromptOptimizer(
    optimized_path="workspace",  # Output directory
    initial_round=1,  # Starting round
    max_rounds=5,  # Maximum optimization rounds
    template=template_name,  # Template file - Change this for each prompt!
    name="Mistral-Prompt-Opt",  # Project name
)

optimizer.optimize()
```

    Overwriting spo.py


Now, let's run prompt optimization ☀️


```python
!python spo.py
```

    [2025-04-19 15:33:24.300 | INFO    | metagpt.const:get_metagpt_package_root:15 - Package root set to /Users/francescocapuano/Desktop/prompt-optimization/third_party/MetaGPT/MetaGPT, 2025-04-19 15:33:24.300 | INFO    | metagpt.const:get_metagpt_package_root:15 - Package root set to /Users/francescocapuano/Desktop/prompt-optimization/third_party/MetaGPT/MetaGPT, 2025-04-19 15:33:25.337 | INFO    | metagpt.ext.spo.components.optimizer:_handle_first_round:80 - 
    ⚡ RUNNING Round 1 PROMPT ⚡
    , 2025-04-19 15:33:43.216 | INFO    | metagpt.utils.cost_manager:update_cost:57 - Total running cost: $0.000 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 226, completion_tokens: 2, 2025-04-19 15:33:43.370 | INFO    | metagpt.ext.spo.components.optimizer:_generate_optimized_prompt:97 - 
    🚀Round 2 OPTIMIZATION STARTING 🚀
    , 2025-04-19 15:33:43.370 | INFO    | metagpt.ext.spo.components.optimizer:_generate_optimized_prompt:98 - 
    Selecting prompt for round 1 and advancing to the iteration phase
    , 2025-04-19 15:33:49.760 | INFO    | metagpt.utils.cost_manager:update_cost:57 - Total running cost: $0.012 | Max budget: $10.000 | Current cost: $0.012, prompt_tokens: 587, completion_tokens: 321, 2025-04-19 15:33:49.761 | INFO    | metagpt.ext.spo.components.optimizer:_generate_optimized_prompt:116 - Modification of 2 round: Streamline the instructions and clarify the input format to reduce confusion and improve robustness against ambiguous job titles., 2025-04-19 15:33:49.761 | INFO    | metagpt.ext.spo.components.optimizer:_optimize_prompt:71 - 
    Round 2 Prompt: Your task is to classify the given job title into one of the following categories: 'RESEARCH', 'ENGINEERING', 'BUSINESS', 'FOUNDER'. If the job title does not fit any of these categories, classify it as 'OTHER'. You must strictly adhere to these categories. Provide your answer using one word only. Do not include any additional context or explanations.
    
        # INPUT declared title: the person's job title is
    , 2025-04-19 15:33:49.762 | INFO    | metagpt.ext.spo.components.optimizer:_evaluate_new_prompt:122 - 
    ⚡ RUNNING OPTIMIZED PROMPT ⚡
    , 2025-04-19 15:33:52.430 | INFO    | metagpt.utils.cost_manager:update_cost:57 - Total running cost: $0.001 | Max budget: $10.000 | Current cost: $0.000, prompt_tokens: 96, completion_tokens: 2, 2025-04-19 15:33:52.430 | INFO    | metagpt.ext.spo.components.optimizer:_evaluate_new_prompt:125 - 
    📊 EVALUATING OPTIMIZED PROMPT 📊
    , ..., 2025-04-19 15:51:11.884 | INFO    | metagpt.ext.spo.components.optimizer:show_final_result:56 - 
    ==================================================
    ]

## Asessing the results

| Original Prompt | Optimized Prompt |
|-----------------|------------------|
| Your task is to provide me with a direct classification of the person's job title into one of 4 categories. The categories you can decide are always: 'RESEARCH', 'ENGINEERING', 'BUSINESS', 'FOUNDER'. There is no possibility for mixed assignments. You always assign one and one only category to each subject. When in doubt, assign to 'OTHER'. You must strictly adhere to the categories I have mentioned, and nothing more. This means that you cannot use any other output apart from 'RESEARCH', 'ENGINEERING', 'BUSINESS', 'FOUNDER', 'OTHER'. Keep your answer very, very concise. Don't give context on your answer. As a matter of fact, only answer with one word based on the category you deem the most appropriate. Absolutely don't change this. You will be penalized if (1) you use a category outside of the ones I have mentioned and (2) you use more than 1 word in your output. # INPUT declared title: the person job title is {job_title} | Your task is to classify the given job title into one of the following categories: 'RESEARCH', 'ENGINEERING', 'BUSINESS', 'FOUNDER'. If the job title does not fit any of these categories, classify it as 'OTHER'. You must strictly adhere to these categories. If a job title is ambiguous or could fit into multiple categories, choose the most relevant category based on common industry standards. For example, 'Data Scientist' could fit into both 'RESEARCH' and 'ENGINEERING', but is typically classified as 'RESEARCH'. Similarly, 'Data Analyst' is typically classified as 'BUSINESS'. Provide your answer using one word only, in all uppercase letters without any additional context or explanations.<br><br># INPUT: The person's job title is: {job_title}<br><br># Example:<br># INPUT: The person's job title is: Software Developer<br># OUTPUT: ENGINEERING |

Results indicate the original prompt is modified according to typical best-practices, such as providing examples to guide the LLM (**few-shot prompting**), or by providing tag-like elements to direct the model's attention towards particular parts of the input prompt.

This revised prompt has been obtained using only 5 optimization "rounds", and can further be optimized (although finally satisfactory performance is of course a heuristic in the context of black-box optimization)