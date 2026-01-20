# How to evaluate LLMs


```python
! pip install mistralai evaluate
```


```python
from mistralai import Mistral
from getpass import getpass

api_key= getpass("Type your API Key")

client = Mistral(api_key = api_key)
```

    Type your API Key··········


## Example 1:  Information extraction benchmark with accuracy


### Evaluation data


```python
prompts = {
    "Johnson": {
        "medical_notes": "A 60-year-old male patient, Mr. Johnson, presented with symptoms of increased thirst, frequent urination, fatigue, and unexplained weight loss. Upon evaluation, he was diagnosed with diabetes, confirmed by elevated blood sugar levels. Mr. Johnson's weight is 210 lbs. He has been prescribed Metformin to be taken twice daily with meals. It was noted during the consultation that the patient is a current smoker. ",
        "golden_answer": {
            "age": 60,
            "gender": "male",
            "diagnosis": "diabetes",
            "weight": 210,
            "smoking": "yes",
        },
    },
    "Smith": {
        "medical_notes": "Mr. Smith, a 55-year-old male patient, presented with severe joint pain and stiffness in his knees and hands, along with swelling and limited range of motion. After a thorough examination and diagnostic tests, he was diagnosed with arthritis. It is important for Mr. Smith to maintain a healthy weight (currently at 150 lbs) and quit smoking, as these factors can exacerbate symptoms of arthritis and contribute to joint damage.",
        "golden_answer": {
            "age": 55,
            "gender": "male",
            "diagnosis": "arthritis",
            "weight": 150,
            "smoking": "yes",
        },
    },
}
```

### How to evaluate?

- Step 1: Define prompt template


```python
def run_mistral(user_message, model="mistral-large-latest"):
    messages = [{"role": "user", "content": user_message}]
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return chat_response.choices[0].message.content


# define prompt template
prompt_template = """
Extract information from the following medical notes:
{medical_notes}

Return json format with the following JSON schema:

{{
        "age": {{
            "type": "integer"
        }},
        "gender": {{
            "type": "string",
            "enum": ["male", "female", "other"]
        }},
        "diagnosis": {{
            "type": "string",
            "enum": ["migraine", "diabetes", "arthritis", "acne", "common cold"]
        }},
        "weight": {{
            "type": "integer"
        }},
        "smoking": {{
            "type": "string",
            "enum": ["yes", "no"]
        }},

}}
```
- Step 2: Define how we compare the model response with the golden answer


```python
import json


def compare_json_objects(obj1, obj2):
    total_fields = 0
    identical_fields = 0
    common_keys = set(obj1.keys()) & set(obj2.keys())
    for key in common_keys:
        identical_fields += obj1[key] == obj2[key]
    percentage_identical = (identical_fields / max(len(obj1.keys()), 1)) * 100
    return percentage_identical
```

- Step 3: Calculate accuracy rate across test cases


```python
accuracy_rates = []

# for each test case
for name in prompts:

    # define user message
    user_message = prompt_template.format(medical_notes=prompts[name]["medical_notes"])

    # run LLM
    response = json.loads(run_mistral(user_message))

    # calculate accuracy rate for this test case
    accuracy_rates.append(
        compare_json_objects(response, prompts[name]["golden_answer"])
    )

# calculate accuracy rate across test cases
sum(accuracy_rates) / len(accuracy_rates)
```




    100.0



## Example 2: evaluate code generation


```python
def run_mistral(user_message, model="mistral-large-latest"):
    client = Mistral(api_key=api_key)
    messages = [{"role":"user", "content": user_message}]
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return chat_response.choices[0].message.content

# define prompt template
python_prompts = {
    "sort_string": {
        "prompt": "Write a python function to sort the given string.",
        "test": 'assert sort_string("data") == "aadt"',
    },
    "is_odd": {
        "prompt": "Write a python function to check whether the given number is odd or not using bitwise operator.",
        "test": "assert is_odd(5) == True",
    },
}
```

- Step 1: Define prompt template


```python
prompt_template = """Write a Python function to execute the following task: {task}
Return only valid Python code. Do not give any explanation.
Never start with ```python.
Always start with def {name}(.
"""
```

- Step 2: Decide how we evaluate the code generation


```python
from evaluate import load
import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
code_eval = load("code_eval")
```

    [Downloading builder script: 9.18k, Downloading extra modules: 6.10k]



```python
code_eval.compute(
    references=["assert is_odd(5) == True"],
    predictions=[["def is_odd(n):\n    return n & 1 != 0"]],
    k=[1],
)
```

    /usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()





    ({'pass@1': 1.0},
     defaultdict(list,
                 {0: [(0,
                    {'task_id': 0,
                     'passed': True,
                     'result': 'passed',
                     'completion_id': 0})]}))



- Step 3: Calculate accuracy rate across test cases


```python
refs = []
preds = []

for name in python_prompts:

    # define user message
    user_message = prompt_template.format(
        task=python_prompts[name]["prompt"], name=name
    )

    # run LLM
    response = run_mistral(user_message)

    refs.append(python_prompts[name]["test"])
    preds.append([response])

# evaluate code generation
pass_at_1, results = code_eval.compute(references=refs, predictions=preds)

pass_at_1
```




    {'pass@1': 1.0}



# Example 3: evaluate summary generation with LLM


```python
news = (
    "BRUSSELS (Reuters) - Theresa May looked despondent , with deep rings under her eyes, EU chief executive Jean-Claude Juncker told aides after dining with the British prime minister last week, a German newspaper said on Sunday. The report by a Frankfurter Allgemeine Zeitung correspondent whose leaked account of a Juncker-May dinner in April caused upset in London, said Juncker thought her marked by battles over Brexit with her own Conservative ministers as she asked for EU help to create more room for maneuver at home. No immediate comment was available from Juncker s office, which has a policy of not commenting on reports of meetings. The FAZ said May, who flew in for a hastily announced dinner in Brussels with the European Commission president last Monday ahead of an EU summit, seemed to Juncker anxious, despondent and disheartened , a woman who trusts hardly anyone but is also not ready for a clear-out to free herself . As she later did over dinner on Thursday with fellow EU leaders, May asked for help to overcome British divisions. She indicated that back home friend and foe are at her back plotting to bring her down, the paper said. May said she had no room left to maneuver. The Europeans have to create it for her. May s face and appearance spoke volumes, Juncker later told his colleagues, the FAZ added. She has deep rings under her eyes. She looks like someone who can t sleep a wink. She smiles for the cameras, it went on, but it looks forced , unlike in the past, when she could shake with laughter. Now she needs all her strength not to lose her poise. As with the April dinner at 10 Downing Street, when the FAZ reported that Juncker thought May in another galaxy in terms of Brexit expectations, both sides issued statements after last week s meeting saying talks were constructive and friendly . They said they agreed negotiations should be accelerated . May dismissed the dinner leak six months ago as Brussels gossip , though officials on both sides said the report in the FAZ did little to foster an atmosphere of trust which they agree will be important to reach a deal. German Chancellor Angela Merkel was also reported to have been irritated by that leak. Although the summit on Thursday and Friday rejected May s call for an immediate start to talks on the future relationship, leaders made a gesture to speed up the process and voiced hopes of opening a new phase in December. Some said they understood May s difficulties in forging consensus in London.",
)
```

- Step 1: Generate summary for the given news


```python
def run_mistral(user_message, model="open-mistral-7b", is_json=False):
    client = Mistral(api_key=api_key)
    messages = [{"role":"user", "content":user_message}]

    if is_json:
        chat_response = client.chat.complete(
            model=model, messages=messages, response_format={"type": "json_object"}
        )
    else:
        chat_response = client.chat.complete(model=model, messages=messages)

    return chat_response.choices[0].message.content
```


```python
summary_prompt = f"""
Summarize the following news. Write the summary based on the following criteria: relevancy and readability. Consider the sources cited, the quality of evidence provided, and any potential biases or misinformation.

## News:
{news}
```
```python
summary = run_mistral(summary_prompt)
```

- Step 2: Define evaluation metrics and rubrics


```python
eval_rubrics = [
    {
        "metric": "relevancy",
        "rubrics": """
        Score 1: The summary is not relevant to the original text.
        Score 2: The summary is somewhat relevant to the original text, but has significant flaws.
        Score 3: The summary is mostly relevant to the original text, and effectively conveys its main ideas and arguments.
        Score 4: The summary is highly relevant to the original text, and provides additional value or insight.
        """,
    },
    {
        "metric": "readability",
        "rubrics": """
        Score 1: The summary is difficult to read and understand.
        Score 2: The summary is somewhat readable, but has significant flaws.
        Score 3: The summary is mostly readable and easy to understand.
        Score 4: The summary is highly readable and engaging.
        """,
    },
]
```

- Step 3: Employ a more powerful LLM (e.g., Mistral Large) as a judge


```python
scoring_prompt = """
Please read the provided news article and its corresponding summary.
Based on the specified evaluation metric and rubrics, assign an integer score between 1 and 4 to the summary.
Then, return a JSON object with the metric as the key and the evaluation score as the value.

# Evaluation metric:
{metric}

# Evaluation rubrics:
{rubrics}

# News article
{news}

# Summary
{summary}

```
```python
for i in eval_rubrics:
    eval_output = run_mistral(
        scoring_prompt.format(
            news=news, summary=summary, metric=i["metric"], rubrics=i["rubrics"]
        ),
        model="mistral-large-latest",
        is_json=True,
    )
    print(eval_output)
```

    [{"relevancy": 4}, {"readability": 3}]



```python

```