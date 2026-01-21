# Building and Running Evaluations with OpenAI Evals

## Introduction

Evaluation is the process of validating and testing the outputs that your LLM applications produce. Strong evaluations ("evals") lead to more stable, reliable applications that are resilient to code and model changes. An eval is a task used to measure the quality of an LLM or LLM system's output. Given an input prompt, we generate an output and evaluate it against a set of ideal answers to determine quality.

### Why Evaluations Matter

If you're building with foundational models like GPT-4, creating high-quality evals is one of the most impactful things you can do. AI development involves iterative design, and without evals, it becomes difficult and time-intensive to understand how different model versions and prompts affect your use case.

With OpenAI's continuous model upgrades, evals allow you to efficiently test model performance for your use cases in a standardized way. Developing a suite of evals customized to your objectives helps you quickly understand how new models may perform. You can also integrate evals into your CI/CD pipeline to ensure desired accuracy before deployment.

### Evaluation Approaches

There are two main ways to evaluate completions:

1. **Logic-based checking**: Write validation logic in code to compare outputs to expected answers
2. **Model grading**: Use the model itself to inspect and judge the quality of responses

The OpenAI Evals framework provides templates for both approaches, making it easier to create standardized evaluations.

## Prerequisites

Before starting, ensure you have:

1. Cloned the OpenAI Evals repository: `git clone git@github.com:openai/evals.git`
2. Followed the [setup instructions](https://github.com/openai/evals)
3. Set your OpenAI API key as an environment variable: `OPENAI_API_KEY`

```python
from openai import OpenAI
import pandas as pd

client = OpenAI()
```

## Step 1: Understanding Eval Components

At its core, an eval consists of:
1. A test dataset in JSONL format
2. An eval class defined in a YAML file

For this tutorial, we'll create an eval that tests a model's ability to generate syntactically correct SQL queries.

## Step 2: Creating the System Prompt

First, define the system prompt that provides context and instructions to the model:

```
TASK: Answer the following question with syntactically correct SQLite SQL. The SQL should be correct and be in context of the previous question-answer pairs.
Table car_makers, columns = [*,Id,Maker,FullName,Country]
Table car_names, columns = [*,MakeId,Model,Make]
Table cars_data, columns = [*,Id,MPG,Cylinders,Edispl,Horsepower,Weight,Accelerate,Year]
Table continents, columns = [*,ContId,Continent]
Table countries, columns = [*,CountryId,CountryName,Continent]
Table model_list, columns = [*,ModelId,Maker,Model]
Foreign_keys = [countries.Continent = continents.ContId,car_makers.Country = countries.CountryId,model_list.Maker = car_makers.Id,car_names.Model = model_list.Model,cars_data.Id = car_names.MakeId]
```

## Step 3: Generating Synthetic Test Data

Instead of manually creating test cases, we can use GPT-4 to generate synthetic question-answer pairs. This accelerates dataset creation while maintaining quality.

```python
# Define the system prompt for data generation
system_prompt = """You are a helpful assistant that can ask questions about a database table and write SQL queries to answer the question.
A user will pass in a table schema and your job is to return a question answer pairing. The question should relevant to the schema of the table,
and you can speculate on its contents. You will then have to generate a SQL query to answer the question. Below are some examples of what this should look like.

Example 1
```````````
User input: Table museum, columns = [*,Museum_ID,Name,Num_of_Staff,Open_Year]\nTable visit, columns = [*,Museum_ID,visitor_ID,Num_of_Ticket,Total_spent]\nTable visitor, columns = [*,ID,Name,Level_of_membership,Age]\nForeign_keys = [visit.visitor_ID = visitor.ID,visit.Museum_ID = museum.Museum_ID]\n
Assistant Response:
Q: How many visitors have visited the museum with the most staff?
A: SELECT count ( * )  FROM VISIT AS T1 JOIN MUSEUM AS T2 ON T1.Museum_ID   =   T2.Museum_ID WHERE T2.Num_of_Staff   =   ( SELECT max ( Num_of_Staff )  FROM MUSEUM ) 
```````````

Example 2
```````````
User input: Table museum, columns = [*,Museum_ID,Name,Num_of_Staff,Open_Year]\nTable visit, columns = [*,Museum_ID,visitor_ID,Num_of_Ticket,Total_spent]\nTable visitor, columns = [*,ID,Name,Level_of_membership,Age]\nForeign_keys = [visit.visitor_ID = visitor.ID,visit.Museum_ID = museum.Museum_ID]\n
Assistant Response:
Q: What are the names who have a membership level higher than 4?
A: SELECT Name   FROM VISITOR AS T1 WHERE T1.Level_of_membership   >   4 
```````````

Example 3
```````````
User input: Table museum, columns = [*,Museum_ID,Name,Num_of_Staff,Open_Year]\nTable visit, columns = [*,Museum_ID,visitor_ID,Num_of_Ticket,Total_spent]\nTable visitor, columns = [*,ID,Name,Level_of_membership,Age]\nForeign_keys = [visit.visitor_ID = visitor.ID,visit.Museum_ID = museum.Museum_ID]\n
Assistant Response:
Q: How many tickets of customer id 5?
A: SELECT count ( * )  FROM VISIT AS T1 JOIN VISITOR AS T2 ON T1.visitor_ID   =   T2.ID WHERE T2.ID   =   5 
```````````
"""

# Define the database schema for our car manufacturing dataset
user_input = "Table car_makers, columns = [*,Id,Maker,FullName,Country]\nTable car_names, columns = [*,MakeId,Model,Make]\nTable cars_data, columns = [*,Id,MPG,Cylinders,Edispl,Horsepower,Weight,Accelerate,Year]\nTable continents, columns = [*,ContId,Continent]\nTable countries, columns = [*,CountryId,CountryName,Continent]\nTable model_list, columns = [*,ModelId,Maker,Model]\nForeign_keys = [countries.Continent = continents.ContId,car_makers.Country = countries.CountryId,model_list.Maker = car_makers.Id,car_names.Model = model_list.Model,cars_data.Id = car_names.MakeId]"

# Create the message structure for GPT-4
messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": user_input
    }
]

# Generate synthetic data
completion = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=messages,
    temperature=0.7,
    n=5
)

# Display the generated questions
for choice in completion.choices:
    print(choice.message.content + "\n")
```

## Step 4: Formatting the Eval Dataset

The evals framework requires data in a specific JSONL format. Each entry should have:
- An `input` field containing the conversation history (system prompt + user question)
- An `ideal` field containing the expected correct answer

```python
eval_data = []
input_prompt = "TASK: Answer the following question with syntactically correct SQLite SQL. The SQL should be correct and be in context of the previous question-answer pairs.\nTable car_makers, columns = [*,Id,Maker,FullName,Country]\nTable car_names, columns = [*,MakeId,Model,Make]\nTable cars_data, columns = [*,Id,MPG,Cylinders,Edispl,Horsepower,Weight,Accelerate,Year]\nTable continents, columns = [*,ContId,Continent]\nTable countries, columns = [*,CountryId,CountryName,Continent]\nTable model_list, columns = [*,ModelId,Maker,Model]\nForeign_keys = [countries.Continent = continents.ContId,car_makers.Country = countries.CountryId,model_list.Maker = car_makers.Id,car_names.Model = model_list.Model,cars_data.Id = car_names.MakeId]"

# Process each generated question-answer pair
for choice in completion.choices:
    question = choice.message.content.split("Q: ")[1].split("\n")[0]  # Extracting the question
    answer = choice.message.content.split("\nA: ")[1].split("\n")[0]  # Extracting the answer
    
    eval_data.append({
        "input": [
            {"role": "system", "content": input_prompt},
            {"role": "user", "content": question},
        ],
        "ideal": answer
    })

# Display the formatted eval data
for item in eval_data:
    print(item)
```

## Step 5: Creating the Eval Registry File

The evals framework uses YAML files to define eval configurations. Create a file called `spider-sql.yaml` with the following structure:

```yaml
spider-sql:
  id: spider-sql.dev.v0
  metrics: [accuracy]
  description: Eval that scores SQL generation accuracy on car manufacturing database
  disclaimer: This eval uses synthetic data generated by GPT-4
```

The key components are:
- `id`: A unique identifier for your eval
- `metrics`: The evaluation metrics to use (accuracy, match, includes, or fuzzyMatch)
- `description`: A brief description of what the eval tests
- `disclaimer`: Any additional notes about the eval

## Step 6: Running the Eval

Once you have your dataset and registry file, you can run the eval using the evals CLI:

```bash
# Save your eval data to a JSONL file
python -c "import json; data = [...]" > eval_data.jsonl

# Run the eval
oaieval gpt-4 spider-sql --data eval_data.jsonl
```

The framework will:
1. Load your eval configuration
2. Run each test case through the specified model
3. Compare outputs to ideal answers using your chosen metrics
4. Generate a report with accuracy scores and detailed results

## Best Practices

1. **Start small**: Begin with a few high-quality test cases before scaling up
2. **Validate manually**: Always manually check a sample of generated synthetic data
3. **Use appropriate metrics**: Choose metrics that match your evaluation needs
4. **Monitor costs**: Be aware of API costs when running evals at scale
5. **Iterate**: Use eval results to improve your prompts and system design

## Conclusion

You've now learned how to:
- Understand the importance of evaluations in LLM development
- Generate synthetic test data using GPT-4
- Format data for the OpenAI Evals framework
- Create eval registry files
- Run evaluations to measure model performance

By integrating evals into your development workflow, you can build more reliable, robust LLM applications that maintain quality across model upgrades and code changes.