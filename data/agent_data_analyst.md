# Build a Data Analyst Agent with LLMs

## Overview
This guide demonstrates how to create an autonomous data analyst agent using `smolagents`. The agent can load datasets, perform exploratory analysis, generate visualizations, and even train machine learning models—all through natural language instructions.

### Prerequisites
- A Hugging Face account and API token
- Basic understanding of Python and data analysis concepts

## Setup

First, install the required dependencies:

```bash
pip install seaborn smolagents transformers -q -U
```

## Step 1: Initialize the Agent

We'll use a `CodeAgent`, which can execute Python code directly. We authorize it to import common data science libraries.

```python
from smolagents import InferenceClientModel, CodeAgent
from huggingface_hub import login
import os

# Authenticate with Hugging Face
login(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# Initialize the model using Llama 3.1 70B via Inference API
model = InferenceClientModel("meta-llama/Llama-3.1-70B-Instruct")

# Create the CodeAgent with data science libraries
agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=["numpy", "pandas", "matplotlib.pyplot", "seaborn"],
    max_iterations=10,
)
```

**Key points:**
- `CodeAgent` executes Python code directly without needing separate tool definitions
- `additional_authorized_imports` specifies which libraries the agent can use
- `max_iterations` limits how many reasoning steps the agent can take

## Step 2: Prepare Analysis Context

Create a directory for saving figures and define additional notes about the dataset:

```python
import os

# Create directory for saving plots
os.mkdir("./figures")

# Define dataset notes for the agent
additional_notes = """
### Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower
age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)
parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.
"""
```

## Step 3: Run Exploratory Data Analysis

Now, instruct the agent to analyze the Titanic dataset:

```python
analysis = agent.run(
    """You are an expert data analyst.
Please load the source file and analyze its content.
According to the variables you have, begin by listing 3 interesting questions that could be asked on this data, for instance about specific correlations with survival rate.
Then answer these questions one by one, by finding the relevant numbers.
Meanwhile, plot some figures using matplotlib/seaborn and save them to the (already existing) folder './figures/': take care to clear each figure with plt.clf() before doing another plot.

In your final answer: summarize these correlations and trends
After each number derive real worlds insights, for instance: "Correlation between is_december and boredness is 1.3453, which suggest people are more bored in winter".
Your final answer should have at least 3 numbered and detailed parts.
""",
    additional_args=dict(
        additional_notes=additional_notes,
        source_file="titanic/train.csv"
    )
)

print(analysis)
```

**Expected output:**
```
The analysis of the Titanic data reveals that socio-economic status and sex are significant factors in determining survival rates. Passengers with lower socio-economic status and males are less likely to survive. The age of a passenger has a minimal impact on their survival rate.
```

The agent will:
1. Load the Titanic dataset
2. Formulate three analytical questions
3. Generate visualizations saved to `./figures/`
4. Provide insights with real-world interpretations

## Step 4: Extend to Machine Learning Tasks

Now, let's enhance the agent to perform predictive modeling by adding `sklearn` to its authorized imports:

```python
# Create a new agent with machine learning capabilities
agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=[
        "numpy",
        "pandas",
        "matplotlib.pyplot",
        "seaborn",
        "sklearn",
    ],
    max_iterations=12,
)

# Train a model and generate predictions
output = agent.run(
    """You are an expert machine learning engineer.
Please train a ML model on "titanic/train.csv" to predict the survival for rows of "titanic/test.csv".
Output the results under './output.csv'.
Take care to import functions and modules before using them!
""",
    additional_args=dict(additional_notes=additional_notes + "\n" + analysis)
)
```

**What happens:**
- The agent loads both training and test datasets
- It preprocesses the data, handles missing values, and encodes categorical variables
- It trains a machine learning model (typically a Random Forest or similar)
- It generates predictions and saves them to `./output.csv`

**Performance note:** When submitted to Kaggle, predictions from this agent achieved a score of **0.78229**, ranking #2824 out of 17,360 submissions—demonstrating competitive performance with minimal manual intervention.

## Key Takeaways

1. **Autonomous Analysis:** The agent can independently formulate questions, analyze data, and generate insights
2. **Visualization:** It creates and saves plots automatically
3. **End-to-End ML:** From data loading to model training and prediction generation
4. **Iterative Improvement:** The agent can learn from errors and adjust its approach

## Next Steps

To adapt this for your own use cases:
1. Replace the dataset path with your own data
2. Adjust the authorized imports based on your required libraries
3. Modify the prompt instructions to focus on your specific analytical questions
4. Consider adding custom tools for specialized operations

This framework provides a powerful starting point for automating data analysis workflows with LLMs.