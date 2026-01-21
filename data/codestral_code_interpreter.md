# Building a Data Analysis Agent with Codestral and Code Interpreter

This guide walks you through creating an AI-powered data analyst using Mistral's Codestral model and the E2B Code Interpreter SDK. You'll build an agent that can load datasets, perform analysis, and generate visualizations—all through natural language prompts.

## Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- API keys for [Mistral AI](https://console.mistral.ai) and [E2B](https://e2b.dev/docs)
- A CSV dataset for analysis (we'll use `global_economy_indicators.csv` in this example)

## Step 1: Install Required Packages

Create a `requirements.txt` file with the following content:

```txt
mistralai
e2b-code-interpreter
pandas
matplotlib
scikit-learn
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## Step 2: Configure API Keys and System Prompt

Create a new Python file and set up your configuration:

```python
import re
from mistralai.client import MistralClient
from e2b_code_interpreter import CodeInterpreter

# Configure your API keys
MISTRAL_API_KEY = "your-mistral-api-key-here"  # Get from https://console.mistral.ai
E2B_API_KEY = "your-e2b-api-key-here"         # Get from https://e2b.dev/docs

MODEL_NAME = "codestral-latest"  # Available models: https://docs.mistral.ai/getting-started/models/

# System prompt that defines the agent's behavior
SYSTEM_PROMPT = """You're a Python data scientist. You are given tasks to complete and you run Python code to solve them.

Information about the csv dataset:
- It's in the `/home/user/global_economy_indicators.csv` file
- The CSV file is using , as the delimiter
- It has the following columns (examples included):
    - country: "Argentina", "Australia"
    - Region: "SouthAmerica", "Oceania"
    - Surface area (km2): for example, 2780400
    - Population in thousands (2017): for example, 44271
    - Population density (per km2, 2017): for example, 16.2
    - Sex ratio (m per 100 f, 2017): for example, 95.9
    - GDP: Gross domestic product (million current US$): for example, 632343
    - GDP growth rate (annual %, const. 2005 prices): for example, 2.4
    - GDP per capita (current US$): for example, 14564.5
    - Economy: Agriculture (% of GVA): for example, 10.0
    - Economy: Industry (% of GVA): for example, 28.1
    - Economy: Services and other activity (% of GVA): for example, 61.9
    - Employment: Agriculture (% of employed): for example, 4.8
    - Employment: Industry (% of employed): for example, 20.6
    - Employment: Services (% of employed): for example, 74.7
    - Unemployment (% of labour force): for example, 8.5
    - Employment: Female (% of employed): for example, 43.7
    - Employment: Male (% of employed): for example, 56.3
    - Labour force participation (female %): for example, 48.5
    - Labour force participation (male %): for example, 71.1
    - International trade: Imports (million US$): for example, 59253
    - International trade: Exports (million US$): for example, 57802
    - International trade: Balance (million US$): for example, -1451
    - Education: Government expenditure (% of GDP): for example, 5.3
    - Health: Total expenditure (% of GDP): for example, 8.1
    - Health: Government expenditure (% of total health expenditure): for example, 69.2
    - Health: Private expenditure (% of total health expenditure): for example, 30.8
    - Health: Out-of-pocket expenditure (% of total health expenditure): for example, 20.2
    - Health: External health expenditure (% of total health expenditure): for example, 0.2
    - Education: Primary gross enrollment ratio (f/m per 100 pop): for example, 111.5/107.6
    - Education: Secondary gross enrollment ratio (f/m per 100 pop): for example, 104.7/98.9
    - Education: Tertiary gross enrollment ratio (f/m per 100 pop): for example, 90.5/72.3
    - Education: Mean years of schooling (female): for example, 10.4
    - Education: Mean years of schooling (male): for example, 9.7
    - Urban population (% of total population): for example, 91.7
    - Population growth rate (annual %): for example, 0.9
    - Fertility rate (births per woman): for example, 2.3
    - Infant mortality rate (per 1,000 live births): for example, 8.9
    - Life expectancy at birth, female (years): for example, 79.7
    - Life expectancy at birth, male (years): for example, 72.9
    - Life expectancy at birth, total (years): for example, 76.4
    - Military expenditure (% of GDP): for example, 0.9
    - Population, female: for example, 22572521
    - Population, male: for example, 21472290
    - Tax revenue (% of GDP): for example, 11.0
    - Taxes on income, profits and capital gains (% of revenue): for example, 12.9
    - Urban population (% of total population): for example, 91.7

Generally, you follow these rules:
- ALWAYS FORMAT YOUR RESPONSE IN MARKDOWN
- ALWAYS RESPOND ONLY WITH CODE IN CODE BLOCK LIKE THIS:
```python
{code}
```
- the Python code runs in jupyter notebook.
- every time you generate Python, the code is executed in a separate cell. it's okay to make multiple calls to `execute_python`.
- display visualizations using matplotlib or any other visualization library directly in the notebook. don't worry about saving the visualizations to a file.
- you have access to the internet and can make api requests.
- you also have access to the filesystem and can read/write files.
- you can install any pip package (if it exists) if you need to be running `!pip install {package}`. The usual packages for data analysis are already preinstalled though.
- you can run any Python code you want, everything is running in a secure sandbox environment
"""
```

## Step 3: Create Helper Functions

### 3.1 Code Extraction Function

This function extracts Python code from the LLM's Markdown response:

```python
# Regular expression pattern to match Python code blocks
pattern = re.compile(r'```python\n(.*?)\n```', re.DOTALL)

def extract_code_block(llm_response):
    """Extract Python code from Markdown code blocks in the LLM response."""
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        print("Extracted code:")
        print(code)
        return code
    return ""
```

### 3.2 Code Execution Function

This function executes Python code in the E2B sandbox:

```python
def execute_code(e2b_code_interpreter, code):
    """Execute Python code in the E2B code interpreter sandbox."""
    print("Running code interpreter...")
    
    # Execute the code cell in the Jupyter notebook
    execution = e2b_code_interpreter.notebook.exec_cell(
        code,
        on_stderr=lambda stderr: print("[Code Interpreter STDERR]", stderr),
        on_stdout=lambda stdout: print("[Code Interpreter STDOUT]", stdout),
    )
    
    if execution.error:
        print("[Code Interpreter ERROR]", execution.error)
    else:
        return execution.results
```

### 3.3 Dataset Upload Function

This function uploads your dataset to the sandbox:

```python
def upload_dataset(code_interpreter, local_path="./global_economy_indicators.csv"):
    """Upload a dataset file to the code interpreter sandbox."""
    print("Uploading dataset to Code Interpreter sandbox...")
    with open(local_path, "rb") as f:
        remote_path = code_interpreter.upload_file(f)
    print(f"Uploaded to {remote_path}")
    return remote_path
```

## Step 4: Implement the Main Chat Function

This function orchestrates the conversation with Codestral and code execution:

```python
def chat_with_codestral(e2b_code_interpreter, user_message):
    """Send a message to Codestral, extract code, and execute it."""
    print(f"\n{'='*50}")
    print(f"User message: {user_message}")
    print(f"{'='*50}")
    
    # Initialize the Mistral client
    client = MistralClient(api_key=MISTRAL_API_KEY)
    
    # Prepare the conversation messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    
    # Call the Codestral model
    response = client.chat(
        model=MODEL_NAME,
        messages=messages,
    )
    
    # Extract the response content
    response_message = response.choices[0].message
    python_code = extract_code_block(response_message.content)
    
    if python_code:
        # Execute the extracted code
        execution_results = execute_code(e2b_code_interpreter, python_code)
        return execution_results
    else:
        print("Failed to extract Python code from the model's response")
        return []
```

## Step 5: Run the Complete Workflow

Now let's put everything together in a main execution block:

```python
def main():
    """Main execution function that orchestrates the entire workflow."""
    
    # Create a code interpreter session
    with CodeInterpreter(api_key=E2B_API_KEY) as code_interpreter:
        
        # Step 1: Upload the dataset
        upload_dataset(code_interpreter)
        
        # Step 2: Send a data analysis request
        user_query = """Make a chart showing linear regression of the relationship between 
        GDP per capita and life expectancy from the global_economy_indicators. 
        Filter out any missing values or values in wrong format."""
        
        results = chat_with_codestral(code_interpreter, user_query)
        
        # Step 3: Process the results
        if results:
            first_result = results[0]
            print("\nAnalysis complete!")
            
            # The result contains the visualization
            # You can access different formats:
            # - first_result.png  # PNG image data
            # - first_result.jpg  # JPEG image data
            # - first_result.pdf  # PDF data
            # - first_result.text # Any text output
            
            return first_result
        else:
            raise Exception("No results returned from code execution")

if __name__ == "__main__":
    result = main()
```

## Step 6: Understanding the Output

When you run the script, Codestral will generate and execute code similar to this:

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('/home/user/global_economy_indicators.csv')

# Filter out missing values
df = df.dropna(subset=['GDP per capita (current US$)', 'Life expectancy at birth, total (years)'])

# Convert columns to numeric, errors='coerce' will turn invalid parsing into NaN
df['GDP per capita (current US$)'] = pd.to_numeric(df['GDP per capita (current US$)'], errors='coerce')
df['Life expectancy at birth, total (years)'] = pd.to_numeric(df['Life expectancy at birth, total (years)'], errors='coerce')

# Drop NaN values after conversion
df = df.dropna(subset=['GDP per capita (current US$)', 'Life expectancy at birth, total (years)'])

# Prepare the data for linear regression
X = df[['GDP per capita (current US$)']]
y = df['Life expectancy at birth, total (years)']

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict life expectancy for all GDP per capita values
y_pred = model.predict(X)

# Plot the data and the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.title('Relationship between GDP per capita and life expectancy')
plt.xlabel('GDP per capita (current US$)')
plt.ylabel('Life expectancy at birth, total (years)')
plt.show()
```

## Key Features of This Implementation

1. **Secure Execution**: Code runs in an isolated Firecracker sandbox
2. **Natural Language Interface**: Ask for analyses in plain English
3. **Automatic Code Generation**: Codestral writes the Python code for you
4. **Visualization Support**: Charts and graphs are generated automatically
5. **Error Handling**: Invalid data is filtered out automatically

## Next Steps

You can extend this agent by:
- Adding support for more complex statistical analyses
- Implementing data preprocessing pipelines
- Creating interactive dashboards
- Adding database connectivity for live data sources
- Implementing multi-step analysis workflows

Remember to always validate the generated code before running it in production environments, and consider adding additional safety checks for more complex operations.