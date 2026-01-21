# Data Validation with Reasoning: A Medical Dataset Tutorial

In this guide, you'll learn how to use OpenAI's reasoning model (`o1-preview`) to perform intelligent data validation. We'll create a synthetic medical dataset, validate it for inconsistencies, and measure the model's accuracy in identifying and explaining issues.

## Prerequisites

First, ensure you have the necessary libraries installed and configured.

```bash
pip install openai scikit-learn pandas
```

```python
from openai import OpenAI
import json
import csv
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize the OpenAI client
client = OpenAI()
MODEL = 'o1-preview'
```

## Step 1: Generate a Synthetic Medical Dataset

We'll create a dataset containing realistic medical records, some of which have intentional errors. This dataset will serve as our testbed for validation.

Define a function to prompt the model to generate CSV-formatted data.

```python
def generate_data():
    messages = [
        {
            "role": "user",
            "content": """
You are a helpful assistant designed to generate data. You will be given a format for the data to generate and some examples of the data.

When generating Patient IDs, use the format 'P' followed by a three-digit number (e.g., P006, P941, P319).

Intentionally make some mistakes in the data generation and document them in the appropriate columns ('Is Valid' and 'Issue') if the row of data is invalid.

The types of mistakes to include are:

- **Allergy Contradictions**: Prescribing a medication that the patient is allergic to (e.g., prescribing Penicillin to a patient allergic to Penicillin).
- **Medical History and Medication Mismatch**: A patient with a medical condition not receiving appropriate medication (e.g., a diabetic patient not prescribed any diabetes medication).
- **Lab Results and Diagnosis Mismatch**: Lab results that do not support the diagnosis (e.g., normal glucose levels but diagnosed with Diabetes Type 2).
- **Other Plausible Mistakes**: Any other realistic errors that could occur in medical records, such as incorrect gender entries, impossible dates of birth, or inconsistent treatment plans.

Ensure that when 'Is Valid' is 'False', the 'Issue' column clearly explains the problem.

Return 100 rows of data for the user. Your response should strictly be in the format of a valid CSV.

Generate Synthetic Medical Records Dataset with the following columns:
    - Patient ID: A randomly generated patient id
    - Date of Birth: Date of birth of the patient
    - Gender: M/F
    - Medical History: Past diagnoses
    - Current Medications: Medication the patient is taking
    - Allergies: Identified allergies
    - Lab Results (Glucose mg/dL)
    - Diagnoses: Current diagnosis
    - Treatment Plan: Current treatment plan
    - Is Valid: Whether or not the current row of data is valid (True/False)
    - Issue: If the row of data is not valid, what the issue is

Patient ID,Date of Birth,Gender,Medical History,Current Medications,Allergies,Lab Results (Glucose mg/dL),Diagnoses,Treatment Plan,Is Valid,Issue
P001,1980-05-14,M,Hypertension,Lisinopril,None,110,Hypertension,Continue Lisinopril,True,
P002,1975-11-30,F,Diabetes Type 2,Metformin,Penicillin,90,Diabetes Type 2,Continue Metformin,True,
P003,1990-07-22,F,Asthma,Albuterol,Aspirin,85,Asthma,Prescribe Albuterol,True,
P004,2000-03-10,M,None,Amoxicillin,Penicillin,95,Infection,Prescribe Amoxicillin,False,Prescribed Amoxicillin despite Penicillin allergy
P005,1985-09-18,F,Hyperlipidemia,Atorvastatin,None,200,Hyperlipidemia,Continue Atorvastatin,True,
P006,1978-12-05,M,Hypertension; Diabetes Type 2,Lisinopril; Insulin,None,55,Diabetes Type 2,Adjust insulin dosage,False,Low glucose level not properly addressed
            """
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    # Clean the response to extract pure CSV
    return response.choices[0].message.content.replace('```csv', '').replace('```', '')
```

Now, generate the data and save it to a CSV file. We'll generate multiple batches to build a substantial dataset.

```python
# Generate data and append to a file
generated_data = []
data = generate_data()
generated_data.extend(data.strip().split('\n'))

# Save the generated data to a CSV file
with open('medicalData.csv', 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for row in generated_data:
        csvwriter.writerow(row.split(','))

print("Synthetic data generation and appending completed.")
```

## Step 2: Validate the Dataset with the Reasoning Model

Next, we'll create a validation function that uses the `o1-preview` model to analyze each row of data. The model will determine if the row is valid and, if not, explain the issue.

```python
def validate_data(input_data):
    messages = [
        {
            "role": "user",
            "content": f"""
You are a helpful assistant designed to validate the quality of medical datasets. You will be given a single row of medical data, and your task is to determine whether the data is valid.

- Carefully analyze the data for any inconsistencies, contradictions, missing values, or implausible information.
- Consider the logical relationships between different fields (e.g., treatments should be appropriate for the diagnoses, medications should not conflict with allergies, lab results should be consistent with diagnoses, etc.).
- Use your general medical knowledge to assess the validity of the data.
- Focus solely on the information provided without making assumptions beyond the given data.

**Return only a JSON object** with the following two properties:

- `"is_valid"`: a boolean (`true` or `false`) indicating whether the data is valid.
- `"issue"`: if `"is_valid"` is `false`, provide a brief explanation of the issue; if `"is_valid"` is `true`, set `"issue"` to `null`.

Both JSON properties must always be present.

Do not include any additional text or explanations outside the JSON object.

MEDICAL DATA:
{input_data}
            """
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )

    response_content = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
    
    try:
        # Parse the JSON response
        response_dict = json.loads(response_content)
        return response_dict
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {response_content}")
        raise e
```

Load the dataset, excluding the ground-truth validation columns, and run the validation function on each row in parallel for efficiency.

```python
# Read the CSV file and exclude the last two columns (ground truth)
input_data = []
with open('medicalData.csv', 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)
    for row in reader:
        input_data.append(row[:-2])  # Exclude "Is Valid" and "Issue" columns

# Extract the true labels from the CSV for later evaluation
true_is_valid = []
true_issues = []

with open('medicalData.csv', 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)
    for row in reader:
        true_is_valid.append(row[-2] == 'True')
        true_issues.append(row[-1])

# Function to validate a single row
def validate_row(row):
    input_str = ','.join(row)
    result_json = validate_data(input_str)
    return result_json

# Validate all rows using parallel processing
pred_is_valid = [False] * len(input_data)
pred_issues = [''] * len(input_data)

with ThreadPoolExecutor() as executor:
    futures = {executor.submit(validate_row, row): i for i, row in enumerate(input_data)}
    
    for future in as_completed(futures):
        i = futures[future]  # Get the index of the current row
        result_json = future.result()
        pred_is_valid[i] = result_json['is_valid']
        pred_issues[i] = result_json['issue']
```

## Step 3: Calculate Validation Accuracy

Now, compare the model's predictions against the ground truth to compute precision, recall, and F1-score for issue detection.

```python
# Convert predicted and true labels to boolean values
pred_is_valid_bool = [bool(val) if isinstance(val, bool) else val == 'True' for val in pred_is_valid]
true_is_valid_bool = [bool(val) if isinstance(val, bool) else val == 'True' for val in true_is_valid]

# Calculate precision, recall, and F1-score
precision = precision_score(true_is_valid_bool, pred_is_valid_bool, pos_label=True)
recall = recall_score(true_is_valid_bool, pred_is_valid_bool, pos_label=True)
f1 = f1_score(true_is_valid_bool, pred_is_valid_bool, pos_label=True)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
```

## Step 4: Assess Issue Explanation Accuracy

For rows where both the model and the ground truth indicate an issue, we need to check if the model's explanation matches the true issue. We'll use a secondary validation function with a simpler model (`o1-preview`) to compare explanations.

```python
def validate_issue(model_generated_answer, correct_answer):
    messages = [
        {
            "role": "user",
            "content": f"""
You are a medical expert assistant designed to validate the quality of an LLM-generated answer.

The model was asked to review a medical dataset row to determine if the data is valid. If the data is not valid, it should provide a justification explaining why.

Your task:

    •	Compare the model-generated justification with the correct reason provided.
    •	Determine if they address the same underlying medical issue or concern, even if phrased differently.
    •	Focus on the intent, medical concepts, and implications rather than exact wording.

Instructions:

    •	If the justifications have the same intent or address the same medical issue, return True.
    •	If they address different issues or concerns, return False.
    •	Only respond with a single word: True or False.

Examples:

    1.	Example 1:
    •	Model Generated Response: “The patient is allergic to penicillin”
    •	Correct Response: “The patient was prescribed penicillin despite being allergic”
    •	Answer: True
    2.	Example 2:
    •	Model Generated Response: “The date of birth of the patient is incorrect”
    •	Correct Response: “The patient was prescribed penicillin despite being allergic”
    •	Answer: False


Model Generated Response: {model_generated_answer}
Correct Response:  {correct_answer}
            """
        }
    ]

    response = client.chat.completions.create(
        model="o1-preview",
        messages=messages
    )

    result = response.choices[0].message.content
    return result
```

Run the issue validation on the subset of rows where an issue was correctly detected.

```python
# Validate issues for rows where both true and predicted 'is_valid' are False
validation_results = []
issue_matches_full = [False] * len(true_is_valid)

with ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(validate_issue, pred_issues[i], true_issues[i]): i
        for i in range(len(pred_is_valid_bool))
        if not pred_is_valid_bool[i] and not true_is_valid_bool[i]
    }
    
    for future in as_completed(futures):
        i = futures[future]  # Get the original index
        issue_match = future.result()
        issue_matches_full[i] = (issue_match == 'True')
        validation_results.append({
            "index": i,
            "predicted_issue": pred_issues[i],
            "true_issue": true_issues[i],
            "issue_match": issue_matches_full[i]
        })
    
    # Calculate issue accuracy
    issue_accuracy = sum([i['issue_match'] for i in validation_results]) / len(validation_results)
    
    # Compile final results
    model_results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "issue_accuracy": issue_accuracy
    }

# Create DataFrames for results
df_results = pd.DataFrame([model_results])
df_validation_results = pd.DataFrame(validation_results)

print("Overall Model Performance:")
print(df_results)
print("\nDetailed Issue Validation Results:")
print(df_validation_results.head())  # Display first few rows
```

## Conclusion

You have successfully built a system that uses a reasoning model to validate medical data. The model demonstrates high precision and recall in detecting invalid rows and reasonable accuracy in explaining the specific issues. This approach can be adapted to streamline data validation across various domains, leveraging AI to catch complex inconsistencies that rule-based systems might miss.