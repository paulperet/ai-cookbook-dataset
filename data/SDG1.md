# Synthetic Data Generation with LLMs: A Practical Guide (Part 1)

Synthetic data generation using large language models (LLMs) offers a powerful solution to a common problem: the availability of high-quality, diverse, and privacy-compliant data. This technique is useful for numerous scenarios, including training machine learning models, fine-tuning other LLMs, solving cold-start problems, building compelling demos, and performing scenario testing.

Key drivers for using synthetic data include:
1.  **Privacy Compliance:** Avoiding the use of real human data with privacy restrictions or identifiable information.
2.  **Structure & Control:** Creating data that is more structured and easier to manipulate than real-world data.
3.  **Data Augmentation:** Enhancing sparse datasets or datasets lacking in specific categories.
4.  **Improving Diversity:** Addressing imbalanced datasets by generating new, varied examples.

This guide is split into two parts. In this first part, we will cover:
1.  Generating a CSV with a structured prompt.
2.  Creating a Python program to generate CSV data.
3.  Building multiple related tables (DataFrames) with a Python program.
4.  Generating simple textual data.
5.  Addressing imbalanced or non-diverse textual data.

Parts 4 and 5 are particularly useful for creating synthetic data to fine-tune another LLM—for instance, using high-quality data from `gpt-4o` to fine-tune the more cost-effective `gpt-3.5-turbo`.

## Prerequisites and Setup

First, ensure you have the necessary libraries installed and your OpenAI API key configured.

```bash
pip install openai pandas scikit-learn matplotlib
```

```python
from openai import OpenAI
import os
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
```

We'll use `gpt-4o-mini` as our data generation model throughout this tutorial.

```python
datagen_model = "gpt-4o-mini"
```

## 1. Generate a CSV with a Structured Prompt

The simplest method is to generate data directly via a prompt. To get good results, clearly specify three things: the output format (CSV), the schema (column names), and the relationships between columns.

```python
question = """
Create a CSV file with 10 rows of housing data.
Each row should include the following fields:
 - id (incrementing integer starting at 1)
 - house size (m^2)
 - house price
 - location
 - number of bedrooms

Make sure that the numbers make sense (i.e. more rooms is usually bigger size, more expensive locations increase price. more size is usually higher price etc. make sure all the numbers make sense). Also only respond with the CSV.
"""

response = client.chat.completions.create(
  model=datagen_model,
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to generate synthetic data."},
    {"role": "user", "content": question}
  ]
)
res = response.choices[0].message.content
print(res)
```

The model will return a clean CSV string:

```csv
id,house_size_m2,house_price,location,number_of_bedrooms
1,50,150000,Suburban,2
2,75,250000,City Center,3
3,100,350000,Suburban,4
4,120,450000,Suburban,4
5,80,300000,City Center,3
6,90,400000,City Center,3
7,150,600000,Premium Area,5
8,200,750000,Premium Area,5
9,55,180000,Suburban,2
10,300,950000,Premium Area,6
```

## 2. Generate a CSV with a Python Program

Direct generation is limited by context window size. A more scalable approach is to ask the LLM to write a Python program that generates the data. This allows you to create much larger datasets and gives you an inspectable, modifiable script.

```python
question = """
Create a Python program to generate 100 rows of housing data.
I want you to at the end of it output a pandas dataframe with 100 rows of data.
Each row should include the following fields:
 - id (incrementing integer starting at 1)
 - house size (m^2)
 - house price
 - location
 - number of bedrooms

Make sure that the numbers make sense (i.e. more rooms is usually bigger size, more expensive locations increase price. more size is usually higher price etc. make sure all the numbers make sense).
"""

response = client.chat.completions.create(
  model=datagen_model,
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to generate synthetic data."},
    {"role": "user", "content": question}
  ]
)
res = response.choices[0].message.content
print(res)
```

The response will include a complete Python program, often with explanatory text. You can extract and run the code block. The program typically defines a function that creates a pandas DataFrame with sensible relationships between columns, providing a solid foundation you can edit for your specific needs.

## 3. Generate Multiple Related Tables with a Python Program

For complex data with relationships (e.g., housing, location, and house type tables), you need to provide more detailed specifications. Describe how the datasets relate, their relative sizes, and ensure foreign/primary keys are correctly implemented.

```python
question = """
Create a Python program to generate 3 different pandas dataframes.

1. Housing data
I want 100 rows. Each row should include the following fields:
 - id (incrementing integer starting at 1)
 - house size (m^2)
 - house price
 - location
 - number of bedrooms
 - house type
 + any relevant foreign keys

2. Location
Each row should include the following fields:
 - id (incrementing integer starting at 1)
 - country
 - city
 - population
 - area (m^2)
 + any relevant foreign keys

 3. House types
 - id (incrementing integer starting at 1)
 - house type
 - average house type price
 - number of houses
 + any relevant foreign keys

Make sure that the numbers make sense (i.e. more rooms is usually bigger size, more expensive locations increase price. more size is usually higher price etc. make sure all the numbers make sense).
Make sure that the dataframe generally follow common sense checks, e.g. the size of the dataframes make sense in comparison with one another.
Make sure the foreign keys match up and you can use previously generated dataframes when creating each consecutive dataframes.
You can use the previously generated dataframe to generate the next dataframe.
"""

response = client.chat.completions.create(
  model=datagen_model,
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to generate synthetic data."},
    {"role": "user", "content": question}
  ]
)
res = response.choices[0].message.content
print(res)
```

The model will return a program that generates three DataFrames (`location_df`, `house_type_df`, `housing_df`) with proper foreign key relationships (e.g., `location_id`, `house_type_id`). The code ensures logical consistency across tables, such as using location data to influence housing prices.

## 4. Generate Simple Textual Data

Textual data generation is useful for creating training pairs to fine-tune another model. In this example, we'll simulate a retailer needing product descriptions. The goal is to create input-output pairs where the input is a product name and category, and the output is a description.

To ensure easy parsing, we must explicitly specify the output format and instruct the model not to deviate from it.

```python
output_string = ""
for i in range(3):
  question = f"""
  I am creating input output training pairs to fine tune my gpt model. The usecase is a retailer generating a description for a product from a product catalogue. I want the input to be product name and category (to which the product belongs to) and output to be description.
  The format should be of the form:
  1.
  Input: product_name, category
  Output: description
  2.
  Input: product_name, category
  Output: description

  Do not add any extra characters around that formatting as it will make the output parsing break.
  Create as many training pairs as possible.
  """

  response = client.chat.completions.create(
    model=datagen_model,
    messages=[
      {"role": "system", "content": "You are a helpful assistant designed to generate synthetic data."},
      {"role": "user", "content": question}
    ]
  )
  res = response.choices[0].message.content
  output_string += res + "\n" + "\n"

print(output_string[:1000]) # Display a truncated sample
```

The output will be a block of text following the specified format:

```
1.
Input: Wireless Bluetooth Headphones, Electronics
Output: Immerse yourself in high-quality sound with these Wireless Bluetooth Headphones, featuring active noise cancellation and a comfortable over-ear design for extended listening sessions.

2.
Input: Organic Green Tea, Beverages
Output: Enjoy a refreshing cup of Organic Green Tea, sourced from the finest leaves, packed with antioxidants, and perfect for a healthy, invigorating boost anytime.
...
```

### Parse the Generated Textual Data

We can use a regular expression to cleanly extract the product names, categories, and descriptions into lists for further use.

```python
# Regex to parse the formatted data
pattern = re.compile(r'Input:\s*(.+?),\s*(.+?)\nOutput:\s*(.+?)(?=\n\n|\Z)', re.DOTALL)
matches = pattern.findall(output_string)

products = []
categories = []
descriptions = []

for match in matches:
    product, category, description = match
    products.append(product.strip())
    categories.append(category.strip())
    descriptions.append(description.strip())

print(products)
```

This will output a list of the generated product names:

```
['Wireless Bluetooth Headphones', 'Organic Green Tea', 'Stainless Steel Kitchen Knife', 'Hiking Backpack', 'Air Fryer']
```

You now have structured data ready for a fine-tuning dataset or other applications.

## Summary

In this first part, you've learned several foundational techniques for synthetic data generation:
1.  **Direct CSV Generation:** Quick for small, simple datasets.
2.  **Programmatic CSV Generation:** Scalable and modifiable for larger datasets.
3.  **Multi-Table Generation:** Essential for creating complex, relational data.
4.  **Textual Data Generation:** Useful for creating training pairs for fine-tuning, with careful attention to output formatting for easy parsing.

These methods provide a robust starting point for creating synthetic datasets to power a wide variety of AI and data science projects. In Part 2, we will explore advanced prompting strategies to generate even higher-quality textual data.