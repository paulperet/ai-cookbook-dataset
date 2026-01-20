# Multiclass Classification for Transactions

For this notebook we will be looking to classify a public dataset of transactions into a number of categories that we have predefined. These approaches should be replicable to any multiclass classification use case where we are trying to fit transactional data into predefined categories, and by the end of running through this you should have a few approaches for dealing with both labelled and unlabelled datasets.

The different approaches we'll be taking in this notebook are:
- **Zero-shot Classification:** First we'll do zero shot classification to put transactions in one of five named buckets using only a prompt for guidance
- **Classification with Embeddings:** Following this we'll create embeddings on a labelled dataset, and then use a traditional classification model to test their effectiveness at identifying our categories
- **Fine-tuned Classification:** Lastly we'll produce a fine-tuned model trained on our labelled dataset to see how this compares to the zero-shot and few-shot classification approaches

## Setup


```python
%load_ext autoreload
%autoreload
%pip install openai 'openai[datalib]' 'openai[embeddings]' transformers scikit-learn matplotlib plotly pandas scipy

```


```python
import openai
import pandas as pd
import numpy as np
import json
import os

COMPLETIONS_MODEL = "gpt-4"
os.environ["OPENAI_API_KEY"] = "<your-api-key>"
client = openai.OpenAI()
```

### Load dataset

We're using a public transaction dataset of transactions over £25k for the Library of Scotland. The dataset has three features that we'll be using:
- Supplier: The name of the supplier
- Description: A text description of the transaction
- Value: The value of the transaction in GBP

**Source**:

https://data.nls.uk/data/organisational-data/transactions-over-25k/


```python
transactions = pd.read_csv('./data/25000_spend_dataset_current.csv', encoding= 'unicode_escape')
print(f"Number of transactions: {len(transactions)}")
print(transactions.head())

```

    Number of transactions: 359
             Date                      Supplier                 Description  \
    0  21/04/2016          M & J Ballantyne Ltd       George IV Bridge Work   
    1  26/04/2016                  Private Sale   Literary & Archival Items   
    2  30/04/2016     City Of Edinburgh Council         Non Domestic Rates    
    3  09/05/2016              Computacenter Uk                 Kelvin Hall   
    4  09/05/2016  John Graham Construction Ltd  Causewayside Refurbishment   
    
       Transaction value (£)  
    0                35098.0  
    1                30000.0  
    2                40800.0  
    3                72835.0  
    4                64361.0  


## Zero-shot Classification

We'll first assess the performance of the base models at classifying these transactions using a simple prompt. We'll provide the model with 5 categories and a catch-all of "Could not classify" for ones that it cannot place.


```python
zero_shot_prompt = '''You are a data expert working for the National Library of Scotland.
You are analysing all transactions over £25,000 in value and classifying them into one of five categories.
The five categories are Building Improvement, Literature & Archive, Utility Bills, Professional Services and Software/IT.
If you can't tell what it is, say Could not classify

Transaction:

Supplier: {}
Description: {}
Value: {}

The classification is:'''

def format_prompt(transaction):
    return zero_shot_prompt.format(transaction['Supplier'], transaction['Description'], transaction['Transaction value (£)'])

def classify_transaction(transaction):

    
    prompt = format_prompt(transaction)
    messages = [
        {"role": "system", "content": prompt},
    ]
    completion_response = openai.chat.completions.create(
                            messages=messages,
                            temperature=0,
                            max_tokens=5,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0,
                            model=COMPLETIONS_MODEL)
    label = completion_response.choices[0].message.content.replace('\n','')
    return label

```


```python
# Get a test transaction
transaction = transactions.iloc[0]
# Use our completion function to return a prediction
print(f"Transaction: {transaction['Supplier']} {transaction['Description']} {transaction['Transaction value (£)']}")
print(f"Classification: {classify_transaction(transaction)}")

```

    Transaction: M & J Ballantyne Ltd George IV Bridge Work 35098.0
    Classification: Building Improvement


Our first attempt is correct, M & J Ballantyne Ltd are a house builder and the work they performed is indeed Building Improvement.

Lets expand the sample size to 25 and see how it performs, again with just a simple prompt to guide it


```python
test_transactions = transactions.iloc[:25]
test_transactions['Classification'] = test_transactions.apply(lambda x: classify_transaction(x),axis=1)

```

    /var/folders/3n/79rgh27s6l7_l91b9shw0_nr0000gp/T/ipykernel_81921/2775604370.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      test_transactions['Classification'] = test_transactions.apply(lambda x: classify_transaction(x),axis=1)



```python
test_transactions['Classification'].value_counts()

```




    Classification
    Building Improvement    17
    Literature & Archive     3
    Software/IT              2
    Could not classify       2
    Utility Bills            1
    Name: count, dtype: int64




```python
test_transactions.head(25)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Supplier</th>
      <th>Description</th>
      <th>Transaction value (£)</th>
      <th>Classification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21/04/2016</td>
      <td>M &amp; J Ballantyne Ltd</td>
      <td>George IV Bridge Work</td>
      <td>35098.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26/04/2016</td>
      <td>Private Sale</td>
      <td>Literary &amp; Archival Items</td>
      <td>30000.0</td>
      <td>Literature &amp; Archive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30/04/2016</td>
      <td>City Of Edinburgh Council</td>
      <td>Non Domestic Rates</td>
      <td>40800.0</td>
      <td>Utility Bills</td>
    </tr>
    <tr>
      <th>3</th>
      <td>09/05/2016</td>
      <td>Computacenter Uk</td>
      <td>Kelvin Hall</td>
      <td>72835.0</td>
      <td>Software/IT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>09/05/2016</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>64361.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>5</th>
      <td>09/05/2016</td>
      <td>A McGillivray</td>
      <td>Causewayside Refurbishment</td>
      <td>53690.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>6</th>
      <td>16/05/2016</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>365344.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>7</th>
      <td>23/05/2016</td>
      <td>Computacenter Uk</td>
      <td>Kelvin Hall</td>
      <td>26506.0</td>
      <td>Software/IT</td>
    </tr>
    <tr>
      <th>8</th>
      <td>23/05/2016</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>32777.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>9</th>
      <td>23/05/2016</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>32777.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>10</th>
      <td>30/05/2016</td>
      <td>ALDL</td>
      <td>ALDL Charges</td>
      <td>32317.0</td>
      <td>Could not classify</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10/06/2016</td>
      <td>Wavetek Ltd</td>
      <td>Kelvin Hall</td>
      <td>87589.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10/06/2016</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>381803.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>13</th>
      <td>28/06/2016</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>32832.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>14</th>
      <td>30/06/2016</td>
      <td>Glasgow City Council</td>
      <td>Kelvin Hall</td>
      <td>1700000.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>15</th>
      <td>11/07/2016</td>
      <td>Wavetek Ltd</td>
      <td>Kelvin Hall</td>
      <td>65692.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>16</th>
      <td>11/07/2016</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>139845.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>17</th>
      <td>15/07/2016</td>
      <td>Sotheby'S</td>
      <td>Literary &amp; Archival Items</td>
      <td>28500.0</td>
      <td>Literature &amp; Archive</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18/07/2016</td>
      <td>Christies</td>
      <td>Literary &amp; Archival Items</td>
      <td>33800.0</td>
      <td>Literature &amp; Archive</td>
    </tr>
    <tr>
      <th>19</th>
      <td>25/07/2016</td>
      <td>A McGillivray</td>
      <td>Causewayside Refurbishment</td>
      <td>30113.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>20</th>
      <td>31/07/2016</td>
      <td>ALDL</td>
      <td>ALDL Charges</td>
      <td>32317.0</td>
      <td>Could not classify</td>
    </tr>
    <tr>
      <th>21</th>
      <td>08/08/2016</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>32795.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>22</th>
      <td>15/08/2016</td>
      <td>Creative Video Productions Ltd</td>
      <td>Kelvin Hall</td>
      <td>26866.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>23</th>
      <td>15/08/2016</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>196807.0</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24/08/2016</td>
      <td>ECG Facilities Service</td>
      <td>Facilities Management Charge</td>
      <td>32795.0</td>
      <td>Building Improvement</td>
    </tr>
  </tbody>
</table>
</div>



Initial results are pretty good even with no labelled examples! The ones that it could not classify were tougher cases with few clues as to their topic, but maybe if we clean up the labelled dataset to give more examples we can get better performance.

## Classification with Embeddings

Lets create embeddings from the small set that we've classified so far - we've made a set of labelled examples by running the zero-shot classifier on 101 transactions from our dataset and manually correcting the 15 **Could not classify** results that we got

### Create embeddings

This initial section reuses the approach from the [Get_embeddings_from_dataset Notebook](Get_embeddings_from_dataset.ipynb) to create embeddings from a combined field concatenating all of our features


```python
df = pd.read_csv('./data/labelled_transactions.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Supplier</th>
      <th>Description</th>
      <th>Transaction value (£)</th>
      <th>Classification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15/08/2016</td>
      <td>Creative Video Productions Ltd</td>
      <td>Kelvin Hall</td>
      <td>26866</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29/05/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>74806</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29/05/2017</td>
      <td>Morris &amp; Spottiswood Ltd</td>
      <td>George IV Bridge Work</td>
      <td>56448</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31/05/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>164691</td>
      <td>Building Improvement</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24/07/2017</td>
      <td>John Graham Construction Ltd</td>
      <td>Causewayside Refurbishment</td>
      <td>27926</td>
      <td>Building Improvement</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['combined'] = "Supplier: " + df['Supplier'].str.strip() + "; Description: " + df['Description'].str.strip() + "; Value: " + str(df['Transaction value (£)']).strip()
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="data