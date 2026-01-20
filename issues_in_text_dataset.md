# Detecting Issues in a Text Dataset with Cleanlab


Authored by: [Aravind Putrevu](https://huggingface.co/aravindputrevu)


In this 5-minute quickstart tutorial, we use Cleanlab to detect various issues in an intent classification dataset composed of (text) customer service requests at an online bank. We consider a subset of the [Banking77-OOS Dataset](https://arxiv.org/abs/2106.04564) containing 1,000 customer service requests which are classified into 10 categories based on their intent (you can run this same code on any text classification dataset). [Cleanlab](https://github.com/cleanlab/cleanlab) automatically identifies bad examples in our dataset, including mislabeled data, out-of-scope examples (outliers), or otherwise ambiguous examples. Consider filtering or correcting such bad examples before you dive deep into modeling your data!

**Overview of what we'll do in this tutorial:**

- Use a pretrained transformer model to extract the text embeddings from the customer service requests

- Train a simple Logistic Regression model on the text embeddings to compute out-of-sample predicted probabilities

- Run Cleanlab's `Datalab` audit with these predictions and embeddings in order to identify problems like: label issues, outliers, and near duplicates in the dataset.


## Quickstart

    
Already have (out-of-sample) `pred_probs` from a model trained on an existing set of labels? Maybe you have some numeric `features` as well? Run the code below to find any potential label errors in your dataset.

**Note:** If running on Colab, may want to use GPU (select: Runtime > Change runtime type > Hardware accelerator > GPU)



```python
from cleanlab import Datalab

lab = Datalab(data=your_dataset, label_name="column_name_of_labels")
lab.find_issues(pred_probs=your_pred_probs, features=your_features)

lab.report()
lab.get_issues()

```

## Install required dependencies


You can use `pip` to install all packages required for this tutorial as follows:



```python
!pip install -U scikit-learn sentence-transformers datasets
!pip install -U "cleanlab[datalab]"
```


```python
import re
import string
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

from cleanlab import Datalab
```


```python
import random
import numpy as np

pd.set_option("display.max_colwidth", None)

SEED = 123456  # for reproducibility
np.random.seed(SEED)
random.seed(SEED)
```

## Load and format the text dataset



```python
from datasets import load_dataset

dataset = load_dataset("PolyAI/banking77", split="train")
data = pd.DataFrame(dataset[:1000])
data.head()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I am still waiting on my card?</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What can I do if my card still hasn't arrived after 2 weeks?</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I have been waiting over a week. Is the card still coming?</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Can I track my card while it is in the process of delivery?</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How do I know if I will get my card, or if it is lost?</td>
      <td>11</td>
    </tr>
  </tbody>
</table>




```python
raw_texts, labels = data["text"].values, data["label"].values
num_classes = len(set(labels))

print(f"This dataset has {num_classes} classes.")
print(f"Classes: {set(labels)}")
```

    This dataset has 7 classes.
    Classes: {32, 34, 36, 11, 13, 46, 17}


Let's view the i-th example in the dataset:


```python
i = 1  # change this to view other examples from the dataset
print(f"Example Label: {labels[i]}")
print(f"Example Text: {raw_texts[i]}")
```

    Example Label: 11
    Example Text: What can I do if my card still hasn't arrived after 2 weeks?


The data is stored as two numpy arrays:

1. `raw_texts` stores the customer service requests utterances in text format
2. `labels` stores the intent categories (labels) for each example

Bringing Your Own Data (BYOD)?

You can easily replace the above with your own text dataset, and continue with the rest of the tutorial.

Next we convert the text strings into vectors better suited as inputs for our ML models.

We will use numeric representations from a pretrained Transformer model as embeddings of our text. The [Sentence Transformers](https://huggingface.co/docs/hub/sentence-transformers) library offers simple methods to compute these embeddings for text data. Here, we load the pretrained `electra-small-discriminator` model, and then run our data through network to extract a vector embedding of each example.


```python
transformer = SentenceTransformer('google/electra-small-discriminator')
text_embeddings = transformer.encode(raw_texts)
```

Our subsequent ML model will directly operate on elements of `text_embeddings` in order to classify the customer service requests.

## Define a classification model and compute out-of-sample predicted probabilities

A typical way to leverage pretrained networks for a particular classification task is to add a linear output layer and fine-tune the network parameters on the new data. However this can be computationally intensive. Alternatively, we can freeze the pretrained weights of the network and only train the output layer without having to rely on GPU(s). Here we do this conveniently by fitting a scikit-learn linear model on top of the extracted embeddings.

To identify label issues, cleanlab requires a probabilistic prediction from your model for each datapoint. However these predictions will be _overfit_ (and thus unreliable) for datapoints the model was previously trained on. cleanlab is intended to only be used with **out-of-sample** predicted class probabilities, i.e. on datapoints held-out from the model during the training.

Here we obtain out-of-sample predicted class probabilities for every example in our dataset using a Logistic Regression model with cross-validation.
Make sure that the columns of your `pred_probs` are properly ordered with respect to the ordering of classes, which for Datalab is: lexicographically sorted by class name.


```python
model = LogisticRegression(max_iter=400)

pred_probs = cross_val_predict(model, text_embeddings, labels, method="predict_proba")
```

## Use Cleanlab to find issues in your dataset

Given feature embeddings and the (out-of-sample) predicted class probabilities obtained from any model you have, cleanlab can quickly help you identify low-quality examples in your dataset.

Here, we use Cleanlab's `Datalab` to find issues in our data. Datalab offers several ways of loading the data; we’ll simply wrap the training features and noisy labels in a dictionary.


```python
data_dict = {"texts": raw_texts, "labels": labels}
```

All that is need to audit your data is to call `find_issues()`. We pass in the predicted probabilities and the feature embeddings obtained above, but you do not necessarily need to provide all of this information depending on which types of issues you are interested in. The more inputs you provide, the more types of issues `Datalab` can detect in your data. Using a better model to produce these inputs will ensure cleanlab more accurately estimates issues.


```python
lab = Datalab(data_dict, label_name="labels")
lab.find_issues(pred_probs=pred_probs, features=text_embeddings)
```

The output would look like:

```bash
Finding null issues ...
Finding label issues ...
Finding outlier issues ...
Fitting OOD estimator based on provided features ...
Finding near_duplicate issues ...
Finding non_iid issues ...
Finding class_imbalance issues ...
Finding underperforming_group issues ...

Audit complete. 62 issues found in the dataset.
```

After the audit is complete, review the findings using the `report` method:


```python
lab.report()
```

    Here is a summary of the different kinds of issues found in the data:
    
        issue_type  num_issues
           outlier          37
    near_duplicate          14
             label          10
           non_iid           1
    
    Dataset Information: num_examples: 1000, num_classes: 7
    
    
    ---------------------- outlier issues ----------------------
    
    About this issue:
    	Examples that are very different from the rest of the dataset 
        (i.e. potentially out-of-distribution or rare/anomalous instances).
        
    
    Number of examples with this issue: 37
    Overall dataset quality in terms of this issue: 0.3671
    
    Examples representing most severe instances of this issue:
         is_outlier_issue  outlier_score
    791              True       0.024866
    601              True       0.031162
    863              True       0.060738
    355              True       0.064199
    157              True       0.065075
    
    
    ------------------ near_duplicate issues -------------------
    
    About this issue:
    	A (near) duplicate issue refers to two or more examples in
        a dataset that are extremely similar to each other, relative
        to the rest of the dataset.  The examples flagged with this issue
        may be exactly duplicated, or lie atypically close together when
        represented as vectors (i.e. feature embeddings).
        
    
    Number of examples with this issue: 14
    Overall dataset quality in terms of this issue: 0.5961
    
    Examples representing most severe instances of this issue:
         is_near_duplicate_issue  near_duplicate_score near_duplicate_sets  distance_to_nearest_neighbor
    459                     True              0.009544               [429]                      0.000566
    429                     True              0.009544               [459]                      0.000566
    501                     True              0.046044          [412, 517]                      0.002781
    412                     True              0.046044               [501]                      0.002781
    698                     True              0.054626               [607]                      0.003314
    
    
    ----------------------- label issues -----------------------
    
    About this issue:
    	Examples whose given label is estimated to be potentially incorrect
        (e.g. due to annotation error) are flagged as having label issues.
        
    
    Number of examples with this issue: 10
    Overall dataset quality in terms of this issue: 0.9930
    
    Examples representing most severe instances of this issue:
         is_label_issue  label_score  given_label  predicted_label
    379           False     0.025486           32               11
    100           False     0.032102           11               36
    300           False     0.037742           32               46
    485            True     0.057666           17               34
    159            True     0.059408           13               11
    
    
    ---------------------- non_iid issues ----------------------
    
    About this issue:
    	Whether the dataset exhibits statistically significant
        violations of the IID assumption like:
        changepoints or shift, drift, autocorrelation, etc.
        The specific violation considered is whether the
        examples are ordered such that almost adjacent examples
        tend to have more similar feature values.
        
    
    Number of examples with this issue: 1
    Overall dataset quality in terms of this issue: 0.0000
    
    Examples representing most severe instances of this issue:
         is_non_iid_issue  non_iid_score
    988              True       0.563774
    975             False       0.570179
    997             False       0.571891
    967             False       0.572357
    956             False       0.577413
    
    Additional Information: 
    p-value: 0.0


### Label issues

The report indicates that cleanlab identified many label issues in our dataset. We can see which examples are flagged as likely mislabeled and the label quality score for each example using the `get_issues` method, specifying `label` as an argument to focus on label issues in the data.


```python
label_issues = lab.get_issues("label")
label_issues.head()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>is_label_issue</th>
      <th>label_score</th>
      <th>given_label</th>
      <th>predicted_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>0.903926</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>0.860544</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>0.658309</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>0.697085</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>0.434934</td>
      <td>11</td>
      <td>11</td>
    </tr>
  </tbody>
</table>




| | is_label_issue | label_score | given_label | predicted_label |
|----------------|-------------|-------------|-----------------|-----------------|
| 0              | False       | 0.903926    | 11              | 11 |
| 1              | False       | 0.860544    | 11              | 11 |
| 2              | False       | 0.658309    | 11              | 11 |
| 3              | False       | 0.697085    | 11              | 11 |
| 4              | False       | 0.434934    | 11              | 11 |


This method returns a dataframe containing a label quality score for each example. These numeric scores lie between 0 and 1, where lower scores indicate examples more likely to be mislabeled. The dataframe also contains a boolean column specifying whether or not each example is identified to have a label issue (indicating it is likely mislabeled).

We can get the subset of examples flagged with label issues, and also sort by label quality score to find the indices of the 5 most likely mislabeled examples in our dataset.


```python
identified_label_issues = label_issues[label_issues["is_label_issue"] == True]
lowest_quality_labels = label_issues["label_score"].argsort()[:5].to_numpy()

print(
    f"cleanlab found {len(identified_label_issues)} potential label errors in the dataset.\n"
    f"Here are indices of the top 5 most likely errors: \n {lowest_quality_labels}"
)
```

    cleanlab found 10 potential label errors in the dataset.
    Here are indices of the top 5 most likely errors: 
     [379 100 300 485 159]


Let's review some of the most likely label errors.

Here we display the top 5 examples identified as the most likely label errors in the dataset, together with their given (original) label and a suggested alternative label from cleanlab.



```python
data_with_suggested_labels = pd.DataFrame(
    {"text": raw_texts, "given_label": labels, "suggested_label": label_issues["predicted_label"]}
)
data_with_suggested_labels.iloc[lowest_quality_labels]
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>given_label</th>
      <th>suggested_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>379</th>
      <td>Is there a specific source that the exchange rate for the transfer I'm planning on making is pulled from?</td>
      <td>32</td>
      <td>11</td>
    </tr>
    <tr>
      <th>100</th>
      <td>can you share card tracking number?</td>
      <td>11</td>
      <td>36</td>
    </tr>
    <tr>
      <th>300</th>
      <td>If I need to cash foreign transfers, how does that work?</td>
      <td>32</td>
      <td>46</td>
    </tr>
    <tr>
      <th>485</th>
      <td>Was I charged more than I should of been for a currency exchange?</td>
      <td>17</td>
      <td>34</td>
    </tr>
    <tr>
      <th>159</th>
      <td>Is there any way to see my card in the app?</td>
      <td>13</td>
      <td>11</td>
    </tr>
  </tbody>
</table>




  The output to the above command would like below:
  
|      | text                                                                                                      | given_label    | suggested_label |
|------|-----------------------------------------------------------------------------------------------------------|----------------|-----------------|
| 379  | Is there a specific source that the exchange rate for the transfer I'm planning on making is pulled from? | 32             | 11              |
| 100  | can you share card tracking number?                                                                       | 11             | 36              |
| 300  | If I