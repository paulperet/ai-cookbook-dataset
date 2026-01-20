```python
# !pip install scikit-learn datasets
```

```python
from datasets import load_dataset
from sklearn.model_selection import train_test_split
```

```python
fib_ds = load_dataset('r-three/fib', split='test')
fib_df = fib_ds.to_pandas()
```

    Repo card metadata block was not found. Setting CardData to empty.

```python
# Visualize the CNN/DM data
fib_df.loc[fib_df['dataset'] == 'cnn_dm', ['input', 'list_choices', 'correct_choice']].head(5)
```

| | input | list_choices | correct_choice |
| --- | --- | --- | --- |
| 3122 | ( cnn ) the american pharmacists association i... | [&lt;t&gt; the american pharmacists association pass... | &lt;t&gt; the american pharmacists association passe... |
| 3123 | ( cnn ) oprah 's in there . so 's bill murray ... | [&lt;t&gt; `` the late show with david letterman '' ... | &lt;t&gt; `` the late show with david letterman '' c... |
| 3124 | ( cnn ) feeling so happy you just ca n't stand... | [&lt;t&gt; a new study has found that acetaminophen ... | &lt;t&gt; subjects taking acetaminophen reacted less... |
| 3125 | ( cnn ) love it or hate it , jared leto 's int... | [&lt;t&gt; the oscar winner put on white makeup -lrb... | &lt;t&gt; leto will play the clown prince of crime i... |
| 3126 | ( the hollywood reporter ) the original cast o... | [&lt;t&gt; -lrb- the hollywood reporter -rrb- the or... | &lt;t&gt; `` twin peaks '' creator david lynch annou... |

```python
# Only keep xsum data
fib_df = fib_df[fib_df['dataset'] == 'xsum']
```

```python
fib_df[['input', 'list_choices', 'correct_choice']].head(5)
```

| | input | list_choices | correct_choice |
| --- | --- | --- | --- |
| 0 | Vehicles and pedestrians will now embark and d... | [ A new service on the Isle of Wight's chain f... | Passengers using a chain ferry have been warne... |
| 1 | If you leave your mobile phone somewhere do yo... | [ You may be worried about your health, but wh... | Do you ever feel lonely, stressed or jealous w... |
| 2 | Speaking on TV, Maria Zakharova said Jews had ... | [ The Russian foreign minister has said she ha... | A spokeswoman on Russian TV has said Jewish pe... |
| 3 | A report by the organisation suggests men, wom... | [ Egyptian police are systematically abusing d... | Egyptian security forces are using sexual viol... |
| 4 | Police in Australia and Europe were aware of a... | [One word and a freckle indirectly led to Huck... | One word and a freckle indirectly led to Huckl... |

```python
# Each list choice contains a positive and negative summary; we'll explode, clean, and drop duplicates
fib_df = fib_df.explode('list_choices')
fib_df['list_choices'] = fib_df['list_choices'].apply(lambda x: x.strip())
fib_df = fib_df.drop_duplicates(subset=['input', 'list_choices'])
fib_df[['input', 'list_choices', 'correct_choice']].head(5)
```

| | input | list_choices | correct_choice |
| --- | --- | --- | --- |
| 0 | Vehicles and pedestrians will now embark and d... | A new service on the Isle of Wight's chain fer... | Passengers using a chain ferry have been warne... |
| 0 | Vehicles and pedestrians will now embark and d... | Passengers using a chain ferry have been warne... | Passengers using a chain ferry have been warne... |
| 1 | If you leave your mobile phone somewhere do yo... | You may be worried about your health, but what... | Do you ever feel lonely, stressed or jealous w... |
| 1 | If you leave your mobile phone somewhere do yo... | Do you ever feel lonely, stressed or jealous w... | Do you ever feel lonely, stressed or jealous w... |
| 2 | Speaking on TV, Maria Zakharova said Jews had ... | The Russian foreign minister has said she has ... | A spokeswoman on Russian TV has said Jewish pe... |

We are going to map to 0 and 1 for the sake of it!

```python
# Create labels where factually consistent = 2 (entailment) and factually inconsistent = 0 (contradiction)
# What happened to label = 1? We drop it as it represents neutral in the NLI task
fib_df.loc[fib_df['correct_choice'] == fib_df['list_choices'], 'target'] = 1
fib_df.loc[fib_df['correct_choice'] != fib_df['list_choices'], 'target'] = 0
fib_df['target'] = fib_df['target'].astype(int)

fib_df[['input', 'list_choices', 'correct_choice', 'target']].head()
```

| | input | list_choices | correct_choice | target |
| --- | --- | --- | --- | --- |
| 0 | Vehicles and pedestrians will now embark and d... | A new service on the Isle of Wight's chain fer... | Passengers using a chain ferry have been warne... | 0 |
| 0 | Vehicles and pedestrians will now embark and d... | Passengers using a chain ferry have been warne... | Passengers using a chain ferry have been warne... | 1 |
| 1 | If you leave your mobile phone somewhere do yo... | You may be worried about your health, but what... | Do you ever feel lonely, stressed or jealous w... | 0 |
| 1 | If you leave your mobile phone somewhere do yo... | Do you ever feel lonely, stressed or jealous w... | Do you ever feel lonely, stressed or jealous w... | 1 |
| 2 | Speaking on TV, Maria Zakharova said Jews had ... | The Russian foreign minister has said she has ... | A spokeswoman on Russian TV has said Jewish pe... | 0 |

```python
# Split into train and val, ensuring that the same source doc doesn't appear across train and val
source_grouped = (fib_df.groupby('input')
                  .agg({'target': 'count'})
                  .reset_index())

input_train, input_val = train_test_split(source_grouped,
                                          test_size=0.18, # mistral fine-tune doesn't want more than 20%
                                          stratify=source_grouped['target'],
                                          random_state=1368)

# input_test, input_val = train_test_split(input_val,
#                                          test_size=0.5,
#                                          stratify=input_val['target'],
#                                          random_state=1368)

fib_train = fib_df[fib_df['input'].isin(input_train['input'])]
fib_val = fib_df[fib_df['input'].isin(input_val['input'])]
# fib_test = fib_df[fib_df['input'].isin(input_test['input'])]
```

```python
# NOTE: In FIB, each doc has 1 positive summary and 5-6 negative summaries. We'll balance it to 1 is to 1.
fib_train = fib_train.drop_duplicates(subset=['input', 'target'])
fib_val = fib_val.drop_duplicates(subset=['input', 'target'])
# fib_test = fib_test.drop_duplicates(subset=['input', 'target'])
```

```python
fib_train.to_csv('data/fib-train.csv', index=False)
fib_val.to_csv('data/fib-val.csv', index=False)
# fib_test.to_csv('data/fib-test.csv', index=False)
```

```python
len(fib_train)
```

    820

```python
fib_train.iloc[0]
```

    id                                                           32168497
    input               Vehicles and pedestrians will now embark and d...
    correct_choice      Passengers using a chain ferry have been warne...
    list_choices        A new service on the Isle of Wight's chain fer...
    lbl                                                                 1
    distractor_model                                            bart-base
    dataset                                                          xsum
    target                                                              0
    Name: 0, dtype: object

```python
# Test loading into dataset
fib_files = {'train': 'data/fib-train.csv',
             'val': 'data/fib-val.csv'}

fib_ds = load_dataset('csv', data_files=fib_files)
fib_ds = fib_ds.select_columns(['input', 'list_choices', 'target'])
fib_ds = fib_ds.rename_column('input', 'premise').rename_column('list_choices', 'hypothesis')
```

    Generating train split: 820 examples [00:00, 23944.92 examples/s]
    Generating val split: 180 examples [00:00, 20254.73 examples/s]

```python
fib_ds["train"][5]
```

    {'premise': 'Speaking on TV, Maria Zakharova said Jews had told her they donated both to Mr Trump and Hillary Clinton.\nShe joked that American Jews were the best guide to US politics.\nThe diplomat\'s remarks caused shock. Anti-US propagandists in the last century peddled an idea that rich New York Jews controlled US politics.\nMs Zakharova was speaking on a chat show on Russian state TV at the weekend but her comments drew more attention after being picked up by media outlets on Thursday.\nShe said she had visited New York with an official Russian delegation at the time of the last UN General Assembly, in September.\n"I have a lot of friends and acquaintances there, of course I was interested to find out: how are the elections going, what are the American people\'s expectations?" she said.\n"If you want to know what will happen in America, who do you need to talk to? You have to talk to the Jews, of course. It goes without saying."\nAt this, the TV studio audience applauded loudly.\n"I went here and there among them, to chat," she continued.\nImitating a Jewish accent, Mrs Zakharova said Jewish people had told her: "\'Marochka, understand this - we\'ll donate to Clinton, of course. But we\'ll give the Republicans twice that amount.\' Enough said! That settled it for me - the picture was clear.\n"If you want to know the future, don\'t read the mainstream newspapers - our people in Brighton [Beach] will tell you everything."\nShe was referring to a district of Brooklyn with a large diaspora of Jewish emigres from the former Soviet Union.\nRussian opposition activist Roman Dobrokhotov wrote on Twitter (in Russian) that the spokeswoman had "explained Trump\'s victory as a Jewish conspiracy".\nMichael McFaul, the former US ambassador to Moscow, commented on Facebook, "Wow. And this is the woman who criticizes me for not being diplomatic."\nDuring the election campaign, Mrs Clinton accused Mr Trump of posting a "blatantly anti-Semitic" tweet after he used an image resembling the Star of David and stacks of money.\nMr Trump, whose son-in-law Jared Kushner is Jewish, dismissed the accusation as "ridiculous".\nAn exit poll by US non-profit J Street suggests an overwhelming majority of US Jews voted for Hillary Clinton in the presidential election.',
     'hypothesis': 'A spokeswoman on Russian TV has said Jewish people in New York told her they had mainly backed Trump in the US election.',
     'target': 1}

```python
ds_train = fib_ds["train"]
ds_val = fib_ds["val"]
# ds_test = fib_ds["test"]
```

```python
ds_train.to_json("data/fib-train.jsonl", orient="records", lines=True)
ds_val.to_json("data/fib-val.jsonl", orient="records", lines=True)
# ds_test.to_json("data/fib-test.jsonl", orient="records", lines=True)
```

    Creating json from Arrow format: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 126.79ba/s]
    Creating json from Arrow format: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 510.19ba/s]

    274757

## USB

```python
usb_files = {'train': 'data/usb-train.csv',
             'val': 'data/usb-val.csv'}

usb_ds = load_dataset('csv', data_files=usb_files)
usb_ds = usb_ds.select_columns(['source', 'summary_sent', 'label'])
usb_ds = usb_ds.rename_column('source', 'premise').rename_column('summary_sent', 'hypothesis')
usb_ds = usb_ds.map(lambda x: {"label": 1 if x["label"] == 2 else x["label"]})
usb_ds = usb_ds.rename_column('label', 'target')

usb_ds_train = usb_ds["train"]
usb_ds_val = usb_ds["val"]
usb_ds_train.to_json("data/usb-train.jsonl", orient="records", lines=True)
usb_ds_val.to_json("data/usb-val.jsonl", orient="records", lines=True)
```

    Generating train split: 5050 examples [00:00, 70732.38 examples/s]
    Generating val split: 2668 examples [00:00, 108077.02 examples/s]
    Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5050/5050 [00:00<00:00, 67926.67 examples/s]
    Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2668/2668 [00:00<00:00, 65323.30 examples/s]
    Creating json from Arrow format: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 325.66ba/s]
    Creating json from Arrow format: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 204.83ba/s]

    1603234

```python
usb_ds_train
```

    Dataset({
        features: ['premise', 'hypothesis', 'target'],
        num_rows: 5050
    })