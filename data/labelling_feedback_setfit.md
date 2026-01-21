# Guide: Zero-shot Text Classification with SetFit for Data Annotation

This guide demonstrates how to use SetFit to generate zero-shot classification suggestions for a dataset in Argilla. By pre-labeling data with model predictions, you can significantly accelerate your annotation workflow. We'll combine two text classification tasks: a single-label sentiment analysis and a multi-label topic classification.

## Prerequisites

Before starting, ensure you have an Argilla server running. You can deploy it locally or on Hugging Face Spaces by following the [official quickstart guide](https://docs.argilla.io/latest/getting_started/quickstart/).

## Step 1: Install Dependencies

Install the required Python libraries:

```bash
pip install argilla
pip install setfit==1.0.3 transformers==4.40.2 huggingface_hub==0.23.5
```

## Step 2: Import Libraries and Initialize Argilla

Import the necessary modules and initialize the Argilla client. If you're using a private Hugging Face Space, uncomment and set the `headers` parameter.

```python
import argilla as rg
from datasets import load_dataset
from setfit import SetFitModel, Trainer, get_templated_dataset

# Replace with your actual API URL and key
client = rg.Argilla(
    api_url="https://[your-owner-name]-[your_space_name].hf.space",
    api_key="[your-api-key]",
    # headers={"Authorization": f"Bearer {HF_TOKEN}"}  # Uncomment for private spaces
)
```

## Step 3: Load and Prepare Your Dataset

We'll use the `banking77` dataset, which contains customer service requests in the banking domain.

```python
data = load_dataset("PolyAI/banking77", split="test")
```

## Step 4: Configure the Argilla Dataset

Define the dataset structure in Argilla. We'll create a text field and two questions: one for multi-label topic classification and another for single-label sentiment analysis.

```python
settings = rg.Settings(
    fields=[rg.TextField(name="text")],
    questions=[
        rg.MultiLabelQuestion(
            name="topics",
            title="Select the topic(s) of the request",
            labels=data.info.features["label"].names,
            visible_labels=10,
        ),
        rg.LabelQuestion(
            name="sentiment",
            title="What is the sentiment of the message?",
            labels=["positive", "neutral", "negative"],
        ),
    ],
)

dataset = rg.Dataset(
    name="setfit_tutorial_dataset",
    settings=settings,
)
dataset.create()
```

## Step 5: Train Zero-shot Classifiers with SetFit

We'll train two separate SetFit models: one for topics (multi-label) and one for sentiment (single-label). The function below handles the training process.

```python
def train_model(question_name, template, multi_label=False):
    # Generate a templated training dataset
    train_dataset = get_templated_dataset(
        candidate_labels=dataset.questions[question_name].labels,
        sample_size=8,
        template=template,
        multi_label=multi_label,
    )

    # Load the base sentence transformer model
    if multi_label:
        model = SetFitModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            multi_target_strategy="one-vs-rest",
        )
    else:
        model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Train the model
    trainer = Trainer(model=model, train_dataset=train_dataset)
    trainer.train()

    return model
```

Now, train both models:

```python
# Train the multi-label topic classifier
topic_model = train_model(
    question_name="topics",
    template="The customer request is about {}",
    multi_label=True,
)

# Train the single-label sentiment classifier
sentiment_model = train_model(
    question_name="sentiment",
    template="This message is {}",
    multi_label=False
)
```

Optionally, save your trained models for later use:

```python
# topic_model.save_pretrained("/path/to/topic_model")
# sentiment_model.save_pretrained("/path/to/sentiment_model")
```

## Step 6: Generate Predictions on Your Data

Create a helper function to obtain prediction probabilities for each text, then apply it to the dataset.

```python
def get_predictions(texts, model, question_name):
    probas = model.predict_proba(texts, as_numpy=True)
    labels = dataset.questions[question_name].labels
    for pred in probas:
        yield [{"label": label, "score": score} for label, score in zip(labels, pred)]

# Add prediction columns to the dataset
data = data.map(
    lambda batch: {
        "topics": list(get_predictions(batch["text"], topic_model, "topics")),
        "sentiment": list(get_predictions(batch["text"], sentiment_model, "sentiment")),
    },
    batched=True,
)
```

## Step 7: Create Argilla Records with Suggestions

Convert predictions into Argilla suggestions. For sentiment, we select the label with the highest score. For topics, we include all labels above a dynamic threshold.

```python
def add_suggestions(record):
    suggestions = []

    # Sentiment: choose label with maximum score
    sentiment = max(record["sentiment"], key=lambda x: x["score"])["label"]
    suggestions.append(rg.Suggestion(question_name="sentiment", value=sentiment))

    # Topics: include labels above threshold (2/num_labels)
    threshold = 2 / len(dataset.questions["topics"].labels)
    topics = [
        label["label"] for label in record["topics"] if label["score"] >= threshold
    ]
    if topics:
        suggestions.append(rg.Suggestion(question_name="topics", value=topics))
    
    return suggestions

# Build records with text and suggestions
records = [
    rg.Record(fields={"text": record["text"]}, suggestions=add_suggestions(record))
    for record in data
]
```

## Step 8: Upload Records to Argilla

Log the records to your Argilla dataset. The suggestions will now be visible in the UI, ready for review and correction by annotators.

```python
dataset.records.log(records)
```

## Step 9: Export and Share (Optional)

You can export your annotated dataset to the Hugging Face Hub for collaboration or further use.

```python
# Export to Hugging Face Hub
dataset.to_hub(repo_id="argilla/my_setfit_dataset")

# Import from Hugging Face Hub
dataset = rg.Dataset.from_hub(repo_id="argilla/my_setfit_dataset")
```

## Conclusion

You've successfully built a pipeline to generate zero-shot classification suggestions using SetFit and Argilla. This approach reduces the manual effort required during annotation by providing pre-labeled suggestions. Experiment with different thresholds and templates to optimize suggestion quality for your specific use case.

### Additional Resources
- [Argilla Documentation](https://docs.argilla.io/latest/)
- [SetFit GitHub Repository](https://github.com/huggingface/setfit)
- [SetFit Documentation](https://huggingface.co/docs/setfit/index)