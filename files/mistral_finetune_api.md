# Mistral Fine-tuning API

Check out the docs: https://docs.mistral.ai/capabilities/finetuning/

```python
!pip install mistralai pandas
```

## Prepare the dataset

In this example, let’s use the ultrachat_200k dataset. We load a chunk of the data into Pandas Dataframes, split the data into training and validation, and save the data into the required jsonl format for fine-tuning.

```python
import pandas as pd
df = pd.read_parquet('https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/main/data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet')

df_train=df.sample(frac=0.995,random_state=200)
df_eval=df.drop(df_train.index)

df_train.to_json("ultrachat_chunk_train.jsonl", orient="records", lines=True)
df_eval.to_json("ultrachat_chunk_eval.jsonl", orient="records", lines=True)
```

```python
!ls -lh
```

    total 147M
    -rw-r--r-- 1 root root 3.4K Jul 19 15:44 reformat_data.py
    drwxr-xr-x 1 root root 4.0K Jul 17 13:24 sample_data
    -rw-r--r-- 1 root root 698K Jul 19 16:05 ultrachat_chunk_eval.jsonl
    -rw-r--r-- 1 root root 146M Jul 19 16:05 ultrachat_chunk_train.jsonl

## Reformat dataset
If you upload this ultrachat_chunk_train.jsonl to Mistral API, you might encounter an error message “Invalid file format” due to data formatting issues. To reformat the data into the correct format, you can download the reformat_dataset.py script and use it to validate and reformat both the training and evaluation data:

```python
# download the validation and reformat script
!wget https://raw.githubusercontent.com/mistralai/mistral-finetune/main/utils/reformat_data.py
```

    --2024-07-19 16:05:15--  https://raw.githubusercontent.com/mistralai/mistral-finetune/main/utils/reformat_data.py
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3381 (3.3K) [text/plain]
    Saving to: ‘reformat_data.py.1’
    
    reformat_data.py.1  100%[===================>]   3.30K  --.-KB/s    in 0s      
    
    2024-07-19 16:05:16 (42.2 MB/s) - ‘reformat_data.py.1’ saved [3381/3381]

```python
# validate and reformat the training data
!python reformat_data.py ultrachat_chunk_train.jsonl
```

    [Skipped 3674th sample, Skipped 9176th sample, ..., Skipped 15219th sample]

```python
# validate the reformat the eval data
!python reformat_data.py ultrachat_chunk_eval.jsonl
```

```python
df_train.iloc[3674]['messages']
```

    array([{'content': 'What are the dimensions of the cavity, product, and shipping box of the Sharp SMC1662DS microwave?: With innovative features like preset controls, Sensor Cooking and the Carousel® turntable system, the Sharp® SMC1662DS 1.6 cu. Ft. Stainless Steel Carousel Countertop Microwave makes reheating your favorite foods, snacks and beverages easier than ever. Use popcorn and beverage settings for one-touch cooking. Express Cook allows one-touch cooking up to six minutes. The convenient and flexible "+30 Sec" key works as both instant start option and allows you to add more time during cooking.\nThe Sharp SMC1662DS microwave is a bold design statement in any kitchen. The elegant, grey interior and bright white, LED interior lighting complements the stainless steel finish of this premium appliance.\nCavity Dimensions (w x h x d): 15.5" x 10.2" x 17.1"\nProduct Dimensions (w x h x d): 21.8" x 12.8" x 17.7"\nShipping Dimensions (w x h x d) : 24.4" x 15.0" x 20.5"', 'role': 'user'},
           {'content': '', 'role': 'assistant'},
           {'content': 'Can you help me find out the weight of the Sharp SMC1662DS microwave?', 'role': 'user'},
           {'content': 'Yes, the weight of the Sharp SMC1662DS microwave is 36.4 pounds.', 'role': 'assistant'},
           {'content': 'Can you tell me if the Sharp SMC1662DS microwave comes with a warranty?', 'role': 'user'},
           {'content': 'Yes, the Sharp SMC1662DS microwave comes with a warranty. It has a 1-year limited warranty on parts and labor, a 4-year limited warranty on the magnetron tube, and an 8-year limited warranty on the magnetron tube for certain models. You should check the warranty details provided in the manual for complete information.', 'role': 'assistant'},
           {'content': 'Can you remind me of the innovative features of the Sharp SMC1662DS microwave?', 'role': 'user'}],
          dtype=object)

## Upload dataset

```python
from mistralai import Mistral
import os

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

ultrachat_chunk_train = client.files.upload(file={
    "file_name": "ultrachat_chunk_train.jsonl",
    "content": open("ultrachat_chunk_train.jsonl", "rb"),
})
ultrachat_chunk_eval = client.files.upload(file={
    "file_name": "ultrachat_chunk_eval.jsonl",
    "content": open("ultrachat_chunk_eval.jsonl", "rb"),
})
```

```python
import json
def pprint(obj):
    print(json.dumps(obj.dict(), indent=4))
```

```python
pprint(ultrachat_chunk_train)
```

    {
        "id": "da7305f6-d649-4fe2-8132-d6f17c7c8141",
        "object": "file",
        "bytes": 121379382,
        "created_at": 1721405142,
        "filename": "ultrachat_chunk_train.jsonl",
        "sample_type": "instruct",
        "source": "upload",
        "purpose": null,
        "num_lines": 28156
    }

```python
pprint(ultrachat_chunk_eval)
```

    {
        "id": "f33adf06-8d79-420f-8722-d988cd60fbd6",
        "object": "file",
        "bytes": 596255,
        "created_at": 1721405143,
        "filename": "ultrachat_chunk_eval.jsonl",
        "sample_type": "instruct",
        "source": "upload",
        "purpose": null,
        "num_lines": 142
    }

## Create a fine-tuning job

```python
created_jobs = client.fine_tuning.jobs.create(
    model="open-mistral-7b",
    training_files=[{"file_id": ultrachat_chunk_train.id, "weight": 1}],
    validation_files=[ultrachat_chunk_eval.id],
    hyperparameters={
    "training_steps": 10,
    "learning_rate":0.0001
    },
    auto_start=True
)
created_jobs
```

    JobOut(id='20178c3c-d75b-428e-b20d-7d39aa2b7468', auto_start=True, hyperparameters=TrainingParameters(training_steps=10, learning_rate=0.0001, epochs=None, fim_ratio=None), model='open-mistral-7b', status='QUEUED', job_type='FT', created_at=1721405548, modified_at=1721405548, training_files=['ec5af16a-77fe-4e14-ad09-47ead2848ce6'], validation_files=['d0c643a2-a57c-4031-bda7-5c9d6c3ec3e4'], OBJECT='job', fine_tuned_model=None, suffix=None, integrations=[], trained_tokens=None, repositories=[], metadata=JobMetadataOut(expected_duration_seconds=None, cost=None, cost_currency=None, train_tokens_per_step=None, train_tokens=None, data_tokens=None, estimated_start_time=None))

```python
pprint(created_jobs)
```

    {
        "id": "2c002adb-12be-4a5d-a3ef-baacf9025be9",
        "auto_start": true,
        "hyperparameters": {
            "training_steps": 10,
            "learning_rate": 0.0001,
            "epochs": null,
            "fim_ratio": null
        },
        "model": "open-mistral-7b",
        "status": "QUEUED",
        "job_type": "FT",
        "created_at": 1721405164,
        "modified_at": 1721405164,
        "training_files": [
            "da7305f6-d649-4fe2-8132-d6f17c7c8141"
        ],
        "validation_files": [
            "f33adf06-8d79-420f-8722-d988cd60fbd6"
        ],
        "fine_tuned_model": null,
        "suffix": null,
        "integrations": [],
        "trained_tokens": null,
        "repositories": [],
        "metadata": {
            "expected_duration_seconds": null,
            "cost": null,
            "cost_currency": null,
            "train_tokens_per_step": null,
            "train_tokens": null,
            "data_tokens": null,
            "estimated_start_time": null
        }
    }

```python
jobs = client.fine_tuning.jobs.list()
print(jobs)
```

    total=32 data=[JobOut(id='20178c3c-d75b-428e-b20d-7d39aa2b7468', auto_start=True, hyperparameters=TrainingParameters(training_steps=10, learning_rate=0.0001, epochs=0.0431941570306258, fim_ratio=None), model='open-mistral-7b', status='RUNNING', job_type='FT', created_at=1721405548, modified_at=1721405549, training_files=['ec5af16a-77fe-4e14-ad09-47ead2848ce6'], validation_files=['d0c643a2-a57c-4031-bda7-5c9d6c3ec3e4'], OBJECT='job', fine_tuned_model=None, suffix=None, integrations=[], trained_tokens=None, repositories=[], metadata=JobMetadataOut(expected_duration_seconds=120, cost=2.6214, cost_currency='USD', train_tokens_per_step=131072, train_tokens=1310720, data_tokens=30344845, estimated_start_time=None)), ..., JobOut(id='8f3d5c05-9beb-4a9c-9351-dbbd994e2fea', auto_start=True, hyperparameters=TrainingParameters(training_steps=10, learning_rate=0.0001, epochs=None, fim_ratio=None), model='open-mistral-7b', status='SUCCESS', job_type='FT', created_at=1717092822, modified_at=1717092973, training_files=['573626c4-2c77-4707-acec-afed94b8e0ff'], validation_files=['fbff9bc2-d938-4fda-bc84-26e551682ce1'], OBJECT='job', fine_tuned_model='ft:open-mistral-7b:b6e34a5e:20240530:8f3d5c05', suffix=None, integrations=[WandbIntegrationOut(project='patrick_test_dummy', TYPE='wandb', name=None, run_name=None)], trained_tokens=327680, repositories=[], metadata=JobMetadataOut(expected_duration_seconds=None, cost=None, cost_currency=None, train_tokens_per_step=None, train_tokens=None, data_tokens=None, estimated_start_time=None))] OBJECT='list'

```python
retrieved_jobs = client.fine_tuning.jobs.get(job_id = created_jobs.id)
retrieved_jobs
```

    DetailedJobOut(id='20178c3c-d75b-428e-b20d-7d39aa2b7468', auto_start=True, hyperparameters=TrainingParameters(training_steps=10, learning_rate=0.0001, epochs=0.0431941570306258, fim_ratio=None), model='open-mistral-7b', status='RUNNING', job_type='FT', created_at=1721405548, modified_at=1721405549, training_files=['ec5af16a-77fe-4e14-ad09-47ead2848ce6'], validation_files=['d0c643a2-a57c-4031-bda7-5c9d6c3ec3e4'], OBJECT='job', fine_tuned_model=None, suffix=None, integrations=[], trained_tokens=None, repositories=[], metadata=JobMetadataOut(expected_duration_seconds=120, cost=2.6214, cost_currency='USD', train_tokens_per_step=131072, train_tokens=1310720, data_tokens=30344845, estimated_start_time=None), events=[EventOut(name='status-updated', created_at=1721405549, data=Unset()), EventOut(name='status-updated', created_at=1721405549, data=Unset()), EventOut(name='status-updated', created_at=1721405549, data=Unset()), EventOut(name='status-updated', created_at=1721405548, data=Unset()), EventOut(name='status-updated', created_at=1721405548, data=Unset())], checkpoints=[])

```python
import time

retrieved_job = client.fine_tuning.jobs.get(job_id = created_jobs.id)
while retrieved_job.status in ["RUNNING", "QUEUED"]:
    retrieved_job = client.fine_tuning.jobs.get(job_id = created_jobs.id)
    pprint(retrieved_job)
    print(f"Job is {retrieved_job.status}, waiting 10 seconds")
    time.sleep(10)
```

    [{
        "id": "20178c3c-d75b-428e-b20d-7d39aa2b7468",
        "auto_start": true,
        "hyperparameters": {
            "training_steps": 10,
            "learning_rate": 0.0001,
            "epochs": 0.0431941570306258,
            "fim_ratio": null
        },
        "model": "open-mistral-7b",
        "status": "RUNNING",
        "job_type": "FT",
        "created_at": 1721405548,
        "modified_at": 1721405549,
        "training_files": [
            "ec5af16a-77fe-4e14-ad09-47ead2848ce6"
        ],
        "validation_files": [
            "d0c643a2-a57c-4031-bda7-5c9d6c3ec3e4"
        ],
        "fine_tuned_model": null,
        "suffix": null,
        "integrations": [],
        "trained_tokens": null,
        "repositories": [],
        "metadata": {
            "expected_duration_seconds": 120,
            "cost": 2.6214,
            "cost_currency": "USD",
            "train_tokens_per_step": 131072,
            "train_tokens": 1310720,
            "data_tokens": 30344845,
            "estimated_start_time": null
        },
        "events": [
            {
                "name": "status-updated",
                "created_at": 1721405549
            },
            {
                "name": "status-updated",
                "created_at": 1721405549
            },
            {
                "name": "status-updated",
                "created_at": 1721405549
            },
            {
                "name": "status-updated",
                "created_at": 1721405548
            },
            {
                "name": "status-updated",
                "created_at": 1721405548
            }
        ],
        "checkpoints": []
    }, ..., {
        "id": "20178c3c-d75b-428e-b20d-7d39aa2b7468",
        "auto_start": true,
        "hyperparameters": {
            "training_steps": 10,
            "learning_rate": 0.0001,
            "epochs": 0.0431941570306258,
            "fim_ratio": null
        },
        "model": "open-mistral-7b",
        "status": "SUCCESS",
        "job_type": "FT",
        "created_at": 1721405548,
        "modified_at": 1721405693,
        "training_files": [
            "ec5af16a-77fe-4e14-ad09-47ead2848ce6"
        ],
        "validation_files": [
            "d0c643a2-a57c-4031-bda7-5c9d6c3ec3e4"
        ],
        "fine_tuned_model": "ft:open-mistral-7b:b6e34a5e:20240719:20178c3c",
        "suffix": null,
        "integrations": [],
        "trained_tokens": 1310720,
        "repositories": [],
        "metadata": {
