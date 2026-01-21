# Guide: Signature-Aware Model Serving from MLflow with Ray Serve

_Authored by: [Jonathan Jin](https://huggingface.co/jinnovation)_

## Introduction

This guide explores how to streamline the deployment of models from a model registry. For teams that need to productionize many models over time, investing in this "transition point" in the AI/ML lifecycle can significantly reduce time-to-production. This is especially valuable for smaller teams without existing infrastructure to form a "golden path" for serving models online.

## Motivation

Optimizing this stage is crucial because your model becomes a production microservice. This introduces new responsibilities:
- Ensuring API backwards-compatibility
- Implementing logging, metrics, and observability
- Managing multiple model versions efficiently

Repeating the same setup for each new model leads to significant development costs. Conversely, streamlining this process pays dividends over the long lifecycle of production models.

Our goal is to enable deployment from a model registry (like MLflow) using **only the model name**, minimizing boilerplate and allowing dynamic version selection without new deployments.

## Prerequisites

First, install the required dependencies:

```bash
pip install "transformers" "mlflow-skinny" "ray[serve]" "torch"
```

## Step 1: Define and Register a Model

We'll use a text translation model where source and destination languages are configurable at registration time. Different "versions" can translate different languages while using the same underlying architecture.

### 1.1 Create the Model Class

```python
import mlflow
from transformers import pipeline

class MyTranslationModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.lang_from = context.model_config.get("lang_from", "en")
        self.lang_to = context.model_config.get("lang_to", "de")
        self.input_label = context.model_config.get("input_label", "prompt")
        self.model_ref = context.model_config.get("hfhub_name", "google-t5/t5-base")
        
        self.pipeline = pipeline(
            f"translation_{self.lang_from}_to_{self.lang_to}",
            self.model_ref,
        )

    def predict(self, context, model_input, params=None):
        prompt = model_input[self.input_label].tolist()
        return self.pipeline(prompt)
```

### 1.2 Register the First Model Version

Register a version that translates from **English to German** using Google's T5 Base model:

```python
import pandas as pd

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        "translation_model",
        registered_model_name="translation_model",
        python_model=MyTranslationModel(),
        pip_requirements=["transformers"],
        input_example=pd.DataFrame({
            "prompt": ["Hello my name is Jonathan."],
        }),
        model_config={
            "hfhub_name": "google-t5/t5-base",
            "lang_from": "en",
            "lang_to": "de",
        },
    )

# Save the version for later use
en_to_de_version = str(model_info.registered_model_version)
```

### 1.3 Examine the Model Signature

MLflow automatically captures the model's input/output signature:

```python
print(model_info.signature)
```

The output shows:
```
inputs: 
  ['prompt': string (required)]
outputs: 
  ['translation_text': string (required)]
params: 
  None
```

This signature will be crucial for ensuring API consistency later.

## Step 2: Basic Model Serving with Ray Serve

Now let's create a basic deployment that serves the registered model via a REST API.

### 2.1 Create the Deployment Class

```python
import mlflow
import pandas as pd
from ray import serve
from fastapi import FastAPI

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, model_name: str = "translation_model", default_version: str = "1"):
        self.model_name = model_name
        self.default_version = default_version
        self.model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{self.default_version}")

    @app.post("/serve")
    async def serve(self, input_string: str):
        return self.model.predict(pd.DataFrame({"prompt": [input_string]}))

deployment = ModelDeployment.bind(default_version=en_to_de_version)
```

Notice the hidden coupling: the API uses `input_string` while the model expects `"prompt"`. We'll address this later.

### 2.2 Run the Deployment and Test

```python
serve.run(deployment, blocking=False)
```

Test the endpoint:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/serve/",
    params={"input_string": "The weather is lovely today"},
)

print(response.json())
```

Output:
```json
[{'translation_text': 'Das Wetter ist heute nett.'}]
```

## Step 3: Serve Multiple Versions with Model Multiplexing

The basic deployment is tethered to a single model version. Let's use Ray Serve's model multiplexing to serve multiple versions through the same endpoint.

### 3.1 Create a Multiplexed Deployment

```python
from ray import serve
from fastapi import FastAPI
import mlflow
import pandas as pd

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class MultiplexedModelDeployment:
    @serve.multiplexed(max_num_models_per_replica=2)
    async def get_model(self, version: str):
        return mlflow.pyfunc.load_model(f"models:/{self.model_name}/{version}")

    def __init__(
        self,
        model_name: str = "translation_model",
        default_version: str = en_to_de_version,
    ):
        self.model_name = model_name
        self.default_version = default_version

    @app.post("/serve")
    async def serve(self, input_string: str):
        model = await self.get_model(serve.get_multiplexed_model_id())
        return model.predict(pd.DataFrame({"prompt": [input_string]}))

multiplexed_deployment = MultiplexedModelDeployment.bind(model_name="translation_model")
serve.run(multiplexed_deployment, blocking=False)
```

### 3.2 Register a Second Model Version

Register a version that translates from **English to French**:

```python
import pandas as pd

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        "translation_model",
        registered_model_name="translation_model",
        python_model=MyTranslationModel(),
        pip_requirements=["transformers"],
        input_example=pd.DataFrame({
            "prompt": ["Hello my name is Jon."],
        }),
        model_config={
            "hfhub_name": "google-t5/t5-base",
            "lang_from": "en",
            "lang_to": "fr",
        },
    )

en_to_fr_version = str(model_info.registered_model_version)
```

### 3.3 Test Both Versions

Test the French translation version:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/serve/",
    params={"input_string": "The weather is lovely today"},
    headers={"serve_multiplexed_model_id": en_to_fr_version},
)

print(response.json())
```

Output:
```json
[{'translation_text': "Le temps est beau aujourd'hui"}]
```

Test the German translation version (still loaded due to LRU caching):

```python
response = requests.post(
    "http://127.0.0.1:8000/serve/",
    params={"input_string": "The weather is lovely today"},
    headers={"serve_multiplexed_model_id": en_to_de_version},
)

print(response.json())
```

Output:
```json
[{'translation_text': 'Das Wetter ist heute nett.'}]
```

Ray Serve dynamically loads models on-demand and evicts least-recently used versions when exceeding `max_num_models_per_replica`.

## Step 4: Automatic Signature Inference

The current implementation still has API signature duplication. Let's create a deployment that automatically mirrors the MLflow model signature.

### 4.1 Create Signature Conversion Utilities

```python
import mlflow
import pydantic

def schema_to_pydantic(schema: mlflow.types.schema.Schema, *, name: str) -> pydantic.BaseModel:
    return pydantic.create_model(
        name,
        **{
            k: (v.type.to_python(), pydantic.Field(required=True))
            for k, v in schema.input_dict().items()
        }
    )

def get_req_resp_signatures(model_signature: mlflow.models.ModelSignature) -> tuple[pydantic.BaseModel, pydantic.BaseModel]:
    inputs = model_signature.inputs
    outputs = model_signature.outputs
    return (schema_to_pydantic(inputs, name="InputModel"), 
            schema_to_pydantic(outputs, name="OutputModel"))
```

### 4.2 Create a Dynamically Typed Deployment

```python
import mlflow
from fastapi import FastAPI, Response, status
from ray import serve
from typing import List

def deployment_from_model_name(model_name: str, default_version: str = "1"):
    app = FastAPI()
    model_info = mlflow.models.get_model_info(f"models:/{model_name}/{default_version}")
    input_datamodel, output_datamodel = get_req_resp_signatures(model_info.signature)

    @serve.deployment
    @serve.ingress(app)
    class DynamicallyDefinedDeployment:
        MODEL_NAME = model_name
        DEFAULT_VERSION = default_version

        @serve.multiplexed(max_num_models_per_replica=2)
        async def get_model(self, model_version: str):
            model = mlflow.pyfunc.load_model(f"models:/{self.MODEL_NAME}/{model_version}")
            
            # Validate signature compatibility
            if model.metadata.get_model_info().signature != model_info.signature:
                raise ValueError(
                    f"Requested version {model_version} has signature incompatible "
                    f"with default version {self.DEFAULT_VERSION}"
                )
            return model

        @app.post("/serve", response_model=List[output_datamodel])
        async def serve(self, model_input: input_datamodel, response: Response):
            model_id = serve.get_multiplexed_model_id()
            if model_id == "":
                model_id = self.DEFAULT_VERSION

            try:
                model = await self.get_model(model_id)
            except ValueError:
                response.status_code = status.HTTP_409_CONFLICT
                return [{"translation_text": "FAILED"}]

            return model.predict(model_input.dict())

    return DynamicallyDefinedDeployment

# Create and run the deployment
deployment = deployment_from_model_name("translation_model", default_version=en_to_fr_version)
serve.run(deployment.bind(), blocking=False)
```

### 4.3 Test the Signature-Aware Deployment

Test with default version (English to French):

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8000/serve/",
    json={"prompt": "The weather is lovely today"},
)

print(resp.json())
```

Output:
```json
[{'translation_text': "Le temps est beau aujourd'hui"}]
```

Test with explicit version header:

```python
resp = requests.post(
    "http://127.0.0.1:8000/serve/",
    json={"prompt": "The weather is lovely today"},
    headers={"serve_multiplexed_model_id": str(en_to_fr_version)},
)

print(resp.json())
```

Output:
```json
[{'translation_text': "Le temps est beau aujourd'hui"}]
```

## Step 5: Validate Signature Compatibility

Let's verify that incompatible signatures are properly rejected.

### 5.1 Register an Incompatible Version

Register a version with a different input label (`text_to_translate` instead of `prompt`):

```python
import pandas as pd

with mlflow.start_run():
    incompatible_version = str(mlflow.pyfunc.log_model(
        "translation_model",
        registered_model_name="translation_model",
        python_model=MyTranslationModel(),
        pip_requirements=["transformers"],
        input_example=pd.DataFrame({
            "text_to_translate": ["Hello my name is Jon."],
        }),
        model_config={
            "input_label": "text_to_translate",
            "hfhub_name": "google-t5/t5-base",
            "lang_from": "en",
            "lang_to": "de",
        },
    ).registered_model_version)
```

### 5.2 Test the Incompatible Version

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8000/serve/",
    json={"prompt": "The weather is lovely today"},
    headers={"serve_multiplexed_model_id": incompatible_version},
)

print(f"Status Code: {resp.status_code}")
print(f"Response: {resp.json()}")
```

The request should fail with a 409 Conflict status code, demonstrating that our signature validation works correctly.

## Conclusion

You've successfully built a signature-aware model serving system that:

1. **Minimizes boilerplate**: Deploy models using only their registry name
2. **Supports version multiplexing**: Serve multiple versions through a single endpoint
3. **Ensures API consistency**: Automatically infer API signatures from MLflow metadata
4. **Validates compatibility**: Reject incompatible model versions at runtime

This approach significantly reduces the operational overhead of deploying and maintaining multiple model versions in production, while ensuring consistent and predictable APIs for downstream consumers.

## Next Steps

Consider extending this solution with:
- Batch inference support for higher throughput
- Advanced monitoring and metrics collection
- Authentication and authorization layers
- Automated canary deployments and A/B testing