# Signature-Aware Model Serving from MLflow with Ray Serve

_Authored by: [Jonathan Jin](https://huggingface.co/jinnovation)_

## Introduction

This notebook explores solutions for streamlining the deployment of models from a model registry. For teams that want to productionize many models over time, investments at this "transition point" in the AI/ML project lifecycle can meaningfully drive down time-to-production. This can be important for a younger, smaller team that may not have the benefit of existing infrastructure to form a "golden path" for serving online models in production.

## Motivation

Optimizing this stage of the model lifecycle is particularly important due to the production-facing aspect of the end result. At this stage, your model becomes, in effect, a microservice. This means that you now need to contend with all elements of service ownership, which can include:

- Standardizing and enforcing API backwards-compatibility;
- Logging, metrics, and general observability concerns;
- Etc.

Needing to repeat the same general-purpose setup each time you want to deploy a new model will result in development costs adding up significantly over time for you and your team. On the flip side, given the "long tail" of production-model ownership (assuming a productionized model is not likely to be decommissioned anytime soon), streamlining investments here can pay healthy dividends over time.

Given all of the above, we motivate our exploration here with the following user story:

> I would like to deploy a model from a model registry (such as [MLflow](https://mlflow.org/)) using **only the name of the model**. The less boilerplate and scaffolding that I need to replicate each time I want to deploy a new model, the better. I would like the ability to dynamically select between different versions of the model without needing to set up a whole new deployment to accommodate those new versions.

## Components

For our exploration here, we'll use the following minimal stack:

- MLflow for model registry;
- Ray Serve for model serving.

For demonstrative purposes, we'll exclusively use off-the-shelf open-source models from Hugging Face Hub.

We will **not** use GPUs for inference because inference performance is orthogonal to our focus here today. Needless to say, in "real life," you will likely not be able to get away with serving your model with CPU compute.

Let's install our dependencies now.

```python
!pip install "transformers" "mlflow-skinny" "ray[serve]" "torch"
```

## Register the Model

First, let's define the model that we'll use for our exploration today. For simplicity's sake, we'll use a simple text translation model, where the source and destination languages are configurable at registration time. In effect, this means that different "versions" of the model can be registered to translate different languages, but the underlying model architecture and weights can stay the same.

```python
import mlflow
from transformers import pipeline

class MyTranslationModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.lang_from = context.model_config.get("lang_from", "en")
        self.lang_to = context.model_config.get("lang_to", "de")

        self.input_label: str = context.model_config.get("input_label", "prompt")

        self.model_ref: str = context.model_config.get("hfhub_name", "google-t5/t5-base")

        self.pipeline = pipeline(
            f"translation_{self.lang_from}_to_{self.lang_to}",
            self.model_ref,
        )

    def predict(self, context, model_input, params=None):
        prompt = model_input[self.input_label].tolist()

        return self.pipeline(prompt)
```

(You might be wondering why we even bothered making the input label configurable. This will be useful to us later.)

Now that our model is defined, let's register an actual version of it. This particular version will use Google's [T5 Base](https://huggingface.co/google-t5/t5-base) model and be configured to translate from **English** to **German**.

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
```

Let's keep track of this exact version. This will be useful later.

```python
en_to_de_version: str = str(model_info.registered_model_version)
```

The registered model metadata contains some useful information for us. Most notably, the registered model version is associated with a strict **signature** that denotes the expected shape of its input and output. This will be useful to us later.

```python
print(model_info.signature)
```

    inputs: 
      ['prompt': string (required)]
    outputs: 
      ['translation_text': string (required)]
    params: 
      None

## Serve the Model

Now that our model is registered in MLflow, let's set up our serving scaffolding using [Ray Serve](https://docs.ray.io/en/latest/serve/index.html). For now, we'll limit our "deployment" to the following behavior:

- Source the seleted model and version from MLflow;
- Receive inference requests and return inference responses via a simple REST API.

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

You might have notice that hard-coding `"prompt"` as the input label here introduces hidden coupling between the registered model's signature and the deployment implementation. We'll come back to this later.

Now, let's run the deployment and play around with it.

```python
serve.run(deployment, blocking=False)
```

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/serve/",
    params={"input_string": "The weather is lovely today"},
)

print(response.json())
```

    [{'translation_text': 'Das Wetter ist heute nett.'}]

This works fine, but you might have noticed that the REST API does not line up with the model signature. Namely, it uses the label `"input_string"` while the served model version itself uses the input label `"prompt"`. Similarly, the model can accept multiple inputs values, but the API only accepts one.

If this feels [smelly](https://en.wikipedia.org/wiki/Code_smell) to you, keep reading; we'll come back to this.

## Multiple Versions, One Endpoint

Now we've got a basic endpoint set up for our model. Great! However, notice that this deployment is strictly tethered to a single version of this model -- specifically, version `1` of the registered `translation_model`.

Imagine, now, that your team would like to come back and refine this model -- maybe retrain it on new data, or configure it to translate to a new language, e.g. French instead of German. Both would result in a new version of the `translation_model` getting registered. However, with our current deployment implementation, we'd need to set up a whole new endpoint for `translation_model/2`, require our users to remember which address and port corresponds to which version of the model, and so on. In other words: very cumbersome, very error-prone, very [toilsome](https://leaddev.com/velocity/what-toil-and-why-it-damaging-your-engineering-org).

Conversely, imagine a scenario where we could reuse the exact same endpoint -- same signature, same address and port, same query conventions, etc. -- to serve both versions of this model. Our user can simply specify which version of the model they'd like to use, and we can treat one of them as the "default" in cases where the user didn't explicitly request one.

This is one area where Ray Serve shines with a feature it calls [model multiplexing](https://docs.ray.io/en/latest/serve/model-multiplexing.html). In effect, this allows you to load up multiple "versions" of your model, dynamically hot-swapping them as needed, as well as unloading the versions that don't get used after some time. Very space-efficient, in other words.

Let's try registering another version of the model -- this time, one that translates from English to French. We'll register this under the version `"2"`; the model server will retrieve the model version that way.

But first, let's extend the model server with multiplexing support.

```python
from ray import serve
from fastapi import FastAPI

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
```

```python
multiplexed_deployment = MultiplexedModelDeployment.bind(model_name="translation_model")
serve.run(multiplexed_deployment, blocking=False)
```

Now let's actually register the new model version.

```python
import pandas as pd

with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        "translation_model",
        registered_model_name="translation_model",
        python_model=MyTranslationModel(),
        pip_requirements=["transformers"],
        input_example=pd.DataFrame({
            "prompt": [
                "Hello my name is Jon.",
            ],
        }),
        model_config={
            "hfhub_name": "google-t5/t5-base",
            "lang_from": "en",
            "lang_to": "fr",
        },
    )

en_to_fr_version: str = str(model_info.registered_model_version)
```

Now that that's registered, we can query for it via the model server like so...

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/serve/",
    params={"input_string": "The weather is lovely today"},
    headers={"serve_multiplexed_model_id": en_to_fr_version},
)

print(response.json())
```

    [{'translation_text': "Le temps est beau aujourd'hui"}]

Note how we were able to immediately access the model version **without redeploying the model server**. Ray Serve's multiplexing capabilities allow it to dynamically fetch the model weights in a just-in-time fashion; if I never requested version 2, it never gets loaded. This helps conserve compute resources for the models that **do** get queried. What's even more useful is that, if the number of models loaded up exceeds the configured maximum (`max_num_models_per_replica`), the [least-recently used model version will get evicted](https://docs.ray.io/en/latest/serve/model-multiplexing.html#why-model-multiplexing).

Given that we set `max_num_models_per_replica=2` above, the "default" English-to-German version of the model should still be loaded up and readily available to serve requests without any cold-start time. Let's confirm that now:

```python
print(
    requests.post(
        "http://127.0.0.1:8000/serve/",
        params={"input_string": "The weather is lovely today"},
        headers={"serve_multiplexed_model_id": en_to_de_version},
    ).json()
)
```

    [{'translation_text': 'Das Wetter ist heute nett.'}]

## Auto-Signature

This is all well and good. However, notice that the following friction point still exists: when defining the server, we need to define a whole new signature for the API itself. At best, this is just some code duplication of the model signature itself (which is registered in MLflow). At worst, this can result in inconsistent APIs across all models that your team or organization owns, which can cause confusion and frustration in your downstream dependencies.

In this particular case, it means that `MultiplexedModelDeployment` is secretly actually **tightly coupled** to the use-case for `translation_model`. What if we wanted to deploy another set of models that don't have to do with language translation? The defined `/serve` API, which returns a JSON object that looks like `{"translated_text": "foo"}`, would no longer make sense.

To address this issue, **what if the API signature for `MultiplexedModelDeployment` could automatically mirror the signature of the underlying models it's serving**?

Thankfully, with MLflow Model Registry metadata and some Python dynamic-class-creation shenanigans, this is entirely possible.

Let's set things up so that the model server signature is inferred from the registered model itself. Since different versions of an MLflow can have different signatures, we'll use the "default version" to "pin" the signature; any attempt to multiplex an incompatible-signature model version we will have throw an error.

Since Ray Serve binds the request and response signatures at class-definition time, we will use a Python metaclass to set this as a function of the specified model name and default model version.

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
    inputs: mlflow.types.schema.Schema = model_signature.inputs
    outputs: mlflow.types.schema.Schema = model_signature.outputs

    return (schema_to_pydantic(inputs, name="InputModel"), schema_to_pydantic(outputs, name="OutputModel"))
```

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

        MODEL_NAME: str = model_name
        DEFAULT_VERSION: str = default_version

        @serve.multiplexed(max_num_models_per_replica=2)
        async def get_model(self, model_version: str):
            model = mlflow.pyfunc.load_model(f"models:/{self.MODEL_NAME}/{model_version}")

            if model.metadata.get_model_info().signature != model_info.signature:
                raise ValueError(f"Requested version {model_version} has signature incompatible with that of default version {self.DEFAULT_VERSION}")
            return model

        # TODO: Extend this to support batching (lists of inputs and outputs)
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

deployment = deployment_from_model_name("translation_model", default_version=en_to_fr_version)

serve.run(deployment.bind(), blocking=False)
```

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8000/serve/",
    json={"prompt": "The weather is lovely today"},
)

assert resp.ok
assert resp.status_code == 200

print(resp.json())
```

    [{'translation_text': "Le temps est beau aujourd'hui"}]

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8000/serve/",
    json={"prompt": "The weather is lovely today"},
    headers={"serve_multiplexed_model_id": str(en_to_fr_version)},
)

assert resp.ok
assert resp.status_code == 200

print(resp.json())
```

    [{'translation_text': "Le temps est beau aujourd'hui"}]

Let's now confirm that the signature-check provision we put in place actually works. For this, let's register this same model with a **slightly** different signature. This should be enough to trigger the failsafe.

(Remember when we made the input label configurable at the start of this exercise? This is where that finally comes into play. 😎)

```python
import pandas as pd

with mlflow.start_run():
    incompatible_version = str(mlflow.pyfunc.log_model(
        "translation_model",
        registered_model_name="translation_model",
        python_model=MyTranslationModel(),
        pip_requirements=["transformers"],
        input_example=pd.DataFrame({
            "text_to_translate": [
                "Hello my name is Jon.",
            ],
        }),
        model_config={
            "input_label": "text_to_translate",
            "hfhub_name": "google-t5/t5-base",
            "lang_from": "en",
            "lang_to": "de",
        },
    ).registered_model_version)
```

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8000/serve/",
    json={"prompt