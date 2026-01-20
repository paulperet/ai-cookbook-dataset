##### Copyright 2024 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: LLM Observability with MLflow

<a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/mlflow/MLflow_Observability.ipynb"></a>


## Overview

[MLflow](https://mlflow.org/) is an open-source platform to assist machine learning practitioners and teams in handling the complexities of the machine learning process.

It provides [MLflow Tracing](https://mlflow.org/docs/latest/tracing/) that enhances LLM observability in your Generative AI applications by capturing detailed information about the execution of your application’s services. Tracing provides a way to record the inputs, outputs, and metadata associated with each intermediate step of a request, enabling you to easily pinpoint the source of bugs and unexpected behaviors.

MLflow provides a built-in integration with Google Gen AI SDK that enables you to instrument your Gemini calls easily. This cookbook describes the basic usage of the MLflow tracing integration with the `google-genai` package.

<!-- Community Contributor Badge -->

This notebook was contributed by Tomu Hirata.

Have a cool Gemini example? Feel free to share it too!

## Installation
Install Google Gen AI SDK (`google-genai`) and MLflow (`mlflow`). See [troubleshooting for MLFlow installation](https://mlflow.org/docs/latest/quickstart_drilldown/#quickstart_drilldown_install) for other install options.


```
# Integration with google-genai is supported mlflow >= 2.20.3
%pip install -q google-genai "mlflow>=2.20.3"
```

[First Entry, ..., Last Entry]

## Create Gemini Client with your API key

Let's create an API client and pass your API key. If you do not have API ket yet, visit [AI Studio](https://aistudio.google.com/app/apikey) to create one.


```
import google.genai as genai
from google.colab import userdata

client = genai.Client(api_key=userdata.get('GOOGLE_API_KEY'))
```

## Tracking Server
There are several options to run MLflow tracking server: local tracking server, Databricks Free Trial, and production Databricks. See [our documentation](https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html) for the comparison. In this example, you use [Databricks Free Trial](https://mlflow.org/docs/latest/getting-started/databricks-trial/) which allows easy connection from Colab notebooks and enable you to use managed MLflow for free. Follow the steps below to create an account and generate a Personal Access Token (PAT) to connect to your workspace.

* Go to the [Databricks Trial Signup Page](http://signup.databricks.com/) and create your account
* Follow the steps in [this guide](https://docs.databricks.com/aws/en/dev-tools/auth/pat) to create a PAT for your Databricks workspace user
* You need following information to connect to your workspace from your notebook
  * Databricks Host: Use "https://\<your workspace host\>.cloud.databricks.com/
  * Token: Your personal access token for your Databricks Workspace. 


```
WORKSPACE_EMAIL = userdata.get("WORKSPACE_EMAIL") # your email
WORKSPACE_PAT = userdata.get("WORKSPACE_PAT") # your databricks host https://<your workspace host>.cloud.databricks.com/
WORKSPACE_URL = userdata.get("WORKSPACE_URL") # your PAT
```

Then, set the tracking server uri and experiment name.


```
import os
import mlflow
mlflow.set_tracking_uri('databricks')
mlflow.set_registry_uri('databricks-uc')
os.environ['DATABRICKS_HOST'] = WORKSPACE_URL
os.environ['DATABRICKS_TOKEN'] = WORKSPACE_PAT
mlflow.login()
```


```
# Create Experiment
mlflow.create_experiment(
    f"/Users/{WORKSPACE_EMAIL}/gemini",
    artifact_location="dbfs:/Volumes/workspace/default/gemini",
)
mlflow.set_experiment(f"/Users/{WORKSPACE_EMAIL}/gemini")
```

## Enable AutoLogging
MLflow Tracing provides automatic tracing capability for Google Gemini. By enabling auto tracing for Gemini by calling the `mlflow.gemini.autolog()` function, MLflow will capture nested traces and log them to the active MLflow Experiment upon invocation of Gemini Python SDK.

MLflow trace automatically captures the following information about Gemini calls:
- Prompts and completion responses
- Latencies
- Model name
- Additional metadata such as temperature, max_tokens, if specified.
- Function calling if returned in the response
- Any exception if raised


```
import mlflow

# Turn on auto tracing for Gemini
mlflow.gemini.autolog()
```

## Call Simple Content Generation
Let's run `client.models.generate_content` to try a simple text generation use case with MLflow tracing. For Jupyter Notebook users, MLflow provides a convenient way to see the generated traces on your notebook. See [this blog](https://mlflow.org/blog/mlflow-tracing-in-jupyter) for more information. For users who use other platforms, visit "http://localhost:5000" to see MLflow UI.


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```


```
result = client.models.generate_content(
    model=MODEL_ID, contents="The opposite of hot is"
)
result.text
```




    'cold\n'



## Multi-Turn Chat Interactions
MLflow tracing captures the structure of your interactions with the Gemini API. Run the following cell to try multi-turn chat and see how MLflow captures the interaction.


```
with mlflow.start_span(name="multi-turn"):
    chat = client.chats.create(model=MODEL_ID)
    response = chat.send_message("In one sentence, explain how a computer works to a young child.")
    print(response.text)
    response = chat.send_message("Okay, how about a more detailed explanation to a high schooler?")
    print(response.text)
```

    A computer follows instructions, like a recipe, to do things with numbers and pictures.
    
    A computer works by executing a sequence of instructions, called a program, written in a language it understands. These instructions manipulate binary data (0s and 1s) representing information, performing calculations, storing and retrieving data from memory and storage devices, and interacting with input and output devices like the keyboard, screen, and network to complete tasks.
    


## Function Call
If your application uses [function calling](https://ai.google.dev/gemini-api/docs/function-calling) of the Gemini API, the function definition, function call and function response are captured by MLflow automatically.


```
def add(a: float, b: float):
    """returns a + b."""
    return a + b


def subtract(a: float, b: float):
    """returns a - b."""
    return a - b


def multiply(a: float, b: float):
    """returns a * b."""
    return a * b


def divide(a: float, b: float):
    """returns a / b."""
    return a / b


operation_tools = [add, subtract, multiply, divide]
```


```
chat = client.chats.create(
    model = MODEL_ID,
    config = {
        "tools": operation_tools,
    }
)
```


```
response = chat.send_message(
    "I have 57 cats, each owns 44 mittens, how many mittens is that in total?"
)
response.text
```




    '2508\n'



## Conclusion

That's all for this cookbook. MLflow tracing provides many features that are not included in this notebook and actively releases new features. Visit [MLflow Gemini tracing integration](https://mlflow.org/docs/latest/tracing/integrations/gemini) for more configurations and [MLflow Tracing Overview](https://mlflow.org/docs/latest/tracing/) for general offerings.