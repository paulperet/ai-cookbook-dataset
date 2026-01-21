# MLflow Cookbook: Deploying Phi-3 as an MLflow Model

## Introduction

[MLflow](https://mlflow.org/) is an open-source platform designed to manage the end-to-end machine learning lifecycle. It provides tools for experiment tracking, packaging code into reproducible runs, and deploying models to various serving environments. This cookbook demonstrates two practical approaches to wrapping Microsoft's Phi-3 small language model (SLM) as an MLflow model, enabling you to deploy it locally or in cloud environments like Azure Machine Learning.

## Prerequisites

Before starting, ensure you have the necessary Python packages installed. The specific requirements will vary depending on which approach you choose.

## Approach 1: Transformer Pipeline Wrapper

This method leverages MLflow's experimental transformers flavor, providing the simplest path to wrap a Hugging Face model.

### Step 1: Install Required Packages

First, install the core dependencies:

```bash
pip install mlflow transformers
```

### Step 2: Import Libraries and Initialize the Pipeline

Begin by importing the necessary modules and creating a transformer pipeline for the Phi-3 model.

```python
import mlflow
import transformers

# Initialize the Phi-3 text generation pipeline
pipeline = transformers.pipeline(
    task="text-generation",
    model="microsoft/Phi-3-mini-4k-instruct"
)
```

### Step 3: Log the Model to MLflow

Save the pipeline as an MLflow model. Specify the task as `llm/v1/chat` to generate an OpenAI-compatible API wrapper.

```python
# Define model configuration (optional, adjust parameters as needed)
model_config = {
    "max_length": 300,
    "temperature": 0.2,
}

# Log the model to MLflow
model_info = mlflow.transformers.log_model(
    transformers_model=pipeline,
    artifact_path="phi3-mlflow-model",
    model_config=model_config,
    task="llm/v1/chat"
)

print(f"Model logged to: {model_info.model_uri}")
```

## Approach 2: Custom Python Wrapper

Use this approach when you need to work with models in ONNX format or require more control over the inference logic. This example uses the ONNX Runtime generate() API.

### Step 1: Install Specialized Dependencies

Install packages for ONNX Runtime and MLflow's Python function flavor.

```bash
pip install mlflow torch onnxruntime-genai numpy
```

### Step 2: Create a Custom Python Model Class

Define a class that inherits from `mlflow.pyfunc.PythonModel`. This class will handle model loading and prediction.

```python
import mlflow
from mlflow.models import infer_signature
import onnxruntime_genai as og

class Phi3Model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Load the ONNX model, generator parameters, and tokenizer from the context artifacts.
        """
        # Path to the ONNX model (stored as an artifact)
        model_path = context.artifacts["phi3-mini-onnx"]
        
        # Define generation parameters
        model_options = {
            "max_length": 300,
            "temperature": 0.2,
        }
        
        # Initialize the ONNX model
        self.phi3_model = og.Model(model_path)
        self.params = og.GeneratorParams(self.phi3_model)
        self.params.set_search_options(**model_options)
        
        # Initialize the tokenizer
        self.tokenizer = og.Tokenizer(self.phi3_model)
    
    def predict(self, context, model_input):
        """
        Generate a response for the given prompt.
        """
        # Extract the prompt from the input DataFrame
        prompt = model_input["prompt"][0]
        
        # Encode the prompt and generate tokens
        self.params.input_ids = self.tokenizer.encode(prompt)
        response = self.phi3_model.generate(self.params)
        
        # Decode and return the generated text
        return self.tokenizer.decode(response[0][len(self.params.input_ids):])
```

### Step 3: Log the Custom Model

Log your custom model to MLflow, specifying the artifacts (the ONNX model files) and dependencies.

```python
# Define an example input for signature inference
input_example = {"prompt": ["Tell me a joke."]}

# Log the custom Python model
model_info = mlflow.pyfunc.log_model(
    artifact_path="phi3-custom-model",
    python_model=Phi3Model(),
    artifacts={
        # This key must match the one used in load_context()
        "phi3-mini-onnx": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
    },
    input_example=input_example,
    signature=infer_signature(input_example, ["Run"]),
    extra_pip_requirements=["torch", "onnxruntime_genai", "numpy"],
)

print(f"Custom model logged to: {model_info.model_uri}")
```

## Understanding Model Signatures and Usage

The two approaches produce MLflow models with different input/output signatures, affecting how you interact with them.

### Transformer Pipeline Signature (OpenAI-Compatible)

By setting `task="llm/v1/chat"`, the model wrapper conforms to the OpenAI Chat Completion API format.

**Input Format:**
```python
messages = [{"role": "user", "content": "What is the capital of Spain?"}]
```

**Output Processing:**
The response is a structured dictionary. Extract the generated text like this:
```python
response[0]['choices'][0]['message']['content']
```

**Example Output:**
```json
"The capital of Spain is Madrid. It is the largest city in Spain and serves as the political, economic, and cultural center of the country..."
```

### Custom Python Wrapper Signature

The signature is inferred from the `input_example`, resulting in a simpler, direct format.

**Input Format:**
```python
{"prompt": "<|system|>You are a stand-up comedian.<|end|><|user|>Tell me a joke about atoms.<|end|><|assistant|>"}
```

**Output Format:**
The model returns a plain string.

**Example Output:**
```
"Alright, here's a little atom-related joke for you! Why don't electrons ever play hide and seek with protons? Because good luck finding them when they're always 'sharing' their electrons!"
```

## Next Steps

Your Phi-3 model is now packaged as an MLflow model. You can:

1. **Load and run it locally** using `mlflow.pyfunc.load_model()`.
2. **Register it** in the MLflow Model Registry for versioning and staging.
3. **Deploy it as a REST API** using `mlflow models serve`.
4. **Deploy to cloud platforms** like Azure Machine Learning for scalable inference.

Choose the wrapper approach that best fits your model format (standard Hugging Face vs. ONNX) and desired API signature (OpenAI-compatible vs. custom).