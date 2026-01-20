# Gemini API: Models with REST

This notebook demonstrates how to list the models that are available for you to use in the Gemini API, and how to find details about a model in `curl`.

You can run this in Google Colab, or you can copy/paste the curl commands into your terminal.

To run this notebook, your API key must be stored it in a Colab Secret named `GEMINI_API_KEY`. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) to learn more.

```
import os
from google.colab import userdata
```

```
os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
```

## Model info

### List models

If you `GET` the models directory, it uses the `list` method to list all of the models available through the API, including both the Gemini models.

```bash
%%bash

curl https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY
```

### Get model

If you `GET` a model's URL, the API uses the `get` method to return information about that model such as version, display name, input token limit, etc.

```bash
%%bash

curl https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview?key=$GEMINI_API_KEY
```

## Learning more

To learn how use a model for prompting, see the [Prompting](https://github.com/google-gemini/cookbook/blob/main/quickstarts/rest/Prompting_REST.ipynb) quickstart.

To learn how use a model for embedding, see the [Embedding](https://github.com/google-gemini/cookbook/blob/main/quickstarts/rest/Embeddings_REST.ipynb) quickstart.

For more information on models, visit the [Gemini models](https://ai.google.dev/models/gemini) documentation.