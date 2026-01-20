

# Gemini API: JSON Mode Quickstart

The Gemini API can be used to generate a JSON output if you set the schema that you would like to use.

Two methods are available. You can either set the desired output in the prompt or supply a schema to the model separately.

### Install dependencies


```
%pip install -U -q "google-genai>=1.0.0"
```

### Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
from google import genai

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Set your constrained output in the prompt

For this first example just describe the schema you want back in the prompt:



```
prompt = """
  List a few popular cookie recipes using this JSON schema:

  Recipe = {'recipe_name': str}
  Return: list[Recipe]
"""
```

Now select the model you want to use in this guide, either by selecting one in the list or writing it down. Keep in mind that some models, like the 2.5 ones are thinking models and thus take slightly more time to respond (cf. [thinking notebook](./Get_started_thinking.ipynb) for more details and in particular learn how to switch the thiking off).

Then activate JSON mode by specifying `respose_mime_type` in the `config` parameter:


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}

raw_response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config={
        'response_mime_type': 'application/json'
    },
)
```

Parse the string to JSON:


```
import json

response = json.loads(raw_response.text)
print(response)
```

    [{'recipe_name': 'Chocolate Chip Cookies'}, {'recipe_name': 'Oatmeal Raisin Cookies'}, {'recipe_name': 'Peanut Butter Cookies'}, {'recipe_name': 'Sugar Cookies'}, {'recipe_name': 'Snickerdoodles'}]


For readability serialize and print it:


```
print(json.dumps(response, indent=4))
```

    [
        {
            "recipe_name": "Chocolate Chip Cookies"
        },
        {
            "recipe_name": "Oatmeal Raisin Cookies"
        },
        {
            "recipe_name": "Peanut Butter Cookies"
        },
        {
            "recipe_name": "Sugar Cookies"
        },
        {
            "recipe_name": "Snickerdoodles"
        }
    ]


## Supply the schema to the model directly

The newest models (1.5 and beyond) allow you to pass a schema object (or a python type equivalent) directly and the output will strictly follow that schema.

Following the same example as the previous section, here's that recipe type:


```
import typing_extensions as typing

class Recipe(typing.TypedDict):
    recipe_name: str
    recipe_description: str
    recipe_ingredients: list[str]
```

For this example you want a list of `Recipe` objects, so pass `list[Recipe]` to the `response_schema` field of the `config`.


```
result = client.models.generate_content(
    model=MODEL_ID,
    contents="List a few imaginative cookie recipes along with a one-sentence description as if you were a gourmet restaurant and their main ingredients",
    config={
        'response_mime_type': 'application/json',
        'response_schema': list[Recipe],
    },
)
```


```
print(json.dumps(json.loads(result.text), indent=4))
```

    [
        {
            "recipe_name": "Lavender & White Chocolate Cloud Kisses",
            "recipe_description": "Delicate lavender-infused meringue cookies, airily light and meltingly tender, are kissed with swirls of creamy white chocolate.",
            "recipe_ingredients": [
                "Egg whites",
                "Sugar",
                "Dried lavender",
                "White chocolate"
            ]
        },
        {
            "recipe_name": "Smoked Paprika & Dark Chocolate Chili Sables",
            "recipe_description": "A sophisticated blend of smoky paprika, intense dark chocolate, and a whisper of chili creates a captivating sweet and savory experience in a crisp sable cookie.",
            "recipe_ingredients": [
                "All-purpose flour",
                "Unsalted butter",
                "Dark chocolate",
                "Smoked paprika",
                "Chili powder"
            ]
        },
        {
            "recipe_name": "Matcha & Black Sesame Shortbread Petals",
            "recipe_description": "Elegantly sculpted shortbread petals, vibrant with ceremonial matcha, offer a sublime earthy sweetness balanced by the nutty depth of toasted black sesame.",
            "recipe_ingredients": [
                "All-purpose flour",
                "Unsalted butter",
                "Sugar",
                "Matcha powder",
                "Black sesame seeds"
            ]
        }
    ]


It is the recommended method if you're using a compatible model.

## Next Steps
### Useful API references:

Check the [structured ouput](https://ai.google.dev/gemini-api/docs/structured-output) documentation or the [`GenerationConfig`](https://ai.google.dev/api/generate-content#generationconfig) API reference for more details

### Related examples

* The constrained output is used in the [Text summarization](../examples/json_capabilities/Text_Summarization.ipynb) example to provide the model a format to summarize a story (genre, characters, etc...)
* The [Object detection](../examples/Object_detection.ipynb) examples are using the JSON constrained output to uniiformize the output of the detection.

### Continue your discovery of the Gemini API

JSON is not the only way to constrain the output of the model, you can also use an [Enum](../quickstarts/Enum.ipynb). [Function calling](../quickstarts/Function_calling.ipynb) and [Code execution](../quickstarts/Code_Execution.ipynb) are other ways to enhance your model by either using your own functions or by letting the model write and run them.