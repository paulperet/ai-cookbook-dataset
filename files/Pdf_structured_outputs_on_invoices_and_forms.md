

# Gemini API: Automated Invoice and Form Data Extraction with Gemini API & Pydantic

This notebook demonstrates how you can convert a PDF file so that it can be read by the Gemini API.

## 1. Set up Environment and create inference Client

The first task is to install the `google-genai` [Python SDK](https://googleapis.github.io/python-genai/) and obtain an API key. If you don”t have a can get one from Google AI Studio: [Get a Gemini API key](https://aistudio.google.com/app/apikey). If you are new to Google Colab checkout the [quickstart](../quickstarts/Authentication.ipynb)).



```
%pip install "google-genai>=1"
```

Once you have the SDK and API key, you can create a client and define the model you are going to use the new Gemini Flash model, which is available via [free tier](https://ai.google.dev/pricing#2_0flash) with 1,500 request per day (at 2025-02-06). 


```
from google import genai
from google.colab import userdata
api_key = userdata.get("GOOGLE_API_KEY") # If you are not using Colab you can set the API key directly

# Create a client
client = genai.Client(api_key=api_key)

# Define the model you are going to use
model_id =  "gemini-2.5-flash" # or "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro","gemini-3-pro-preview"
```

*Note: If you want to use Vertex AI see [here](https://googleapis.github.io/python-genai/#create-a-client) how to create your client*

## 2. Work with PDFs and other files

Gemini models are able to process [images and videos](https://ai.google.dev/gemini-api/docs/vision?lang=python#image-input), which can used with base64 strings or using the `files`api. After uploading the file you can include the file uri in the call directly. The Python API includes a [upload](https://googleapis.github.io/python-genai/#upload) and [delete](https://googleapis.github.io/python-genai/#delete) method. 

For this example you have 2 PDFs samples, one basic invoice and on form with and written values. 



```
!wget -q -O handwriting_form.pdf https://storage.googleapis.com/generativeai-downloads/data/pdf_structured_outputs/handwriting_form.pdf
!wget -q -O invoice.pdf https://storage.googleapis.com/generativeai-downloads/data/pdf_structured_outputs/invoice.pdf
```

You can now upload the files using our client with the `upload` method. Let's try this for one of the files.



```
invoice_pdf = client.files.upload(file="invoice.pdf", config={'display_name': 'invoice'})
```

_Note: The File API lets you store up to 20 GB of files per project, with a per-file maximum size of 2 GB. Files are stored for 48 hours. They can be accessed in that period with your API key, but they cannot be downloaded. File uploads are available at no cost._

After a file is uploaded you can check to how many tokens it got converted. This not only help us understand the context you are working with it also helps to keep track of the cost. 


```
file_size = client.models.count_tokens(model=model_id,contents=invoice_pdf)
print(f'File: {invoice_pdf.display_name} equals to {file_size.total_tokens} tokens')
# File: invoice equals to 821 tokens
```

    File: invoice equals to 821 tokens

## 3. Structured outputs with Gemini 2.0 and Pydantic

Structured Outputs is a feature that ensures Gemini always generate responses that adhere to a predefined format, such as JSON Schema. This means you have more control over the output and how to integrate it into our application as it is guaranteed to return a valid JSON object with the schema you define. 

Gemini 2.0 currenlty supports 3 dfferent types of how to define a JSON schemas:
- A single python type, as you would use in a [typing annotation](https://docs.python.org/3/library/typing.html).
- A Pydantic [BaseModel](https://docs.pydantic.dev/latest/concepts/models/)
- A dict equivalent of [genai.types.Schema](https://googleapis.github.io/python-genai/genai.html#genai.types.Schema) / [Pydantic BaseModel](https://docs.pydantic.dev/latest/concepts/models/)

Lets look at quick text-based example.


```
from pydantic import BaseModel, Field

# Define a Pydantic model
# Use the Field class to add a description and default value to provide more context to the model
class Topic(BaseModel):
    name: str = Field(description="The name of the topic")

class Person(BaseModel):
    first_name: str = Field(description="The first name of the person")
    last_name: str = Field(description="The last name of the person")
    age: int = Field(description="The age of the person, if not provided please return 0")
    work_topics: list[Topic] = Field(description="The fields of interest of the person, if not provided please return an empty list")


# Define the prompt
prompt = "Philipp Schmid is a Senior AI Developer Relations Engineer at Google DeepMind working on Gemini, Gemma with the mission to help every developer to build and benefit from AI in a responsible way.  "

# Generate a response using the Person model
response = client.models.generate_content(model=model_id, contents=prompt, config={'response_mime_type': 'application/json', 'response_schema': Person})

# print the response as a json string
print(response.text)

# sdk automatically converts the response to the pydantic model
philipp: Person = response.parsed

# access an attribute of the json response
print(f"First name is {philipp.first_name}")
```

    {
      "age": 0,
      "first_name": "Philipp",
      "last_name": "Schmid",
      "work_topics": [
        {
          "name": "AI"
        },
        {
          "name": "Gemini"
        },
        {
          "name": "Gemma"
        }
      ]
    }
    First name is Philipp

## 4. Extract Structured data from PDFs using Gemini 2.0

Now, let's combine the File API and structured output to extract information from our PDFs. You can create a simple method that accepts a local file path and a pydantic model and return the structured data for us. The method will:

1. Upload the file to the File API
2. Generate a structured response using the Gemini API
3. Convert the response to the pydantic model and return it



```
def extract_structured_data(file_path: str, model: BaseModel):
    # Upload the file to the File API
    file = client.files.upload(file=file_path, config={'display_name': file_path.split('/')[-1].split('.')[0]})
    # Generate a structured response using the Gemini API
    prompt = f"Extract the structured data from the following PDF file"
    response = client.models.generate_content(model=model_id, contents=[prompt, file], config={'response_mime_type': 'application/json', 'response_schema': model})
    # Convert the response to the pydantic model and return it
    return response.parsed
```

In our Example every PDF is a different to each other. So you want to define unique Pydantic models for each PDF to show the performance of the Gemini 2.0. If you have very similar PDFs and want to extract the same information you can use the same model for all of them. 

- `Invoice.pdf` : Extract the invoice number, date and all list items with description, quantity and gross worth and the total gross worth
- `handwriting_form.pdf` : Extract the form number, plan start date and the plan liabilities beginning of the year and end of the year

_Note: Using Pydantic features you can add more context to the model to make it more accurate as well as some validation to the data. Adding a comprehensive description can significantly improve the performance of the model. Libraries like [instructor](https://python.useinstructor.com/) added automatic retries based on validation errors, which can be a great help, but come at the cost of additional requests._

### Invoice.pdf


```
from pydantic import BaseModel, Field

class Item(BaseModel):
    description: str = Field(description="The description of the item")
    quantity: float = Field(description="The Qty of the item")
    gross_worth: float = Field(description="The gross worth of the item")

class Invoice(BaseModel):
    """Extract the invoice number, date and all list items with description, quantity and gross worth and the total gross worth."""
    invoice_number: str = Field(description="The invoice number e.g. 1234567890")
    date: str = Field(description="The date of the invoice e.g. 2024-01-01")
    items: list[Item] = Field(description="The list of items with description, quantity and gross worth")
    total_gross_worth: float = Field(description="The total gross worth of the invoice")


result = extract_structured_data("invoice.pdf", Invoice)
print(type(result))
print(f"Extracted Invoice: {result.invoice_number} on {result.date} with total gross worth {result.total_gross_worth}")
for item in result.items:
    print(f"Item: {item.description} with quantity {item.quantity} and gross worth {item.gross_worth}")
```

    <class '__main__.Invoice'>
    Extracted Invoice: 27301261 on 10/09/2012 with total gross worth 544.46
    Item: Lilly Pulitzer dress Size 2 with quantity 5.0 and gross worth 247.5
    Item: New ERIN Erin Fertherston Straight Dress White Sequence Lining Sleeveless SZ 10 with quantity 1.0 and gross worth 65.99
    Item: Sequence dress Size Small with quantity 3.0 and gross worth 115.5
    Item: fire los angeles dress Medium with quantity 3.0 and gross worth 21.45
    Item: Eileen Fisher Women's Long Sleeve Fleece Lined Front Pockets Dress XS Gray with quantity 3.0 and gross worth 52.77
    Item: Lularoe Nicole Dress Size Small Light Solid Grey/ White Ringer Tee Trim with quantity 2.0 and gross worth 8.25
    Item: J.Crew Collection Black & White sweater Dress sz S with quantity 1.0 and gross worth 33.0

Fantastic! The model did a great job extracting the information from the invoice. 

### handwriting_form.pdf


```
class Form(BaseModel):
    """Extract the form number, fiscal start date, fiscal end date, and the plan liabilities beginning of the year and end of the year."""
    form_number: str = Field(description="The Form Number")
    start_date: str = Field(description="Effective Date")
    beginning_of_year: float = Field(description="The plan liabilities beginning of the year")
    end_of_year: float = Field(description="The plan liabilities end of the year")

result = extract_structured_data("handwriting_form.pdf", Form)

print(f'Extracted Form Number: {result.form_number} with start date {result.start_date}. \nPlan liabilities beginning of the year {result.beginning_of_year} and end of the year {result.end_of_year}')
```

    Extracted Form Number: CA530082 with start date 02/05/2022. 
    Plan liabilities beginning of the year 40000.0 and end of the year 55000.0

## Learning more

If you want to learn more about the File API, Structured Outputs and how to use it to process images, audio, and video files, check out the following resources:

* Learn more about the [File API](../quickstarts/File_API.ipynb) with the quickstart.
* Learn more about prompting with [media files](https://ai.google.dev/gemini-api/docs/file-prompting-strategies) in the docs, including the supported formats and maximum length.
* Learn more about [Structured Outputs](https://ai.google.dev/gemini-api/docs/structured-output?lang=python) in the docs.

## Property ordering with Gemini 2.0

**Important:** Gemini 2.0 models require explicit ordering of keys in structured output schemas. When working with Gemini 2.0, you must define the desired property ordering as a list within the `propertyOrdering` field as part of your schema configuration.

This ensures consistent and predictable ordering of properties in the JSON response, which is particularly important when the output property order matters for downstream processing.


```
# Example: Specifying property ordering for Gemini 2.0
from google import genai
from google.genai import types

# Define a schema with explicit property ordering
invoice_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "invoice_number": types.Schema(type=types.Type.STRING),
        "date": types.Schema(type=types.Type.STRING),
        "vendor": types.Schema(type=types.Type.STRING),
        "total_amount": types.Schema(type=types.Type.NUMBER),
    },
    # REQUIRED for Gemini 2.0: Specify the exact order of properties, not Gemini 2.5 or newer
    property_ordering=["invoice_number", "date", "vendor", "total_amount"],
)

# Use the schema with generateContent
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Extract invoice details: Invoice #12345 dated 2024-01-15 from Acme Corp for $1,250.00",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=invoice_schema,
    ),
)

print(response.text)
```

    {
      "invoice_number": "12345",
      "date": "2024-01-15",
      "vendor": "Acme Corp",
      "total_amount": 1250.00
    }