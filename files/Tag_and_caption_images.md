

# Gemini API: Using Gemini API to tag and caption images

You will use the Gemini model's vision capabilities and the embedding model to add tags and captions to images of pieces of clothing.

These descriptions can be used alongside embeddings to allow you to search for specific pieces of clothing using natural language, or other images.

## Setup


```
%pip install -U -q "google-genai>=1.0.0"
```


```
from google import genai
from google.genai import types
```

## Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
api_key = userdata.get('GOOGLE_API_KEY')

client = genai.Client(api_key=api_key)
```

## Downloading dataset
First, you need to download a dataset with images. It contains images of various clothing that you can use to test the model.


```
!wget https://storage.googleapis.com/generativeai-downloads/data/clothes-dataset.zip
```

    [Length: 730831 (714K) [application/zip], ..., 2025-04-08 18:09:37 (1.42 MB/s) - ‘clothes-dataset.zip.3’ saved [730831/730831]]


Unzip the data in `clothes-dataset.zip` and place them in a folder in your Colab environment.


```
!unzip -o clothes-dataset.zip
```

    [Archive:  clothes-dataset.zip, inflating: clothes-dataset/6.jpg, ..., inflating: clothes-dataset/3.jpg]



```
from glob import glob
images = glob("/content/clothes-dataset/*")
images.sort(reverse=True)
```

## Generating keywords
You can use the LLM to extract relevant keywords from the images.

Here is a helper function for calling Gemini API with images. Sleep is for ensuring that the quota is not exceeded. Refer to our [princing](https://ai.google.dev/pricing) page for current quotas.


```
from PIL import Image as PILImage
import time

MODEL_ID='gemini-2.0-flash' # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro","gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}

# a helper function for calling

def generate_text_using_image(prompt, image_path, sleep_time=4):
  start = time.perf_counter()
  response = client.models.generate_content(
    model=MODEL_ID,
    contents=[PILImage.open(image_path)],
    config=types.GenerateContentConfig(
        system_instruction=prompt
    ),
)
  end = time.perf_counter()
  duration = end - start
  time.sleep(sleep_time - duration if duration < sleep_time else 0)
  return response.text
```

First, define the list of possible keywords.


```
import numpy as np
keywords = np.concatenate((
    ["flannel", "shorts", "pants", "dress", "T-shirt", "shirt", "suit"],
    ["women", "men", "boys", "girls"],
    ["casual", "sport", "elegant"],
    ["fall", "winter", "spring", "summer"],
    ["red", "violet", "blue", "green", "yellow", "orange", "black", "white"],
    ["polyester", "cotton", "denim", "silk", "leather", "wool", "fur"]
)
)
```

Go ahead and define a prompt that will help define keywords that describe clothing. In the following prompt, few-shot prompting is used to prime the LLM with examples of how these keywords should be generated and which are valid.


```
keyword_prompt = f"""
     You are an expert in clothing that specializes in tagging images of clothes,
     shoes, and accessories.
     Your job is to extract all relevant keywords from
     a photo that will help describe an item.
     You are going to see an image,
     extract only the keywords for the clothing, and try to provide as many
     keywords as possible.

     Allowed keywords: {list(keywords)}

     Extract tags only when it is obvious that it describes the main item in
     the image. Return the keywords as a list of strings:

     example1: ["blue", "shoes", "denim"]
     example2: ["sport", "skirt", "cotton", "blue", "red"]
"""
```


```
def generate_keywords(image_path):
  return generate_text_using_image(keyword_prompt, image_path)
```

Generate keywords for each of the images.


```
from IPython.display import Image, display
for image_path in images[:4]:
  response_text = generate_keywords(image_path)
  display(Image(image_path))
  print(response_text)
```

    ["shorts", "denim", "blue"]
    ["suit", "men", "blue", "elegant"]
    ["suit", "blue", "black", "men", "elegant"]
    Here are the extracted keywords:
    ["T-shirt", "cotton", "casual", "women", "spring", "summer", "red"]


### Keyword correction and deduplication

Unfortunately, despite providing a list of possible keywords, the model, at least in theory, can return an invalid keyword. It may be a duplicate e.g. "denim" for "jeans", or be completely unrelated to any keyword from the list.

To address these issues, you can use embeddings to map the keywords to predefined ones and remove unrelated ones.


```
import pandas as pd

EMBEDDINGS_MODEL_ID = "embedding-001" # @param ["embedding-001", "text-embedding-004","gemini-embedding-exp-03-07"] {"allow-input":true, isTemplate: true}

def embed(text):
    embedding = client.models.embed_content(
        model=EMBEDDINGS_MODEL_ID,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="semantic_similarity"
        )
    )
    return np.array(embedding.embeddings[0].values)

keywords_df = pd.DataFrame({'Keywords': keywords})
keywords_df["Embeddings"] = keywords_df['Keywords'].apply(embed)
```


```
keywords_df.head()
```

For demonstration purposes, define a function that assesses the similarity between two embedding vectors. In this case, you will use cosine similarity, but other measures such as dot product work too.


```
def cosine_similarity(array_1, array_2):
  return np.dot(array_1,array_2)/(np.linalg.norm(array_1)*np.linalg.norm(array_2))
```

Next, define a function that allows you to replace a keyword with the most similar word in the keyword dataframe that you have previously created.

Note that the threshold is decided arbitrarily, it may require tweaking depending on use case and dataset.


```
def replace_word_with_most_similar(keyword, keywords_df, threshold=0.7):
  # No need for embeddings if the keyword is valid.
  if keyword in keywords_df["Keywords"]:
    return keyword
  embedding = embed(keyword)
  similarities = keywords_df['Embeddings'].apply(lambda row_embedding: cosine_similarity(embedding, row_embedding))
  most_similar_keyword_index = similarities.idxmax()
  if similarities[most_similar_keyword_index] < threshold:
    return None
  return keywords_df.loc[most_similar_keyword_index, "Keywords"]
```

Here is an example of how these keywords can be mapped to a keyword with the closest meaning.


```
for word in ["purple", "tank top", "everyday"]:
  print(word, "->", replace_word_with_most_similar(word, keywords_df))
```

    purple -> violet
    tank top -> T-shirt
    everyday -> casual


You can now either leave words that do not fit our predefined categories or delete them. In this scenario, all words without a suitable replacement will be omitted.


```
def map_generated_keywords_to_predefined(generated_keywords, keywords_df=keywords_df):
  output_keywords = set()
  for keyword in generated_keywords:
    if mapped_keyword := replace_word_with_most_similar(keyword, keywords_df):
      output_keywords.add(mapped_keyword)
  return output_keywords

print(map_generated_keywords_to_predefined(["white", "business", "sport", "women", "polyester"]))
print(map_generated_keywords_to_predefined(["blue", "jeans", "women", "denim", "casual"]))
```

    {'polyester', 'women', 'white', 'sport'}
    {'women', 'blue', 'casual', 'denim'}


### Generating captions


```
caption_prompt ="""
     You are an expert in clothing that specializes in describing images of
     clothes, shoes and accessories.
     Your job is to extract information from a photo that will help describe an item.
     You are going to see an image, focus only on the piece of clothing,
     ignore suroundings.
     Be specific, but stay concise, the description should only be one sentence long.
     Most important aspects are color, type of clothing, material, style
     and who is it meant for.
     If you are not sure about a part of the image, ignore it.
"""
def generate_caption(image_path):
  return generate_text_using_image(caption_prompt, image_path)

for image_path in images[8:]:
  response_text = generate_caption(image_path)
  display(Image(image_path))
  print(response_text)
```

    This is a red, short-sleeved, knee-length women's dress with a colorful floral pattern.
    This is a khaki button-up shirt with two chest pockets and long sleeves, designed for men.


## Searching for specific clothes

### Preparing out dataset
First, you need to generate caption and keywords for every image. Then, you will use embeddings, which will be used later to compare the images in the search dataset with other descriptions and images.

Also, the `ast.literal_eval()` helper function allows you to evaluate an object passed in and get the literal object. For instance, if you passed in a string `"[1, 2, 3]"`, the `ast.literal_eval()` function would return it as a list `[1, 2, 3]`. For more information on this function, here is the [documentation](https://docs.python.org/3/library/ast.html).


```
import ast

def generate_keyword_and_caption(image_path):
  keywords = generate_keywords(image_path)
  try:
    keywords = ast.literal_eval(keywords)
    keywords = map_generated_keywords_to_predefined(keywords)
  except SyntaxError:
    pass
  caption = generate_caption(image_path)
  return {
      "image_path": image_path,
      "keywords": keywords,
      "caption": caption
  }
```

You will use only the first 8 images, so the rest can be used for testing.


```
described_df = pd.DataFrame([generate_keyword_and_caption(image_path) for image_path in images[:8]])
```


```
def embed_row(row):
  text = ", ".join(row["keywords"]) + ".\n" + row["caption"]
  return embed(text)
```


```
described_df["embeddings"] = described_df.apply(lambda x: embed_row(x), axis=1)
```


```
described_df
```

### Finding clothes using natural language


```
def find_image_from_text(text):
  text_embedding = embed(text)
  similarities = described_df['embeddings'].apply(lambda row_embedding: cosine_similarity(text_embedding, row_embedding))
  most_fitting_image_index = similarities.idxmax()
  return described_df["image_path"][most_fitting_image_index]
```


```
display(Image(find_image_from_text("A suit for a wedding.")))
```



```
display(Image(find_image_from_text("A colorful dress.")))
```

### Finding similar clothes using images


```
def find_image_from_image(image_path):
  text_embedding = embed_row(generate_keyword_and_caption(image_path))
  similarities = described_df['embeddings'].apply(lambda row_embedding: cosine_similarity(text_embedding, row_embedding))
  most_fitting_image_index = similarities.idxmax()
  return described_df["image_path"][most_fitting_image_index]
```


```
image_path = images[8]
display(Image(image_path))
display(Image(find_image_from_image(image_path)))
```



```
image_path = images[9]
display(Image(image_path))
display(Image(find_image_from_image(image_path)))
```

# Summary
You have used Gemini API's Python SDK to tag and caption images of clothing. Using embedding models, you were able to search a database of images for clothing matching our description, or similar to the provided clothing item.