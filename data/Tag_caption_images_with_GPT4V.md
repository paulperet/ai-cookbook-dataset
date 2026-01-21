# Guide: Using GPT-4o mini to Tag & Caption Images for Search

This guide demonstrates how to leverage the multimodal capabilities of OpenAI's GPT-4o mini model to generate descriptive tags and captions for product images. We'll walk through a practical use case: enhancing an e-commerce furniture catalog to enable powerful text and image-based search.

## Prerequisites

Ensure you have the necessary Python packages installed and your OpenAI API key configured.

```bash
pip install openai scikit-learn pandas numpy
```

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

# Initialize the OpenAI client
# Ensure your API key is set in your environment or client configuration
client = OpenAI()
```

## 1. Load the Dataset

We'll use a sample dataset of Amazon furniture items. Each entry contains a product title, image URL, and other metadata.

```python
# Load the dataset
dataset_path = "data/amazon_furniture_dataset.csv"
df = pd.read_csv(dataset_path)
df.head()
```

## 2. Generate Keywords for Images

Our first goal is to tag each product with relevant search keywords. We'll use GPT-4o mini to analyze the product image and title, extracting concise keywords.

### 2.1 Define the Keyword Extraction System Prompt

Create a system prompt that instructs the model on the types of keywords to extract (e.g., item type, material, style, color).

```python
system_prompt = '''
You are an agent specialized in tagging images of furniture items, decorative items, or furnishings with relevant keywords that could be used to search for these items on a marketplace.

You will be provided with an image and the title of the item that is depicted in the image, and your goal is to extract keywords for only the item specified.

Keywords should be concise and in lower case.

Keywords can describe things like:
- Item type e.g. 'sofa bed', 'chair', 'desk', 'plant'
- Item material e.g. 'wood', 'metal', 'fabric'
- Item style e.g. 'scandinavian', 'vintage', 'industrial'
- Item color e.g. 'red', 'blue', 'white'

Only deduce material, style or color keywords when it is obvious that they make the item depicted in the image stand out.

Return keywords in the format of an array of strings, like this:
['desk', 'industrial', 'metal']
'''
```

### 2.2 Create the Keyword Extraction Function

This function sends the image URL and product title to the GPT-4o mini model.

```python
def analyze_image(img_url, title):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url,
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": title
            }
        ],
        max_tokens=300,
        top_p=0.1
    )
    return response.choices[0].message.content
```

### 2.3 Test Keyword Extraction

Let's test the function on a few sample products.

```python
# Select the first 5 examples
examples = df.iloc[:5]

for index, row in examples.iterrows():
    url = row['primary_image']
    result = analyze_image(url, row['title'])
    print(f"Title: {row['title'][:50]}...")
    print(f"Keywords: {result}\n")
```

## 3. Deduplicate Keywords Using Embeddings

To maintain a clean set of tags, we'll use text embeddings to identify and merge synonyms or overly similar keywords.

### 3.1 Create Embedding Helper Function

```python
def get_embedding(value, model="text-embedding-3-large"):
    embeddings = client.embeddings.create(
        model=model,
        input=value,
        encoding_format="float"
    )
    return embeddings.data[0].embedding
```

### 3.2 Build a Reference List of Keywords

Start with a small, curated list of canonical keywords.

```python
# Define a base set of keywords
base_keywords = ['industrial', 'metal', 'wood', 'vintage', 'bed']

# Create a DataFrame and compute embeddings
df_keywords = pd.DataFrame(base_keywords, columns=['keyword'])
df_keywords['embedding'] = df_keywords['keyword'].apply(lambda x: get_embedding(x))
df_keywords.head()
```

### 3.3 Define a Keyword Deduplication Function

This function compares a new keyword against the existing list. If it's too similar (based on a cosine similarity threshold), it returns the existing keyword.

```python
def replace_keyword(keyword, df_keywords, threshold=0.6):
    # Get embedding for the new keyword
    new_embedding = get_embedding(keyword)

    # Calculate similarity to all existing keywords
    df_keywords['similarity'] = df_keywords['embedding'].apply(
        lambda x: cosine_similarity(np.array(x).reshape(1, -1),
                                    np.array(new_embedding).reshape(1, -1))
    )

    # Find the most similar existing keyword
    most_similar = df_keywords.sort_values('similarity', ascending=False).iloc[0]

    # Replace if similarity exceeds the threshold
    if most_similar['similarity'] > threshold:
        print(f"Replacing '{keyword}' with existing keyword: '{most_similar['keyword']}'")
        return most_similar['keyword']

    # Otherwise, add the new keyword to our list
    new_row = {'keyword': keyword, 'embedding': new_embedding}
    df_keywords = pd.concat([df_keywords, pd.DataFrame([new_row])], ignore_index=True)
    return keyword
```

### 3.4 Test the Deduplication

```python
# Example new keywords to process
new_keywords = ['bed frame', 'wooden', 'vintage', 'old school', 'desk', 'table', 'old', 'metal', 'metallic', 'woody']
final_keywords = []

for k in new_keywords:
    final_keywords.append(replace_keyword(k, df_keywords))

final_keywords = set(final_keywords)
print(f"Final deduplicated keywords: {final_keywords}")
```

## 4. Generate Descriptive Captions

Next, we'll create short, engaging captions for each product. This is a two-step process: first, generate a detailed description, then refine it into a concise caption.

### 4.1 Clean the Dataset

We'll work with a subset of relevant columns.

```python
selected_columns = ['title', 'primary_image', 'style', 'material', 'color', 'url']
df = df[selected_columns].copy()
df.head()
```

### 4.2 Create an Image Description Function

Use GPT-4o mini to generate a detailed description of the product from its image and title.

```python
describe_system_prompt = '''
You are a system generating descriptions for furniture items, decorative items, or furnishings on an e-commerce website.
Provided with an image and a title, you will describe the main item that you see in the image, giving details but staying concise.
You can describe unambiguously what the item is and its material, color, and style if clearly identifiable.
If there are multiple items depicted, refer to the title to understand which item you should describe.
'''

def describe_image(img_url, title):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": describe_system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url,
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": title
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content
```

### 4.3 Test the Description Function

```python
# Test on a few examples
examples = df.iloc[:3]

for index, row in examples.iterrows():
    print(f"Title: {row['title'][:50]}... - {row['url']}")
    description = describe_image(row['primary_image'], row['title'])
    print(f"Description: {description}\n---\n")
```

### 4.4 Refine Descriptions into Captions

We'll use a few-shot prompting approach with GPT-4o mini to turn the long description into a one-sentence caption.

```python
caption_system_prompt = '''
Your goal is to generate short, descriptive captions for images of furniture items, decorative items, or furnishings based on an image description.
You will be provided with a description of an item image and you will output a caption that captures the most important information about the item.
Your generated caption should be short (1 sentence), and include the most relevant information about the item.
The most important information could be: the type of the item, the style (if mentioned), the material if especially relevant and any distinctive features.
'''

# Define few-shot examples to guide the model
few_shot_examples = [
    {
        "description": "This is a multi-layer metal shoe rack featuring a free-standing design. It has a clean, white finish that gives it a modern and versatile look, suitable for various home decors. The rack includes several horizontal shelves dedicated to organizing shoes, providing ample space for multiple pairs. Above the shoe storage area, there are 8 double hooks arranged in two rows, offering additional functionality for hanging items such as hats, scarves, or bags. The overall structure is sleek and space-saving, making it an ideal choice for placement in living rooms, bathrooms, hallways, or entryways where efficient use of space is essential.",
        "caption": "White metal free-standing shoe rack"
    },
    {
        "description": "The image shows a set of two dining chairs in black. These chairs are upholstered in a leather-like material, giving them a sleek and sophisticated appearance. The design features straight lines with a slight curve at the top of the high backrest, which adds a touch of elegance. The chairs have a simple, vertical stitching detail on the backrest, providing a subtle decorative element. The legs are also black, creating a uniform look that would complement a contemporary dining room setting. The chairs appear to be designed for comfort and style, suitable for both casual and formal dining environments.",
        "caption": "Set of 2 modern black leather dining chairs"
    },
    {
        "description": "This is a square plant repotting mat designed for indoor gardening tasks such as transplanting and changing soil for plants. It measures 26.8 inches by 26.8 inches and is made from a waterproof material, which appears to be a durable, easy-to-clean fabric in a vibrant green color. The edges of the mat are raised with integrated corner loops, likely to keep soil and water contained during gardening activities. The mat is foldable, enhancing its portability, and can be used as a protective surface for various gardening projects, including working with succulents. It's a practical accessory for garden enthusiasts and makes for a thoughtful gift for those who enjoy indoor plant care.",
        "caption": "Waterproof square plant repotting mat"
    }
]

# Format the examples for the API
formatted_examples = []
for ex in few_shot_examples:
    formatted_examples.append({"role": "user", "content": ex['description']})
    formatted_examples.append({"role": "assistant", "content": ex['caption']})
```

### 4.5 Create the Caption Generation Function

```python
def caption_image(description, model="gpt-4o-mini"):
    # Build the message list
    messages = [{"role": "system", "content": caption_system_prompt}]
    messages.extend(formatted_examples)
    messages.append({"role": "user", "content": description})

    # Call the API
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=messages
    )
    return response.choices[0].message.content
```

### 4.6 Test the Full Caption Pipeline

```python
# Test on a few new examples
examples = df.iloc[5:8]

for index, row in examples.iterrows():
    print(f"Title: {row['title'][:50]}... - {row['url']}")
    description = describe_image(row['primary_image'], row['title'])
    print(f"Description: {description}")
    caption = caption_image(description)
    print(f"Caption: {caption}\n---\n")
```

## 5. Enable Search with Generated Metadata

Now we'll use the generated keywords and captions to build a search system. We'll create embeddings for each product's combined metadata (keywords + caption) and use cosine similarity to find matches.

### 5.1 Prepare the Dataset with Generated Metadata

We'll create a function that processes a product row to generate keywords, a description, and a caption.

```python
import ast

def tag_and_caption(row, df_keywords):
    """
    Process a single product row to generate tags and captions.
    """
    # Step 1: Generate keywords
    keywords_raw = analyze_image(row['primary_image'], row['title'])
    try:
        keywords_list = ast.literal_eval(keywords_raw)
        # Deduplicate keywords against our reference list
        mapped_keywords = [replace_keyword(k, df_keywords) for k in keywords_list]
    except Exception as e:
        print(f"Error parsing keywords: {keywords_raw}")
        mapped_keywords = []

    # Step 2: Generate description and caption
    img_description = describe_image(row['primary_image'], row['title'])
    caption = caption_image(img_description)

    return {
        'keywords': mapped_keywords,
        'img_description': img_description,
        'caption': caption
    }
```

**Note:** Processing a large dataset can be time-consuming due to API calls. For demonstration, we'll process a small subset. You can save and load the results to avoid reprocessing.

```python
# Initialize an empty DataFrame for our reference keywords
df_keywords_ref = pd.DataFrame(columns=['keyword', 'embedding'])

# Add columns to our main DataFrame to store results
df['keywords'] = ''
df['img_description'] = ''
df['caption'] = ''

# Process the first 10 rows as an example
for index, row in df[:10].iterrows():
    print(f"Processing {index}: {row['title'][:50]}...")
    updates = tag_and_caption(row, df_keywords_ref)
    df.loc[index, updates.keys()] = updates.values()

# Save the processed data
df.to_csv("data/items_tagged_and_captioned_sample.csv", index=False)
```

### 5.2 Create Search Embeddings

For each product, create an embedding based on its combined keywords and caption.

```python
def embed_tags_caption(row):
    """
    Create a single embedding from a product's keywords and caption.
    """
    if row['caption'] != '':
        try:
            # Combine keywords and caption into a single string
            keywords_string = ",".join(k for k in row['keywords']) + '\n'
            content = keywords_string + row['caption']
            embedding = get_embedding(content)
            return embedding
        except Exception as e:
            print(f"Error creating embedding for {row['title']}: {e}")
    return None

# Apply the function to create embeddings
df['embedding'] = df.apply(lambda x: embed_tags_caption(x), axis=1)

# Remove rows without embeddings
df_search = df.dropna(subset=['embedding'])
print(f"Products with embeddings: {df_search.shape[0]}")

# Save the embeddings for later use
df_search.to_csv("data/items_with_embeddings_sample.csv", index=False)
```

### 5.3 Implement Text-Based Search

Create a function that finds the most similar products to a user's text query.

```python
def search_from_input_text(query, df_search, n=3):
    """
    Find the N most similar products to a text query.
    """
    # Embed the query
    query_embedding = get_embedding(query)

    # Calculate cosine similarity for each product
    df_search['similarity'] = df_search['embedding'].apply(
        lambda x: cosine_similarity(np.array(x).reshape(1, -1),
                                    np.array(query_embedding).reshape(1, -1))
    )

    # Return the top N matches
    most_similar = df_search.sort_values('similarity', ascending=False).iloc[:n]
    return most_similar
```

### 5.4 Test the Search Function

```python
# Example queries
user_queries = ['shoe storage', 'black metal side table', 'doormat']

for query in user_queries:
    print(f"\nSearch results for: '{query}'")
    results = search_from_input_text(query, df_search, n=2)

    for index, row in results.iterrows():
        # Extract the similarity score
        sim_score = row['similarity']
        if isinstance(sim_score, np.ndarray):
            sim_score = sim_score[0][0]

        print(f"  - {row['title'][:60]}... (Similarity: {sim_score:.3f})")
        print(f"    URL: {row['url']}")
```

## Summary

You've successfully built a pipeline that:
1.  Uses GPT-4o mini to generate relevant keywords from product images and titles.
2.  Deduplicates keywords using text embeddings to maintain a clean tag set.
3.  Creates detailed image descriptions and refines them into concise captions.
4.  Combines this metadata into embeddings to power a semantic search system.

This approach can significantly enhance product discoverability on e-commerce platforms, enabling both precise text search and the foundation for future image-to-image search capabilities.