# Personalized Product Descriptions with Weaviate and the Gemini API

## Overview
This guide demonstrates how to build an AI-powered e-commerce application using Weaviate (a vector database) and the Gemini API. You'll learn to:
1. Store and search product data using semantic search
2. Generate AI-powered product descriptions
3. Create personalized descriptions for different customer personas

## Prerequisites
- **Gemini API Key**: Get one from [AI Studio](https://aistudio.google.com/)
- **Weaviate Cluster**: Choose from:
  - Weaviate Embedded (runs in your notebook)
  - Weaviate Cloud (managed service)
  - Local Docker instance

## Setup

### 1. Install Required Libraries
```bash
pip install weaviate-client==4.7.1
pip install -U "google-genai>=1.0.0"
pip install requests
pip install 'protobuf>=5'
```

### 2. Import Dependencies
```python
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.embedded import EmbeddedOptions
import weaviate.classes as wvc
from weaviate.util import generate_uuid5

import os
import json
import requests
from PIL import Image
from io import BytesIO
from IPython.display import display

# Helper functions for image handling
def url_to_pil(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def display_image(url, size=100):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    image = Image.open(image_data)
    resized_image = image.resize((size, size))
    display(resized_image)
```

### 3. Configure Gemini API
```python
from google import genai
from google.colab import userdata  # For Colab environments

GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")  # Or use os.environ.get()
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
```

## Part 1: Connect to Weaviate and Set Up Schema

### Step 1: Connect to Weaviate
For this tutorial, we'll use Weaviate Embedded for simplicity. It runs locally in your notebook session.

```python
client = weaviate.WeaviateClient(
    embedded_options=EmbeddedOptions(
        version="1.25.10",
        additional_env_vars={
            "ENABLE_MODULES": "text2vec-palm, generative-palm"
        }),
    additional_headers={
        "X-Google-Studio-Api-Key": GEMINI_API_KEY
    }
)

client.connect()
```

**Alternative Connection Options:**

**Weaviate Cloud:**
```python
WCD_URL = "https://sandbox.gcp.weaviate.cloud"
WCD_AUTH_KEY = "your-auth-key"

client = weaviate.connect_to_wcs(
    cluster_url=WCD_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WCD_AUTH_KEY),
    headers={"X-Google-Studio-Api-Key": GEMINI_API_KEY},
)
```

**Local Docker:**
```python
client = weaviate.connect_to_local()
```

### Step 2: Define the Schema
The schema defines how your data is structured in Weaviate. We'll create two collections: `Products` and `Personas`.

```python
# Clear existing collections (optional)
client.collections.delete("Products")
client.collections.delete("Personas")

# Configuration
PROJECT_ID = ""  # Leave empty for Gemini API
API_ENDPOINT = "generativelanguage.googleapis.com"
embedding_model = "text-embedding-004"
generative_model = "gemini-3-flash-preview"

# Create Products collection
if not client.collections.exists("Products"):
    products_collection = client.collections.create(
        name="Products",
        vectorizer_config=Configure.Vectorizer.text2vec_palm(
            project_id=PROJECT_ID,
            api_endpoint=API_ENDPOINT,
            model_id=embedding_model
        ),
        generative_config=Configure.Generative.palm(
            project_id=PROJECT_ID,
            api_endpoint=API_ENDPOINT,
            model_id=generative_model
        ),
        properties=[
            Property(name="product_id", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="category", data_type=DataType.TEXT),
            Property(name="link", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
            Property(name="brand", data_type=DataType.TEXT),
            Property(name="generated_description", data_type=DataType.TEXT),
        ]
    )

# Create Personas collection
if not client.collections.exists("Personas"):
    personas_collection = client.collections.create(
        name="Personas",
        vectorizer_config=Configure.Vectorizer.text2vec_palm(
            project_id=PROJECT_ID,
            api_endpoint=API_ENDPOINT,
            model_id=embedding_model
        ),
        generative_config=Configure.Generative.palm(
            project_id=PROJECT_ID,
            api_endpoint=API_ENDPOINT,
            model_id=generative_model
        ),
        properties=[
            Property(name="name", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
        ]
    )
```

### Step 3: Import Product Data
We'll use a sample dataset of Google merchandise products.

```python
# Load product data from URL
url = 'https://raw.githubusercontent.com/bkauf/next-store/main/first_99_objects.json'
response = requests.get(url)
data = json.loads(response.text)

# Inspect the first product
print("Sample product:", json.dumps(data[0], indent=2))
```

**Output:**
```json
{
  "id": "id_1",
  "product_id": "GGOEGAYC135814",
  "title": "Google Badge Tee",
  "category": "Apparel  Accessories Tops  Tees Tshirts",
  "link": "https://shop.googlemerchandisestore.com/store/20160512512/assets/items/images/GGOEGXXX1358.jpg",
  "description": "A classic crew neck tee made from 100 cotton Its soft and comfortable and features a small Google logo on the chest",
  "color": "['Blue']",
  "gender": "Unisex",
  "brand": "Google"
}
```

### Step 4: Upload Data to Weaviate
First, let's test with a single product:

```python
products = client.collections.get("Products")

# Insert first product
first_object = data[0]
products.data.insert(
    properties={
        "product_id": first_object['product_id'],
        "title": first_object['title'],
        "category": first_object['category'],
        "link": first_object['link'],
        "description": first_object['description'],
        "brand": first_object['brand']
    }
)

# Verify insertion
response = products.aggregate.over_all(total_count=True)
print(f"Total products: {response.total_count}")  # Should output 1
```

Now, import the remaining products using batch processing:

```python
remaining_data = data[1:]

with products.batch.dynamic() as batch:
    for item in remaining_data:
        batch.add_object(
            properties={
                "product_id": item['product_id'],
                "title": item['title'],
                "category": item['category'],
                "link": item['link'],
                "description": item['description'],
                "brand": item['brand']
            }
        )

# Verify all products are loaded
response = products.aggregate.over_all(total_count=True)
print(f"Total products: {response.total_count}")  # Should output 99
```

### Step 5: Inspect Stored Data
Let's examine how data is stored in Weaviate:

```python
# Get the first product
first_product = next(products.iterator())
print("UUID:", first_product.uuid)
print("Properties:", json.dumps(first_product.properties, indent=2))

# Fetch with vector data
product = products.query.fetch_object_by_id(
    first_product.uuid,
    include_vector=True
)

print(f"\nProduct: {product.properties['title']}")
print(f"Vector Dimensionality: {len(product.vector['default'])}")
print(f"Vector Preview (first 5 values): {product.vector['default'][:5]}")
```

## Part 2: Vector Search

### Step 1: Basic Vector Search
Use semantic search to find products similar to a text query:

```python
products = client.collections.get("Products")

response = products.query.near_text(
    query="travel mug",
    return_properties=["title", "description", "link"],
    limit=3
)

print("Search results for 'travel mug':")
for i, product in enumerate(response.objects, 1):
    print(f"\n{i}. {product.properties['title']}")
    print(f"   Description: {product.properties['description']}")
    # Uncomment to display images in notebook environments:
    # display_image(product.properties['link'])
```

### Step 2: Hybrid Search
Combine keyword (BM25) and vector search for better results:

```python
response = products.query.hybrid(
    query="Google mug for coffee",
    alpha=0.5,  # 0 = pure keyword search, 1 = pure vector search
    return_properties=["title", "description", "brand"],
    limit=3
)

print("Hybrid search results:")
for i, product in enumerate(response.objects, 1):
    print(f"\n{i}. {product.properties['title']}")
    print(f"   Brand: {product.properties['brand']}")
    print(f"   Description: {product.properties['description'][:100]}...")
```

## Part 3: Generative Feedback Loops

### Step 1: Generate Enhanced Product Descriptions
Use Gemini to create improved product descriptions and store them back in Weaviate:

```python
# Get a product that needs description enhancement
response = products.query.near_text(
    query="Google t-shirt",
    limit=1
)

product = response.objects[0]
print("Original product:")
print(f"Title: {product.properties['title']}")
print(f"Description: {product.properties['description']}")

# Generate enhanced description using Gemini
prompt = f"""
Create an engaging, marketing-focused product description for this item:

Product: {product.properties['title']}
Current Description: {product.properties['description']}
Brand: {product.properties['brand']}

Make it compelling for e-commerce, highlighting key features and benefits.
"""

generated_response = gemini_client.models.generate_content(
    model="gemini-1.5-flash",
    contents=prompt
)

generated_description = generated_response.text
print(f"\nGenerated Description:\n{generated_description}")

# Update the product with the generated description
products.data.update(
    uuid=product.uuid,
    properties={
        "generated_description": generated_description
    }
)

# Verify the update
updated_product = products.query.fetch_object_by_id(product.uuid)
print(f"\nUpdated product has generated_description: {updated_product.properties['generated_description'] is not None}")
```

## Part 4: Personalization

### Step 1: Create Customer Personas
Define different customer profiles for personalized recommendations:

```python
personas = client.collections.get("Personas")

# Define customer personas
customer_personas = [
    {
        "name": "Alice",
        "description": "Tech enthusiast, early adopter, values innovation and premium quality, works in software engineering"
    },
    {
        "name": "Bob", 
        "description": "Practical buyer, values functionality and durability, prefers classic designs, works in education"
    }
]

# Insert personas into Weaviate
with personas.batch.dynamic() as batch:
    for persona in customer_personas:
        batch.add_object(
            properties={
                "name": persona["name"],
                "description": persona["description"]
            }
        )

# Verify personas are stored
response = personas.aggregate.over_all(total_count=True)
print(f"Total personas: {response.total_count}")
```

### Step 2: Generate Personalized Product Descriptions
Create descriptions tailored to each customer persona:

```python
# Get a product to personalize
products = client.collections.get("Products")
response = products.query.near_text(
    query="Google hoodie",
    limit=1
)
product = response.objects[0]

# Get customer personas
personas = client.collections.get("Personas")
persona_objects = list(personas.iterator())

# Generate personalized descriptions
personalized_descriptions = {}

for persona in persona_objects:
    prompt = f"""
    Create a personalized product description for {persona.properties['name']}.
    
    Customer Profile: {persona.properties['description']}
    
    Product: {product.properties['title']}
    Original Description: {product.properties['description']}
    Brand: {product.properties['brand']}
    
    Tailor the description to appeal specifically to this customer's preferences and values.
    """
    
    generated_response = gemini_client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    
    personalized_descriptions[persona.properties['name']] = generated_response.text
    print(f"\n=== Description for {persona.properties['name']} ===")
    print(generated_response.text)
```

### Step 3: Create Personalized Collection (Optional)
Store personalized descriptions in a dedicated collection for retrieval:

```python
# Create Personalized collection if it doesn't exist
if not client.collections.exists("Personalized"):
    personalized_collection = client.collections.create(
        name="Personalized",
        vectorizer_config=Configure.Vectorizer.text2vec_palm(
            project_id=PROJECT_ID,
            api_endpoint=API_ENDPOINT,
            model_id=embedding_model
        ),
        properties=[
            Property(name="product_id", data_type=DataType.TEXT),
            Property(name="persona_name", data_type=DataType.TEXT),
            Property(name="personalized_description", data_type=DataType.TEXT),
        ]
    )

# Store personalized descriptions
personalized = client.collections.get("Personalized")

for persona_name, description in personalized_descriptions.items():
    personalized.data.insert(
        properties={
            "product_id": product.properties['product_id'],
            "persona_name": persona_name,
            "personalized_description": description
        }
    )

print("Personalized descriptions stored successfully!")
```

## Summary
You've successfully built an AI-powered e-commerce system that:

1. **Stores product data** in Weaviate with vector embeddings
2. **Performs semantic search** to find relevant products
3. **Generates enhanced descriptions** using the Gemini API
4. **Creates personalized content** for different customer personas

This workflow demonstrates how vector databases and LLMs can work together to create dynamic, personalized shopping experiences. You can extend this by:
- Adding more customer personas
- Implementing recommendation systems
- Creating A/B testing for different descriptions
- Integrating with a frontend application

## Cleanup
```python
# Disconnect from Weaviate
client.close()
```

Remember to manage your Weaviate cluster appropriately based on your deployment choice (embedded, cloud, or local).