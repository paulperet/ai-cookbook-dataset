# Building a Book Search Engine with Zilliz and OpenAI

This guide walks you through creating a semantic book search engine. You'll generate embeddings for book descriptions using OpenAI's API, store them in Zilliz Cloud, and perform similarity searches to find relevant books based on natural language queries.

## Prerequisites & Setup

Before you begin, ensure you have the following:
1.  An active [Zilliz Cloud](https://zilliz.com/doc/quick_start) account and a running database cluster.
2.  An [OpenAI API key](https://platform.openai.com/api-keys).

Install the required Python libraries:

```bash
pip install openai pymilvus datasets tqdm
```

## Step 1: Configure Your Environment

Set up your connection details and API keys. Replace the placeholder values with your own credentials.

```python
import openai

# Zilliz Cloud Configuration
URI = 'your_uri'  # e.g., 'https://your-cluster.aws-us-west-2.vectordb.zillizcloud.com:443'
TOKEN = 'your_token'  # Format: 'username:password' or 'api_key'

# Collection & Model Configuration
COLLECTION_NAME = 'book_search'
DIMENSION = 1536  # Dimension of OpenAI's text-embedding-3-small embeddings
OPENAI_ENGINE = 'text-embedding-3-small'
openai.api_key = 'sk-your-key'  # Your OpenAI API key

# Search Index Parameters
INDEX_PARAM = {
    'metric_type': 'L2',
    'index_type': "AUTOINDEX",
    'params': {}
}

# Query Search Parameters
QUERY_PARAM = {
    "metric_type": "L2",
    "params": {},
}

# Batch processing size for efficiency
BATCH_SIZE = 1000
```

## Step 2: Set Up the Zilliz Collection

Now, connect to your Zilliz Cloud database and create a collection to store your book data and embeddings.

```python
from pymilvus import connections, utility, FieldSchema, Collection, CollectionSchema, DataType

# 1. Connect to Zilliz Cloud
connections.connect(uri=URI, token=TOKEN)

# 2. Remove the collection if it already exists (for a clean start)
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# 3. Define the schema for the collection
fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]
schema = CollectionSchema(fields=fields)

# 4. Create the collection
collection = Collection(name=COLLECTION_NAME, schema=schema)

# 5. Create an index on the embedding field and load the collection into memory
collection.create_index(field_name="embedding", index_params=INDEX_PARAM)
collection.load()
```

## Step 3: Load the Book Dataset

We'll use a public dataset from Hugging Face containing over 1 million book titles and descriptions.

```python
import datasets

# Download the dataset (the train split is ~800MB)
dataset = datasets.load_dataset('Skelebor/book_titles_and_descriptions_en_clean', split='train')
```

## Step 4: Create the Embedding Function

Define a helper function that sends text to the OpenAI API and returns the corresponding vector embeddings.

```python
def embed(texts):
    """
    Converts a list of text strings into embeddings using OpenAI.
    """
    embeddings = openai.Embedding.create(
        input=texts,
        engine=OPENAI_ENGINE
    )
    # Extract just the embedding vectors from the API response
    return [x['embedding'] for x in embeddings['data']]
```

## Step 5: Insert Data into Zilliz in Batches

To efficiently process over a million records, we'll embed and insert the data in batches.

```python
from tqdm import tqdm

# Initialize lists to hold batch data
data_batch = [
    [],  # For titles
    [],  # For descriptions
]

# Iterate through the dataset with a progress bar
for i in tqdm(range(0, len(dataset))):
    # Add the current book's data to the batch
    data_batch[0].append(dataset[i]['title'])
    data_batch[1].append(dataset[i]['description'])

    # When the batch reaches the defined size, process it
    if len(data_batch[0]) % BATCH_SIZE == 0:
        # Generate embeddings for the descriptions in this batch
        embeddings = embed(data_batch[1])
        # Append embeddings to the batch data
        data_batch.append(embeddings)
        # Insert the full batch (titles, descriptions, embeddings) into Zilliz
        collection.insert(data_batch)
        # Reset the batch lists for the next chunk of data
        data_batch = [[], []]

# Don't forget the final, potentially incomplete batch
if len(data_batch[0]) != 0:
    embeddings = embed(data_batch[1])
    data_batch.append(embeddings)
    collection.insert(data_batch)
```

> **Note:** This insertion process will take some time due to the dataset size. You can interrupt the cell early to test with a smaller subset of data, though search accuracy may be lower.

## Step 6: Query Your Search Engine

With the data loaded, you can now perform semantic searches. Define a query function that finds books similar to a given text description.

```python
import textwrap

def query(queries, top_k=5):
    """
    Takes a string or list of strings as a search query and prints the top_k results.
    """
    # Ensure the input is a list for consistent processing
    if not isinstance(queries, list):
        queries = [queries]

    # Generate embeddings for the search query/queries
    search_embeddings = embed(queries)

    # Perform the vector similarity search in Zilliz
    results = collection.search(
        search_embeddings,
        anns_field='embedding',
        param=QUERY_PARAM,
        limit=top_k,
        output_fields=['title', 'description']  # Specify which fields to return
    )

    # Print the results in a readable format
    for i, hits in enumerate(results):
        print('Query:', queries[i])
        print('--- Top Results ---')
        for rank, hit in enumerate(hits):
            print(f'\tRank: {rank + 1}, Score: {hit.score:.4f}')
            print(f'\tTitle: {hit.entity.get("title")}')
            # Use textwrap to format long descriptions
            print(textwrap.fill(f'\tDescription: {hit.entity.get("description")}', 88))
            print()
```

## Step 7: Execute a Search

Test your book search engine with a natural language query.

```python
query('Book about a k-9 from europe')
```

**Example Output:**
```
Query: Book about a k-9 from europe
--- Top Results ---
	Rank: 1, Score: 0.3048
	Title: Bark M For Murder
	Description: Who let the dogs out? Evildoers beware! Four of mystery fiction's top
	storytellers are setting the hounds on your trail -- in an incomparable quartet of
	crime stories with a canine edge...

	Rank: 2, Score: 0.3283
	Title: Texas K-9 Unit Christmas: Holiday Hero\Rescuing Christmas
	Description: CHRISTMAS COMES WRAPPED IN DANGER Holiday Hero by Shirlee McCoy Emma
	Fairchild never expected to find trouble in sleepy Sagebrush, Texas...
```
*(Output truncated for brevity)*

## Summary

You've successfully built a semantic book search engine. You learned how to:
1.  Configure connections to Zilliz Cloud and OpenAI.
2.  Create a vector database collection with an appropriate schema and index.
3.  Load a large-scale dataset and generate embeddings in batches.
4.  Perform fast, accurate similarity searches using natural language queries.

You can extend this project by building a web interface, adding filtering by genre or author, or experimenting with different embedding models.