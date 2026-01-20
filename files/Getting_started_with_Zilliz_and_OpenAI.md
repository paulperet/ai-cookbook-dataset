# Getting Started with Zilliz and OpenAI
### Finding your next book

In this notebook we will be going over generating embeddings of book descriptions with OpenAI and using those embeddings within Zilliz to find relevant books. The dataset in this example is sourced from HuggingFace datasets, and contains a little over 1 million title-description pairs.

Lets begin by first downloading the required libraries for this notebook:
- `openai` is used for communicating with the OpenAI embedding service
- `pymilvus` is used for communicating with the Zilliz instance
- `datasets` is used for downloading the dataset
- `tqdm` is used for the progress bars



```python
! pip install openai pymilvus datasets tqdm
```

    [First Entry, ..., Last Entry]

To get Zilliz up and running take a look [here](https://zilliz.com/doc/quick_start). With your account and database set up, proceed to set the following values:
- URI: The URI your database is running on
- USER: Your database username
- PASSWORD: Your database password
- COLLECTION_NAME: What to name the collection within Zilliz
- DIMENSION: The dimension of the embeddings
- OPENAI_ENGINE: Which embedding model to use
- openai.api_key: Your OpenAI account key
- INDEX_PARAM: The index settings to use for the collection
- QUERY_PARAM: The search parameters to use
- BATCH_SIZE: How many texts to embed and insert at once


```python
import openai

URI = 'your_uri'
TOKEN = 'your_token' # TOKEN == user:password or api_key
COLLECTION_NAME = 'book_search'
DIMENSION = 1536
OPENAI_ENGINE = 'text-embedding-3-small'
openai.api_key = 'sk-your-key'

INDEX_PARAM = {
    'metric_type':'L2',
    'index_type':"AUTOINDEX",
    'params':{}
}

QUERY_PARAM = {
    "metric_type": "L2",
    "params": {},
}

BATCH_SIZE = 1000
```

## Zilliz
This segment deals with Zilliz and setting up the database for this use case. Within Zilliz we need to setup a collection and index it.


```python
from pymilvus import connections, utility, FieldSchema, Collection, CollectionSchema, DataType

# Connect to Zilliz Database
connections.connect(uri=URI, token=TOKEN)
```


```python
# Remove collection if it already exists
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
```


```python
# Create collection which includes the id, title, and embedding.
fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]
schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)
```


```python
# Create the index on the collection and load it.
collection.create_index(field_name="embedding", index_params=INDEX_PARAM)
collection.load()
```

## Dataset
With Zilliz up and running we can begin grabbing our data. `Hugging Face Datasets` is a hub that holds many different user datasets, and for this example we are using Skelebor's book dataset. This dataset contains title-description pairs for over 1 million books. We are going to embed each description and store it within Zilliz along with its title.


```python
import datasets

# Download the dataset and only use the `train` portion (file is around 800Mb)
dataset = datasets.load_dataset('Skelebor/book_titles_and_descriptions_en_clean', split='train')
```

    [First Entry, ..., Last Entry]

## Insert the Data
Now that we have our data on our machine we can begin embedding it and inserting it into Zilliz. The embedding function takes in text and returns the embeddings in a list format.


```python
# Simple function that converts the texts to embeddings
def embed(texts):
    embeddings = openai.Embedding.create(
        input=texts,
        engine=OPENAI_ENGINE
    )
    return [x['embedding'] for x in embeddings['data']]

```

This next step does the actual inserting. Due to having so many datapoints, if you want to immediately test it out you can stop the inserting cell block early and move along. Doing this will probably decrease the accuracy of the results due to less datapoints, but it should still be good enough.


```python
from tqdm import tqdm

data = [
    [], # title
    [], # description
]

# Embed and insert in batches
for i in tqdm(range(0, len(dataset))):
    data[0].append(dataset[i]['title'])
    data[1].append(dataset[i]['description'])
    if len(data[0]) % BATCH_SIZE == 0:
        data.append(embed(data[1]))
        collection.insert(data)
        data = [[],[]]

# Embed and insert the remainder 
if len(data[0]) != 0:
    data.append(embed(data[1]))
    collection.insert(data)
    data = [[],[]]

```

    [First Entry, ..., Last Entry]

## Query the Database
With our data safely inserted in Zilliz, we can now perform a query. The query takes in a string or a list of strings and searches them. The results print out your provided description and the results that include the result score, the result title, and the result book description.


```python
import textwrap

def query(queries, top_k = 5):
    if type(queries) != list:
        queries = [queries]
    res = collection.search(embed(queries), anns_field='embedding', param=QUERY_PARAM, limit = top_k, output_fields=['title', 'description'])
    for i, hit in enumerate(res):
        print('Description:', queries[i])
        print('Results:')
        for ii, hits in enumerate(hit):
            print('\t' + 'Rank:', ii + 1, 'Score:', hits.score, 'Title:', hits.entity.get('title'))
            print(textwrap.fill(hits.entity.get('description'), 88))
            print()
```


```python
query('Book about a k-9 from europe')
```

    Description: Book about a k-9 from europe
    Results:
    	Rank: 1 Score: 0.3047754764556885 Title: Bark M For Murder
    Who let the dogs out? Evildoers beware! Four of mystery fiction's top storytellers are
    setting the hounds on your trail -- in an incomparable quartet of crime stories with a
    canine edge. Man's (and woman's) best friends take the lead in this phenomenal
    collection of tales tense and surprising, humorous and thrilling: New York
    Timesbestselling author J.A. Jance's spellbinding saga of a scam-busting septuagenarian
    and her two golden retrievers; Anthony Award winner Virginia Lanier's pureblood thriller
    featuring bloodhounds and bloody murder; Chassie West's suspenseful stunner about a
    life-saving German shepherd and a ghastly forgotten crime; rising star Lee Charles
    Kelley's edge-of-your-seat yarn that pits an ex-cop/kennel owner and a yappy toy poodle
    against a craven killer.
    
    	Rank: 2 Score: 0.3283390402793884 Title: Texas K-9 Unit Christmas: Holiday Hero\Rescuing Christmas
    CHRISTMAS COMES WRAPPED IN DANGER Holiday Hero by Shirlee McCoy Emma Fairchild never
    expected to find trouble in sleepy Sagebrush, Texas. But when she's attacked and left
    for dead in her own diner, her childhood friend turned K-9 cop Lucas Harwood offers a
    chance at justice--and love. Rescuing Christmas by Terri Reed She escaped a kidnapper,
    but now a killer has set his sights on K-9 dog trainer Lily Anderson. When fellow
    officer Jarrod Evans appoints himself her bodyguard, Lily knows more than her life is at
    risk--so is her heart. Texas K-9 Unit: These lawmen solve the toughest cases with the
    help of their brave canine partners
    
    	Rank: 3 Score: 0.33899369835853577 Title: Dogs on Duty: Soldiers' Best Friends on the Battlefield and Beyond
    When the news of the raid on Osama Bin Laden's compound broke, the SEAL team member that
    stole the show was a highly trained canine companion. Throughout history, dogs have been
    key contributors to military units. Dorothy Hinshaw Patent follows man's best friend
    onto the battlefield, showing readers why dogs are uniquely qualified for the job at
    hand, how they are trained, how they contribute to missions, and what happens when they
    retire. With full-color photographs throughout and sidebars featuring heroic canines
    throughout history, Dogs on Duty provides a fascinating look at these exceptional
    soldiers and companions.
    
    	Rank: 4 Score: 0.34207457304000854 Title: Toute Allure: Falling in Love in Rural France
    After saying goodbye to life as a successful fashion editor in London, Karen Wheeler is
    now happy in her small village house in rural France. Her idyll is complete when she
    meets the love of her life - he has shaggy hair, four paws and a wet nose!
    
    	Rank: 5 Score: 0.343595951795578 Title: Otherwise Alone (Evan Arden, #1)
    Librarian's note: This is an alternate cover edition for ASIN: B00AP5NNWC. Lieutenant
    Evan Arden sits in a shack in the middle of nowhere, waiting for orders that will send
    him back home - if he ever gets them. Other than his loyal Great Pyrenees, there's no
    one around to break up the monotony. The tedium is excruciating, but it is suddenly
    interrupted when a young woman stumbles up his path. "It's only 50-something pages, but
    in that short amount of time, the author's awesome writing packs in a whole lotta
    character detail. And sets the stage for the series, perfectly." -Maryse.net, 4.5 Stars
    He has two choices - pick her off from a distance with his trusty sniper-rifle, or dare
    let her approach his cabin and enter his life. Why not? It's been ages, and he is
    otherwise alone...
    
