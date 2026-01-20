# Retrieval augmented generation using Elasticsearch and OpenAI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](openai/openai-cookbook/blob/main/examples/vector_databases/elasticsearch/elasticsearch-retrieval-augmented-generation.ipynb)

This notebook demonstrates how to: 
- Index the OpenAI Wikipedia vector dataset into Elasticsearch 
- Embed a question with the OpenAI [`embeddings`](https://platform.openai.com/docs/api-reference/embeddings) endpoint
- Perform semantic search on the Elasticsearch index using the encoded question
- Send the top search results to the OpenAI [Chat Completions](https://platform.openai.com/docs/guides/gpt/chat-completions-api) API endpoint for retrieval augmented generation (RAG)

ℹ️ If you've already worked through our semantic search notebook, you can skip ahead to the final step!

## Install packages and import modules 

```python
# install packages

!python3 -m pip install -qU openai pandas wget elasticsearch

# import modules

from getpass import getpass
from elasticsearch import Elasticsearch, helpers
import wget
import zipfile
import pandas as pd
import json
import openai
```

## Connect to Elasticsearch

ℹ️ We're using an Elastic Cloud deployment of Elasticsearch for this notebook.
If you don't already have an Elastic deployment, you can sign up for a free [Elastic Cloud trial](https://cloud.elastic.co/registration?utm_source=github&utm_content=openai-cookbook).

To connect to Elasticsearch, you need to create a client instance with the Cloud ID and password for your deployment.

Find the Cloud ID for your deployment by going to https://cloud.elastic.co/deployments and selecting your deployment.

```python
CLOUD_ID = getpass("Elastic deployment Cloud ID")
CLOUD_PASSWORD = getpass("Elastic deployment Password")
client = Elasticsearch(
  cloud_id = CLOUD_ID,
  basic_auth=("elastic", CLOUD_PASSWORD) # Alternatively use `api_key` instead of `basic_auth`
)

# Test connection to Elasticsearch
print(client.info())
```

    {'name': 'instance-0000000001', 'cluster_name': '29ef9817e13142f5ba0ea7b29c2a86e2', 'cluster_uuid': 'absjWgQvRw63IlwWKisN8w', 'version': {'number': '8.9.1', 'build_flavor': 'default', 'build_type': 'docker', 'build_hash': 'a813d015ef1826148d9d389bd1c0d781c6e349f0', 'build_date': '2023-08-10T05:02:32.517455352Z', 'build_snapshot': False, 'lucene_version': '9.7.0', 'minimum_wire_compatibility_version': '7.17.0', 'minimum_index_compatibility_version': '7.0.0'}, 'tagline': 'You Know, for Search'}

## Download the dataset 

In this step we download the OpenAI Wikipedia embeddings dataset, and extract the zip file.

```python
embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'
wget.download(embeddings_url)

with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip",
"r") as zip_ref:
    zip_ref.extractall("data")
```

##  Read CSV file into a Pandas DataFrame.

Next we use the Pandas library to read the unzipped CSV file into a DataFrame. This step makes it easier to index the data into Elasticsearch in bulk.

```python

wikipedia_dataframe = pd.read_csv("data/vector_database_wikipedia_articles_embedded.csv")
```

## Create index with mapping

Now we need to create an Elasticsearch index with the necessary mappings. This will enable us to index the data into Elasticsearch.

We use the `dense_vector` field type for the `title_vector` and  `content_vector` fields. This is a special field type that allows us to store dense vectors in Elasticsearch.

Later, we'll need to target the `dense_vector` field for kNN search.

```python
index_mapping= {
    "properties": {
      "title_vector": {
          "type": "dense_vector",
          "dims": 1536,
          "index": "true",
          "similarity": "cosine"
      },
      "content_vector": {
          "type": "dense_vector",
          "dims": 1536,
          "index": "true",
          "similarity": "cosine"
      },
      "text": {"type": "text"},
      "title": {"type": "text"},
      "url": { "type": "keyword"},
      "vector_id": {"type": "long"}
      
    }
}

client.indices.create(index="wikipedia_vector_index", mappings=index_mapping)
```

## Index data into Elasticsearch 

The following function generates the required bulk actions that can be passed to Elasticsearch's Bulk API, so we can index multiple documents efficiently in a single request.

For each row in the DataFrame, the function yields a dictionary representing a single document to be indexed. 

```python
def dataframe_to_bulk_actions(df):
    for index, row in df.iterrows():
        yield {
            "_index": 'wikipedia_vector_index',
            "_id": row['id'],
            "_source": {
                'url' : row["url"],
                'title' : row["title"],
                'text' : row["text"],
                'title_vector' : json.loads(row["title_vector"]),
                'content_vector' : json.loads(row["content_vector"]),
                'vector_id' : row["vector_id"]
            }
        }
```

As the dataframe is large, we will index data in batches of `100`. We index the data into Elasticsearch using the Python client's [helpers](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/client-helpers.html#bulk-helpers) for the bulk API.

```python
start = 0
end = len(wikipedia_dataframe)
batch_size = 100
for batch_start in range(start, end, batch_size):
    batch_end = min(batch_start + batch_size, end)
    batch_dataframe = wikipedia_dataframe.iloc[batch_start:batch_end]
    actions = dataframe_to_bulk_actions(batch_dataframe)
    helpers.bulk(client, actions)
```

Let's test the index with a simple match query.

```python
print(client.search(index="wikipedia_vector_index", body={
    "_source": {
        "excludes": ["title_vector", "content_vector"]
    },
    "query": {
        "match": {
            "text": {
                "query": "Hummingbird"
            }
        }
    }
}))
```

    {'took': 10, 'timed_out': False, '_shards': {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0}, 'hits': {'total': {'value': 4, 'relation': 'eq'}, 'max_score': 14.917897, 'hits': [{'_index': 'wikipedia_vector_index', '_id': '34227', '_score': 14.917897, '_source': {'url': 'https://simple.wikipedia.org/wiki/Hummingbird', 'title': 'Hummingbird', 'text': "Hummingbirds are small birds of the family Trochilidae.\n\nThey are among the smallest of birds: most species measure 7.5–13\xa0cm (3–5\xa0in). The smallest living bird species is the 2–5\xa0cm Bee Hummingbird. They can hover in mid-air by rapidly flapping their wings 12–80 times per second (depending on the species). They are also the only group of birds able to fly backwards. Their rapid wing beats do actually hum. They can fly at speeds over 15\xa0m/s (54\xa0km/h, 34\xa0mi/h).\n\nEating habits and pollination \nHummingbirds help flowers to pollinate, though most insects are best known for doing so. The hummingbird enjoys nectar, like the butterfly and other flower-loving insects, such as bees.\n\nHummingbirds do not have a good sense of smell; instead, they are attracted to color, especially the color red. Unlike the butterfly, the hummingbird hovers over the flower as it drinks nectar from it, like a moth. When it does so, it flaps its wings very quickly to stay in one place, which makes it look like a blur and also beats so fast it makes a humming sound. A hummingbird sometimes puts its whole head into the flower to drink the nectar properly. When it takes its head back out, its head is covered with yellow pollen, so that when it moves to another flower, it can pollinate. Or sometimes it may pollinate with its beak.\n\nLike bees, hummingbirds can assess the amount of sugar in the nectar they eat. They reject flowers whose nectar has less than 10% sugar. Nectar is a poor source of nutrients, so hummingbirds meet their needs for protein, amino acids, vitamins, minerals, etc. by preying on insects and spiders.\n\nFeeding apparatus \nMost hummingbirds have bills that are long and straight or nearly so, but in some species the bill shape is adapted for specialized feeding. Thornbills have short, sharp bills adapted for feeding from flowers with short corollas and piercing the bases of longer ones. The Sicklebills' extremely decurved bills are adapted to extracting nectar from the curved corollas of flowers in the family Gesneriaceae. The bill of the Fiery-tailed Awlbill has an upturned tip, as in the Avocets. The male Tooth-billed Hummingbird has barracuda-like spikes at the tip of its long, straight bill.\n\nThe two halves of a hummingbird's bill have a pronounced overlap, with the lower half (mandible) fitting tightly inside the upper half (maxilla). When hummingbirds feed on nectar, the bill is usually only opened slightly, allowing the tongue to dart out into the nectar.\n\nLike the similar nectar-feeding sunbirds and unlike other birds, hummingbirds drink by using grooved or trough-like tongues which they can stick out a long way.\nHummingbirds do not spend all day flying, as the energy cost would be prohibitive; the majority of their activity consists simply of sitting or perching. Hummingbirds feed in many small meals, consuming many small invertebrates and up to twelve times their own body weight in nectar each day. They spend an average of 10–15% of their time feeding and 75–80% sitting and digesting.\n\nCo-evolution with flowers\n\nSince hummingbirds are specialized nectar-eaters, they are tied to the bird-flowers they feed upon. Some species, especially those with unusual bill shapes such as the Sword-billed Hummingbird and the sicklebills, are co-evolved with a small number of flower species.\n\nMany plants pollinated by hummingbirds produce flowers in shades of red, orange, and bright pink, though the birds will take nectar from flowers of many colors. Hummingbirds can see wavelengths into the near-ultraviolet. However, their flowers do not reflect these wavelengths as many insect-pollinated flowers do. The narrow color spectrum may make hummingbird-pollinated flowers inconspicuous to insects, thereby reducing nectar robbing by insects. Hummingbird-pollinated flowers also produce relatively weak nectar (averaging 25% sugars w/w) containing high concentrations of sucrose, whereas insect-pollinated flowers typically produce more concentrated nectars dominated by fructose and glucose.\n\nTaxonomy \nHummingbirds have traditionally been a part of the bird order Apodiformes. This order includes the hummingbirds, the swifts and the tree swifts. The Sibley-Ahlquist taxonomy of birds, based on DNA studies done in the 1970s and 1980s, changed the classification of hummingbirds. Instead of being in the same order as the swifts, the hummingbirds were made an order, the Trochiliformes. Their previous order, Apodiformes was changed to the superorder Apodimorphae. This superorder contains the three families of birds which were in it when it was an order.\n\nReferences", 'vector_id': 10024}}, {'_index': 'wikipedia_vector_index', '_id': '84773', '_score': 10.951234, '_source': {'url': 'https://simple.wikipedia.org/wiki/Inagua', 'title': 'Inagua', 'text': "Inagua is the southernmost district of the Bahamas.  It is the islands of Great Inagua and Little Inagua.\n\nGreat Inagua is the third largest island in the Bahamas at 596 square miles (1544\xa0km²) and lies about 55 miles (90\xa0km) from the eastern tip of Cuba. The island is about 55 × 19 miles (90 × 30\xa0km) in extent, the highest point being 108\xa0ft (33 m) on East Hill. It encloses several lakes, most notably the 12-mile long Lake Windsor (also called Lake Rosa) which occupies nearly ¼ of the interior. The population of Great Inagua is 969 (2000 census).\n\nThe island's capital and only harbour is Matthew Town.\n\nThere is a large bird sanctuary in the centre of the island.  There are more than 80,000 West Indian Flamingoes and many other exotic birds such as the native Bahama Parrot, the Bahama woodstar hummingbird, Bahama pintails, Brown pelicans, Tri-colored herons, Snowy egrets, Reddish egrets, Stripe-headed tanangers, Cormorants, Roseate spoonbills, American kestrels, and Burrowing owls.\n\nDistricts of the Bahamas\nIslands of the Bahamas\n1999 establishments in the Bahamas", 'vector_id': 22383}}, {'_index': 'wikipedia_vector_index', '_id': '3707', '_score': 1.1967773, '_source': {'url': 'https://simple.wikipedia.org/wiki/Bird', 'title': 'Bird', 'text': 'Birds (Aves) are a group of animals with backbones which evolved from dinosaurs. Technically speaking, they are dinosaurs. \n\nBirds are endothermic. The heat loss from their bodies is slowed down by their feathers. \nModern birds are toothless: they have beaked jaws. They lay hard-shelled eggs. They have a high metabolic rate, a four-chambered heart and a strong yet lightweight skeleton.\n\nBirds live all over the world. They range in size from the 5 cm (2 in) bee hummingbird to the 2.70 m (9 ft) ostrich. They are the tetrapods with the most living species: about ten thousand. More than half of these are passerines, sometimes known as perching birds.\n\nBirds are the closest living relatives of the Crocodilia. This is because they are the two main survivors of a once huge group called the Archosaurs. \n\nModern birds are not descended from Archaeopteryx. According to DNA evidence, modern birds (Neornithes) evolved in the long Upper Cretaceous period. More recent estimates showed that  modern birds originated early in the Upper Cretaceous.\n\nPrimitive bird-like dinosaurs are in the broader group Avialae. They have been found back to the mid-Jurassic period, around 170 million years ago. Many of these early "stem-birds", such as Anchiornis, were not yet capable of fully powered flight. Many had primitive characteristics like teeth in their jaws and long bony tails.p274\n\nThe Cretaceous–Palaeogene extinction event 66 million years ago killed off all the non-avian dinosaur lines. Birds, especially those in the southern continents, survived this event and then migrated to other parts of the world. Diversification occurred around the Cretaceous–Palaeogene extinction event. \n\nBirds have wings which are more or less developed depending on the species. The only known groups without wings are the extinct moa and elephant birds. Wings, which evolved from forelimbs, gave birds the ability to fly. Later, many groups evolved with reduced wings, such as ratites, penguins and many island species of birds. The digestive and respiratory systems of birds are also adapted for flight. Some bird species in aquatic environments, particularly seabirds and some waterbirds, have evolved as good swimmers.\n\nIn general, birds are effective, and inherit their behaviour almost entirely. The key elements of their life are inherited. It was a great discovery that birds never learn to fly. \nSo it is quite wrong to say, when a chick waves its wings in the nest "It\'s learning to fly". What the chick is doing is exercising its muscles. They develop the ability to fly automatically (assuming they are species that do fly). And if they are species which migrate, that behaviour is also inherited. Many species migrate over great distances each year. Other main features of their life may be inherited, though they can and do learn. Birds have good memories which they use, for example, when they search for food.\n\nSeveral bird species make and use tools. Some social species pass on some knowledge across generations, a form of culture. Birds are social. They communicate with visual signals, calls and bird songs. Most of their social behaviours are inherited, such as cooperative breeding and hunting, flocking and mobbing of predators.\n\nMost bird species are socially monogamous, usually for one breeding season at a time, sometimes for years, but rarely for life. Other species are polygynous (one male with many females) or, rarely, polyandrous (one female with many males). Birds produce offspring by laying eggs which are fertilised by sexual reproduction. They are often laid in a nest and incubated by the parents. Most birds have an extended period of parental care after hatching. Some birds, such as hens, lay eggs even when not fertilised, though unfertilised eggs do not produce offspring.\n\nMany species of birds are eaten by humans. Domesticated and undomesticated birds are sources of eggs, meat, and feathers.  In English, domesticated birds are often called poultry, undomesticated birds are called game. Songbirds, parrots and other species are popular as pets. Guano, which is bird manure, is harvested for use as a fertiliser. Birds figure throughout human culture. About 120–130 species have become extinct due to human activity since the 17th century and hundreds more before then. Human activity threatens about 1,200 bird species with extinction, though efforts are underway to protect them. Recreational bird-watching is an important part of the ecotourism industry.\n\nBird colours \n\nBirds come in a huge range