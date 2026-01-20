# Kusto as a Vector database for AI embeddings

This Notebook provides step by step instuctions on using Azure Data Explorer (Kusto) as a vector database with OpenAI embeddings. 

This notebook presents an end-to-end process of:

1. Using precomputed embeddings created by OpenAI API.
2. Storing the embeddings in Kusto.
3. Converting raw text query to an embedding with OpenAI API.
4. Using Kusto to perform cosine similarity search in the stored embeddings

### Prerequisites

For the purposes of this exercise we need to prepare a couple of things:

1. Azure Data Explorer(Kusto) server instance. https://azure.microsoft.com/en-us/products/data-explorer
3. Azure OpenAI credentials or OpenAI API key.

```python
%pip install wget
```

[Collecting wget, ..., Successfully installed wget-3.2]

```python
%pip install openai
```

[Collecting openai, ..., Successfully installed openai-0.27.6]

```python
%pip install azure-kusto-data
```

[Requirement already satisfied: azure-kusto-data in /nfs4/pyenv-27214bb4-edfd-4fdd-b888-8a99075a1416/lib/python3.10/site-packages (4.1.4), ...]

### Download precomputed Embeddings

In this section we are going to load prepared embedding data, so you don't have to recompute the embeddings of Wikipedia articles with your own credits.

```python
import wget

embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"

# The file is ~700 MB so this will take some time
wget.download(embeddings_url)
```

'vector_database_wikipedia_articles_embedded.zip'

```python
import zipfile

with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip","r") as zip_ref:
    zip_ref.extractall("/lakehouse/default/Files/data")
```

```python
import pandas as pd

from ast import literal_eval

article_df = pd.read_csv('/lakehouse/default/Files/data/vector_database_wikipedia_articles_embedded.csv')
# Read vectors from strings back into a list
article_df["title_vector"] = article_df.title_vector.apply(literal_eval)
article_df["content_vector"] = article_df.content_vector.apply(literal_eval)
article_df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>url</th>
      <th>title</th>
      <th>text</th>
      <th>title_vector</th>
      <th>content_vector</th>
      <th>vector_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>https://simple.wikipedia.org/wiki/April</td>
      <td>April</td>
      <td>April is the fourth month of the year in the J...</td>
      <td>[0.001009464613161981, -0.020700545981526375, ...</td>
      <td>[-0.011253940872848034, -0.013491976074874401,...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>https://simple.wikipedia.org/wiki/August</td>
      <td>August</td>
      <td>August (Aug.) is the eighth month of the year ...</td>
      <td>[0.0009286514250561595, 0.000820168002974242, ...</td>
      <td>[0.0003609954728744924, 0.007262262050062418, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>https://simple.wikipedia.org/wiki/Art</td>
      <td>Art</td>
      <td>Art is a creative activity that expresses imag...</td>
      <td>[0.003393713850528002, 0.0061537534929811954, ...</td>
      <td>[-0.004959689453244209, 0.015772193670272827, ...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>https://simple.wikipedia.org/wiki/A</td>
      <td>A</td>
      <td>A or a is the first letter of the English alph...</td>
      <td>[0.0153952119871974, -0.013759135268628597, 0....</td>
      <td>[0.024894846603274345, -0.022186409682035446, ...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>https://simple.wikipedia.org/wiki/Air</td>
      <td>Air</td>
      <td>Air refers to the Earth's atmosphere. Air is a...</td>
      <td>[0.02224554680287838, -0.02044147066771984, -0...</td>
      <td>[0.021524671465158463, 0.018522677943110466, -...</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

### Store vectors in a Kusto table

Create a table & load the vectors in Kusto based on the contents in the dataframe. The spark option CreakeIfNotExists will automatically create a table if it doesn't exist

```python
# replace with your AAD Tenant ID, Kusto Cluster URI, Kusto DB name and Kusto Table
AAD_TENANT_ID = ""
KUSTO_CLUSTER =  ""
KUSTO_DATABASE = "Vector"
KUSTO_TABLE = "Wiki"
```

```python
kustoOptions = {"kustoCluster": KUSTO_CLUSTER, "kustoDatabase" :KUSTO_DATABASE, "kustoTable" : KUSTO_TABLE }

# Replace the auth method based on your desired authentication mechanism  - https://github.com/Azure/azure-kusto-spark/blob/master/docs/Authentication.md
access_token=mssparkutils.credentials.getToken(kustoOptions["kustoCluster"])
```

```python
#Pandas data frame to spark dataframe
sparkDF=spark.createDataFrame(article_df)
```

/opt/spark/python/lib/pyspark.zip/pyspark/sql/pandas/conversion.py:604: FutureWarning: iteritems is deprecated and will be removed in a future version. Use .items instead.

```python
# Write data to a Kusto table
sparkDF.write. \
format("com.microsoft.kusto.spark.synapse.datasource"). \
option("kustoCluster",kustoOptions["kustoCluster"]). \
option("kustoDatabase",kustoOptions["kustoDatabase"]). \
option("kustoTable", kustoOptions["kustoTable"]). \
option("accessToken", access_token). \
option("tableCreateOptions", "CreateIfNotExist").\
mode("Append"). \
save()
```

### Prepare your OpenAI API key

The OpenAI API key is used for vectorization of the documents and queries. You can follow the instructions to create and retrieve your Azure OpenAI key and endpoint. https://learn.microsoft.com/en-us/azure/cognitive-services/openai/tutorials/embeddings

Please make sure to use the `text-embedding-3-small` model. Since the precomputed embeddings were created with `text-embedding-3-small` model we also have to use it during search.

```python
import openai
```

#### If using Azure Open AI

```python
openai.api_version = '2022-12-01'
openai.api_base = '' # Please add your endpoint here
openai.api_type = 'azure'
openai.api_key = ''  # Please add your api key here

def embed(query):
    # Creates embedding vector from user query
    embedded_query = openai.Embedding.create(
            input=query,
            deployment_id="embed", #replace with your deployment id
            chunk_size=1
    )["data"][0]["embedding"]
    return embedded_query
```

#### If using Open AI

Only run this cell if you plan to use Open AI for embedding

```python
openai.api_key = ""


def embed(query):
    # Creates embedding vector from user query
    embedded_query = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small",
    )["data"][0]["embedding"]
    return embedded_query
```

### Generate embedding for the search term

```python
searchedEmbedding = embed("places where you worship")
#print(searchedEmbedding)
```

#### Semantic search in Kusto 

We will search the Kusto table for the closest vectors.

We will be using the series-cosine-similarity-fl UDF for similarity search. 

Please create the function in your database before proceeding -
https://learn.microsoft.com/en-us/azure/data-explorer/kusto/functions-library/series-cosine-similarity-fl?tabs=query-defined

```python
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError
from azure.kusto.data.helpers import dataframe_from_result_table
import pandas as pd
```

```python
KCSB = KustoConnectionStringBuilder.with_aad_device_authentication(
    KUSTO_CLUSTER)
KCSB.authority_id = AAD_TENANT_ID
```

```python
KUSTO_CLIENT = KustoClient(KCSB)
```

```python
KUSTO_QUERY = "Wiki | extend similarity = series_cosine_similarity_fl(dynamic("+str(searchedEmbedding)+"), content_vector,1,1) | top 10 by similarity desc "

RESPONSE = KUSTO_CLIENT.execute(KUSTO_DATABASE, KUSTO_QUERY)
```

```python
df = dataframe_from_result_table(RESPONSE.primary_results[0])
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>url</th>
      <th>title</th>
      <th>text</th>
      <th>title_vector</th>
      <th>content_vector</th>
      <th>vector_id</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>852</td>
      <td>https://simple.wikipedia.org/wiki/Temple</td>
      <td>Temple</td>
      <td>A temple is a building where people go to prac...</td>
      <td>[-0.021837441250681877, -0.007722342386841774,...</td>
      <td>[-0.0019541378132998943, 0.007151313126087189,...</td>
      <td>413</td>
      <td>0.834495</td>
    </tr>
    <tr>
      <th>1</th>
      <td>78094</td>
      <td>https://simple.wikipedia.org/wiki/Christian%20...</td>
      <td>Christian worship</td>
      <td>In Christianity, worship has been thought as b...</td>
      <td>[0.0017675267299637198, -0.008890199474990368,...</td>
      <td>[0.020530683919787407, 0.0024345638230443, -0....</td>
      <td>20320</td>
      <td>0.832132</td>
    </tr>
    <tr>
      <th>2</th>
      <td>59154</td>
      <td>https://simple.wikipedia.org/wiki/Service%20of...</td>
      <td>Service of worship</td>
      <td>A service of worship is a religious meeting wh...</td>
      <td>[-0.007969820871949196, 0.0004240311391185969,...</td>
      <td>[0.003784010885283351, -0.0030924836173653603,...</td>
      <td>15519</td>
      <td>0.831633</td>
    </tr>
    <tr>
      <th>3</th>
      <td>51910</td>
      <td>https://simple.wikipedia.org/wiki/Worship</td>
      <td>Worship</td>
      <td>Worship is a word often used in religion.  It ...</td>
      <td>[0.0036036288365721703, -0.01276545226573944, ...</td>
      <td>[0.007925753481686115, -0.0110504487529397, 0....</td>
      <td>14010</td>
      <td>0.828185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29576</td>
      <td>https://simple.wikipedia.org/wiki/Altar</td>
      <td>Altar</td>
      <td>An altar is a place, often a table, where a re...</td>
      <td>[0.007887467741966248, -0.02706138789653778, -...</td>
      <td>[0.023901859298348427, -0.031175222247838977, ...</td>
      <td>8708</td>
      <td>0.824124</td>
    </tr>
    <tr>
      <th>5</th>
      <td>92507</td>
      <td>https://simple.wikipedia.org/wiki/Shrine</td>
      <td>Shrine</td>
      <td>A shrine is a holy or sacred place with someth...</td>
      <td>[-0.011601685546338558, 0.006366696208715439, ...</td>
      <td>[0.016423320397734642, -0.0015560361789539456,...</td>
      <td>23945</td>
      <td>0.823863</td>
    </tr>
    <tr>
      <th>6</th>
      <td>815</td>
      <td>https://simple.wikipedia.org/wiki/Synagogue</td>
      <td>Synagogue</td>
      <td>A synagogue is a place where Jews meet to wors...</td>
      <td>[-0.017317570745944977, 0.0022673190105706453,...</td>
      <td>[-0.004515442531555891, 0.003739549545571208, ...</td>
      <td>398</td>
      <td>0.819942</td>
    </tr>
    <tr>
      <th>7</th>
      <td>68080</td>
      <td>https://simple.wikipedia.org/wiki/Shinto%20shrine</td>
      <td>Shinto shrine</td>
      <td>A Shinto shrine is a sacred place or site wher...</td>
      <td>[0.0035740730818361044, 0.0028098472394049168,...</td>
      <td>[0.011014971882104874, 0.00042272370774298906,...</td>
      <td>18106</td>
      <td>0.818475</td>
    </tr>
    <tr>
      <th>8</th>
      <td>57790</td>
      <td>https://simple.wikipedia.org/wiki/Chapel</td>
      <td>Chapel</td>
      <td>A chapel is a place for Christian worship. The...</td>
      <td>[-0.01371884811669588, 0.0031672674231231213, ...</td>
      <td>[0.002526090247556567, 0.02482965588569641, 0....</td>
      <td>15260</td>
      <td>0.817608</td>
    </tr>
    <tr>
      <th>9</th>
      <td>142</td>
      <td>https://simple.wikipedia.org/wiki/Church%20%28...</td>
      <td>Church (building)</td>
      <td>A church is a building that was constructed to...</td>
      <td>[0.0021336888894438744, 0.0029748091474175453,...</td>
      <td>[0.016109377145767212, 0.022908871993422508, 0....</td>
      <td>74</td>
      <td>0.812636</td>
    </tr>
  </tbody>
</table>
</div>

```python
searchedEmbedding = embed("unfortunate events in history")
```

```python
KUSTO_QUERY = "Wiki | extend similarity = series_cosine_similarity_fl(dynamic("+str(searchedEmbedding)+"), title_vector,1,1) | top 10 by similarity desc "
RESPONSE = KUSTO_CLIENT.execute(KUSTO_DATABASE, KUSTO_QUERY)

df = dataframe_from_result_table(RESPONSE.primary_results[0])
df
```

<div>
<style scoped