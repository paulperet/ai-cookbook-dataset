## Semantic text search using embeddings

We can search through all our reviews semantically in a very efficient manner and at very low cost, by embedding our search query, and then finding the most similar reviews. The dataset is created in the [Get_embeddings_from_dataset Notebook](Get_embeddings_from_dataset.ipynb).

```python
import pandas as pd
import numpy as np
from ast import literal_eval

datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"

df = pd.read_csv(datafile_path)
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)
```

Here we compare the cosine similarity of the embeddings of the query and the documents, and show top_n best matches.

```python
from utils.embeddings_utils import get_embedding, cosine_similarity

# search through the reviews for a specific product
def search_reviews(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        model="text-embedding-3-small"
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results

results = search_reviews(df, "delicious beans", n=3)
```

```
Delicious!:  I enjoy this white beans seasoning, it gives a rich flavor to the beans I just love it, my mother in law didn't know about this Zatarain's brand and now she is traying different seasoning

Fantastic Instant Refried beans:  Fantastic Instant Refried Beans have been a staple for my family now for nearly 20 years.  All 7 of us love it and my grown kids are passing on the tradition.

Delicious:  While there may be better coffee beans available, this is my first purchase and my first time grinding my own beans.  I read several reviews before purchasing this brand, and am extremely 
```

```python
results = search_reviews(df, "whole wheat pasta", n=3)
```

```
Tasty and Quick Pasta:  Barilla Whole Grain Fusilli with Vegetable Marinara is tasty and has an excellent chunky vegetable marinara.  I just wish there was more of it.  If you aren't starving or on a 

sooo good:  tastes so good. Worth the money. My boyfriend hates wheat pasta and LOVES this. cooks fast tastes great.I love this brand and started buying more of their pastas. Bulk is best.

Bland and vaguely gamy tasting, skip this one:  As far as prepared dinner kits go, "Barilla Whole Grain Mezze Penne with Tomato and Basil Sauce" just did not do it for me...and this is coming from a p
```

We can search through these reviews easily. To speed up computation, we can use a special algorithm, aimed at faster search through embeddings.

```python
results = search_reviews(df, "bad delivery", n=1)
```

```
great product, poor delivery:  The coffee is excellent and I am a repeat buyer.  Problem this time was with the UPS delivery.  They left the box in front of my garage door in the middle of the drivewa
```

As we can see, this can immediately deliver a lot of value. In this example we show being able to quickly find the examples of delivery failures.

```python
results = search_reviews(df, "spoilt", n=1)
```

```
Disappointed:  The metal cover has severely disformed. And most of the cookies inside have been crushed into small pieces. Shopping experience is awful. I'll never buy it online again.
```

```python
results = search_reviews(df, "pet food", n=2)
```

```
Great food!:  I wanted a food for a a dog with skin problems. His skin greatly improved with the switch, though he still itches some.  He loves the food. No recalls, American made with American ingred

Great food!:  I wanted a food for a a dog with skin problems. His skin greatly improved with the switch, though he still itches some.  He loves the food. No recalls, American made with American ingred
```