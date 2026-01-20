## Visualizing embeddings in W&B

We will upload the data to [Weights & Biases](http://wandb.ai) and use an [Embedding Projector](https://docs.wandb.ai/ref/app/features/panels/weave/embedding-projector) to visualize the embeddings using common dimension reduction algorithms like PCA, UMAP, and t-SNE. The dataset is created in the [Get_embeddings_from_dataset Notebook](Get_embeddings_from_dataset.ipynb).

## What is Weights & Biases?

[Weights & Biases](http://wandb.ai) is a machine learning platform used by OpenAI and other ML teams to build better models faster. They use it to quickly track experiments, evaluate model performance, reproduce models, visualize results, and share findings with colleagues.

### 1. Log the data to W&B

We create a [W&B Table](https://docs.wandb.ai/guides/data-vis/log-tables) with the original data and the embeddings. Each review is a new row and the 1536 embedding floats are given their own column named `emb_{i}`.

```python
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from ast import literal_eval

# Load the embeddings
datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"
df = pd.read_csv(datafile_path)

# Convert to a list of lists of floats
matrix = np.array(df.embedding.apply(literal_eval).to_list())
```

```python
import wandb

original_cols = df.columns[1:-1].tolist()
embedding_cols = ['emb_'+str(idx) for idx in range(len(matrix[0]))]
table_cols = original_cols + embedding_cols

with wandb.init(project='openai_embeddings'):
    table = wandb.Table(columns=table_cols)
    for i, row in enumerate(df.to_dict(orient="records")):
        original_data = [row[col_name] for col_name in original_cols]
        embedding_data = matrix[i].tolist()
        table.add_data(*(original_data + embedding_data))
    wandb.log({'openai_embedding_table': table})
```

### 2. Render as 2D Projection

After navigating to the W&B run link, we click the ⚙️ icon in the top right of the Table and change "Render As:" to "Combined 2D Projection". 

Example: http://wandb.me/openai_embeddings