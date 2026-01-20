# 🛰️ Finetuning Pixtral on a satellite imagery dataset 🛰️

TL;DR: This notebook will show you:
- How to call Mistral's **batch inference** API
- How to **pass images** (encoded in base64) **in your API calls** to Mistral's VLM (here Pixtral-12B)
- How to **fine-tune Pixtral-12B** on an image classification problem in order to improve its accuracy.

For additional references check out the docs:
- https://docs.mistral.ai/capabilities/finetuning/
- https://docs.mistral.ai/capabilities/vision/


```python
from IPython.display import clear_output
!pip install mistralai==1.9.3
clear_output()
```

## Prepare the dataset

We will use [AID: A scene classification dataset](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets) introduced by Xia et al. hosted on Kaggle under a Public Domain license.

To downloading it, you will have to generate your Kaggle API token:
- Go to your Kaggle account in the [Kaggle API Token](https://www.kaggle.com/settings/account) section,
- Click "Create New API Token" → this will download kaggle.json.
- Upload kaggle.json to Google Colab.

## Download and parse the data


```python
def is_colab_runtime() -> bool:
    try:
        import google.colab
        return True
    except ImportError:
        return False
```


```python
if is_colab_runtime():

    from google.colab import files

    # This will prompt you to upload kaggle.json
    print("Please upload your kaggle.json file below:")
    files.upload()
    clear_output()
```


```python
from pathlib import Path

if (Path() / "kaggle.json").exists():
  !mkdir -p ~/.kaggle
  !cp kaggle.json ~/.kaggle/
  !chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d jiayuanchengala/aid-scene-classification-datasets
# This might take a few minutes
!unzip aid-scene-classification-datasets.zip -d satellite_dataset

clear_output()
```

The dataset consists in:
- satelite images (jpg files)
- each image belongs to a specific class (e.g. airport, commercial, dense residential, medium residential, park, forest, farmland etc.)

We first transform this dataset into something usable for finetuning
- Create pairs of (image, labels)
- Load the images and encode them in base64 (the format expected by Pixtral API)
- Downgrade the quality of the image in order to be a bit more memory-efficient

_Note that smaller, specialized vision models could potentially achieve comparable performance levels. This cookbook aims to guide you through the process of effectively fine-tuning Mistral’s Vision Language Model (VLM) using a straightforward example, and to demonstrate its impact on basic classification metrics. More advanced applications of fine-tuning could include interactions like "speak with an image" or generating image captions._


```python
from pathlib import Path
import pandas as pd
from PIL import Image
import base64
import io
from sklearn.model_selection import train_test_split

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Extract pairs of (image, label)
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

root_dir = Path() / "satellite_dataset" / "AID"
data = []
for d in root_dir.iterdir():
    if not d.is_dir():
        continue
    data.extend([{"label": d.name, "img_path": p} for p in d.iterdir()])

dataset_df = pd.DataFrame(data)
classes = [*dataset_df["label"].unique()]

# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Load image and encode in base64 (this might take a few minutes)
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# NOTE: This is not needed here, but a nice additional step would be to resize
# the images into 1024 longest edge (if the image was too big)
# For more details see: https://docs.mistral.ai/capabilities/vision/

def encode_image_to_base64(image_path: str | Path) -> str:
    image = Image.open(image_path)

    # Resize the image by a factor of 0.5
    new_size = (image.width // 2, image.height // 2)
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)

    encoded_string = (
        "data:image/jpeg;base64,"
        + base64.b64encode(buffer.read()).decode('utf-8')
    )

    return encoded_string


dataset_df["img_b64"] = [
    encode_image_to_base64(img_path)
    for img_path in dataset_df["img_path"]
]


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Split dataset in train / test
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

train_df, test_df = train_test_split(
    dataset_df,
    test_size=0.2,
    random_state=42,
    stratify=dataset_df["label"]
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Release a bit of memory
del dataset_df


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Light check at the data
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
print("Classes:", classes)
print("Size train:", len(train_df))
print("Size test:", len(test_df))

train_df.head()
```

    Classes: ['School', 'Farmland', 'Airport', 'BaseballField', 'Resort', 'Viaduct', 'Forest', 'Beach', 'Parking', 'MediumResidential', 'Pond', 'Park', 'Port', 'Meadow', 'BareLand', 'Playground', 'SparseResidential', 'Desert', 'DenseResidential', 'Bridge', 'Square', 'River', 'StorageTanks', 'Commercial', 'Center', 'Stadium', 'Industrial', 'RailwayStation', 'Mountain', 'Church']
    Size train: 8000
    Size test: 2000






  <div id="df-bd97ac81-772c-4b8e-9efa-ed5128a6cc85" class="colab-df-container">
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
      <th>label</th>
      <th>img_path</th>
      <th>img_b64</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bridge</td>
      <td>satellite_dataset/AID/Bridge/bridge_268.jpg</td>
      <td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mountain</td>
      <td>satellite_dataset/AID/Mountain/mountain_220.jpg</td>
      <td>
    </tr>
    <tr>
      <th>2</th>
      <td>Park</td>
      <td>satellite_dataset/AID/Park/park_161.jpg</td>
      <td>
    </tr>
    <tr>
      <th>3</th>
      <td>Farmland</td>
      <td>satellite_dataset/AID/Farmland/farmland_173.jpg</td>
      <td>
    </tr>
    <tr>
      <th>4</th>
      <td>Center</td>
      <td>satellite_dataset/AID/Center/center_256.jpg</td>
      <td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-bd97ac81-772c-4b8e-9efa-ed5128a6cc85')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-bd97ac81-772c-4b8e-9efa-ed5128a6cc85 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-bd97ac81-772c-4b8e-9efa-ed5128a6cc85');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-21a865f2-5b3f-4da1-9491-6dc9d757d403">
      <button class="colab-df-quickchart" onclick="quickchart('df-21a865f2-5b3f-4da1-9491-6dc9d757d403')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-21a865f2-5b3f-4da1-9491-6dc9d757d403 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
from IPython.display import display, HTML

def display_image(dataset_df: pd.DataFrame, idx: int) -> None:
    img_b64 = dataset_df["img_b64"].iloc[idx]
    label = dataset_df["label"].iloc[idx]
    display(HTML(f'<h2>{label}</h2><img src="{img_b64