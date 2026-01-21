# Fine-Tuning Pixtral on Satellite Imagery: A Step-by-Step Guide

This guide will walk you through fine-tuning Mistral's Pixtral-12B Vision Language Model (VLM) on a satellite image classification task. You'll learn how to:

- Call Mistral's batch inference API
- Pass base64-encoded images in API calls to Pixtral
- Fine-tune Pixtral-12B to improve its accuracy on a specific image classification problem

## Prerequisites

Before starting, ensure you have the necessary dependencies installed:

```bash
pip install mistralai==1.3.0
```

## Step 1: Prepare Your Dataset

We'll use the **AID: A Scene Classification Dataset** from Kaggle, which contains satellite images across 30 different scene categories under a Public Domain license.

### 1.1 Set Up Kaggle Authentication

To download the dataset, you need a Kaggle API token:

1. Go to your Kaggle account's [API Token section](https://www.kaggle.com/settings/account)
2. Click "Create New API Token" to download `kaggle.json`
3. Upload `kaggle.json` to your working directory

### 1.2 Download and Extract the Dataset

```python
from pathlib import Path

# Set up Kaggle authentication
if (Path() / "kaggle.json").exists():
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json

# Download and extract the dataset
!kaggle datasets download -d jiayuanchengala/aid-scene-classification-datasets
!unzip aid-scene-classification-datasets.zip -d satellite_dataset
```

## Step 2: Process and Prepare the Data

The dataset consists of satellite images (JPEG files) organized into folders by class. We need to:

1. Create (image, label) pairs
2. Load images and encode them in base64 (Pixtral's expected format)
3. Resize images for memory efficiency
4. Split into training and test sets

```python
from pathlib import Path
import pandas as pd
from PIL import Image
import base64
import io
from sklearn.model_selection import train_test_split

# Define the root directory
root_dir = Path() / "satellite_dataset" / "AID"

# Extract (image, label) pairs
data = []
for directory in root_dir.iterdir():
    if not directory.is_dir():
        continue
    data.extend([{"label": directory.name, "img_path": path} 
                 for path in directory.iterdir()])

# Create DataFrame
dataset_df = pd.DataFrame(data)
classes = list(dataset_df["label"].unique())

# Function to encode images to base64 with resizing
def encode_image_to_base64(image_path: str | Path) -> str:
    """Load an image, resize it, and encode as base64 string."""
    image = Image.open(image_path)
    
    # Resize to half the original dimensions for memory efficiency
    new_size = (image.width // 2, image.height // 2)
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)
    
    encoded_string = (
        "data:image/jpeg;base64," 
        + base64.b64encode(buffer.read()).decode('utf-8')
    )
    
    return encoded_string

# Apply encoding to all images
dataset_df["img_b64"] = [
    encode_image_to_base64(img_path) 
    for img_path in dataset_df["img_path"]
]

# Split into training and test sets (80/20 split)
train_df, test_df = train_test_split(
    dataset_df,
    test_size=0.2,
    random_state=42,
    stratify=dataset_df["label"]
)

# Reset indices for clean DataFrames
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Clean up to save memory
del dataset_df

# Verify the data
print("Classes:", classes)
print("Training set size:", len(train_df))
print("Test set size:", len(test_df))
print("\nTraining data preview:")
print(train_df.head())
```

**Expected Output:**
```
Classes: ['School', 'Farmland', 'Airport', 'BaseballField', 'Resort', 'Viaduct', 'Forest', 'Beach', 'Parking', 'MediumResidential', 'Pond', 'Park', 'Port', 'Meadow', 'BareLand', 'Playground', 'SparseResidential', 'Desert', 'DenseResidential', 'Bridge', 'Square', 'River', 'StorageTanks', 'Commercial', 'Center', 'Stadium', 'Industrial', 'RailwayStation', 'Mountain', 'Church']
Training set size: 8000
Test set size: 2000

Training data preview:
   label                                       img_path                                          img_b64
0  Bridge  satellite_dataset/AID/Bridge/bridge_268.jpg  data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...
1  Mountain  satellite_dataset/AID/Mountain/mountain_220.jpg  data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...
2  Park  satellite_dataset/AID/Park/park_161.jpg  data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...
3  Farmland  satellite_dataset/AID/Farmland/farmland_173.jpg  data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...
4  Center  satellite_dataset/AID/Center/center_256.jpg  data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...
```

## Step 3: Visualize Sample Images

Let's create a helper function to display images from our dataset:

```python
from IPython.display import display, HTML

def display_image(dataset_df: pd.DataFrame, idx: int) -> None:
    """Display an image with its label from the DataFrame."""
    img_b64 = dataset_df["img_b64"].iloc[idx]
    label = dataset_df["label"].iloc[idx]
    display(HTML(f'<h2>{label}</h2><img src="{img_b64}" width="300">'))

# Display a sample image
display_image(train_df, 0)
```

## Next Steps

You now have a properly formatted dataset ready for fine-tuning. The next steps in this tutorial will cover:

1. **Baseline Evaluation**: Testing Pixtral's zero-shot performance on the test set
2. **Batch Inference**: Using Mistral's batch API to process multiple images efficiently
3. **Fine-tuning Setup**: Preparing training data in the correct format for Pixtral
4. **Model Training**: Initiating and monitoring the fine-tuning job
5. **Evaluation**: Comparing the fine-tuned model's performance against the baseline

**Note**: While smaller, specialized vision models might achieve comparable performance, this guide demonstrates the process of fine-tuning Mistral's VLM. More advanced applications could include interactive "speak with an image" features or image caption generation.

## References

- [Mistral Fine-tuning Documentation](https://docs.mistral.ai/capabilities/finetuning/)
- [Mistral Vision Capabilities](https://docs.mistral.ai/capabilities/vision/)