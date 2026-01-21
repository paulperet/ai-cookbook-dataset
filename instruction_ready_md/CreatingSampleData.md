# Guide: Creating an Image Dataset from Hugging Face

This guide walks you through creating a local image dataset by downloading images from URLs in a Hugging Face dataset. You'll filter out any rows with failed downloads and save the cleaned dataset as a CSV file.

## Prerequisites

Ensure you have the required Python libraries installed:

```bash
pip install pandas datasets requests pillow
```

## Step 1: Import Required Libraries

Begin by importing the necessary modules:

```python
import os
import pandas as pd
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO
```

## Step 2: Define the Image Download Function

Create a helper function to download and save images from URLs. This function returns `True` on success and `False` on failure, providing error feedback.

```python
def download_image(image_url, save_path):
    """
    Downloads an image from a URL and saves it locally.
    
    Args:
        image_url (str): URL of the image to download
        save_path (str): Local path to save the image
    
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad status codes
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
        return True
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        return False
```

## Step 3: Load and Prepare the Dataset

Replace `'Insert_Your_Dataset'` with your actual Hugging Face dataset identifier. Then load the dataset and convert it to a pandas DataFrame for easier manipulation.

```python
# Load dataset from Hugging Face
dataset = load_dataset('Insert_Your_Dataset')

# Convert to pandas DataFrame (assuming we use the 'train' split)
df = dataset['train'].to_pandas()
```

## Step 4: Create Directory Structure

Set up directories to store your dataset files and downloaded images:

```python
# Define base directory (customize 'DataSetName' as needed)
dataset_dir = './data/DataSetName'
images_dir = os.path.join(dataset_dir, 'images')
os.makedirs(images_dir, exist_ok=True)
```

## Step 5: Download Images and Filter the Dataset

Iterate through the DataFrame, attempt to download each image, and collect only the rows where downloads succeed. This step assumes your DataFrame has columns named `'imageurl'` (URL) and `'product_code'` (unique identifier).

```python
filtered_rows = []

for idx, row in df.iterrows():
    image_url = row['imageurl']
    image_name = f"{row['product_code']}.jpg"
    image_path = os.path.join(images_dir, image_name)
    
    if download_image(image_url, image_path):
        # Add local path to the row and keep it
        row['local_image_path'] = image_path
        filtered_rows.append(row)
```

## Step 6: Save the Filtered Dataset

Create a new DataFrame from the successfully downloaded rows and save it as a CSV file:

```python
# Create filtered DataFrame
filtered_df = pd.DataFrame(filtered_rows)

# Save to CSV
dataset_path = os.path.join(dataset_dir, 'Dataset.csv')
filtered_df.to_csv(dataset_path, index=False)

print(f"Dataset and images saved to {dataset_dir}")
```

## Complete Script

Here's the complete script for reference:

```python
import os
import pandas as pd
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO

def download_image(image_url, save_path):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
        return True
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        return False

# Load dataset
dataset = load_dataset('Insert_Your_Dataset')
df = dataset['train'].to_pandas()

# Create directories
dataset_dir = './data/DataSetName'
images_dir = os.path.join(dataset_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

# Filter rows with successful downloads
filtered_rows = []
for idx, row in df.iterrows():
    image_url = row['imageurl']
    image_name = f"{row['product_code']}.jpg"
    image_path = os.path.join(images_dir, image_name)
    if download_image(image_url, image_path):
        row['local_image_path'] = image_path
        filtered_rows.append(row)

# Save filtered dataset
filtered_df = pd.DataFrame(filtered_rows)
dataset_path = os.path.join(dataset_dir, 'Dataset.csv')
filtered_df.to_csv(dataset_path, index=False)

print(f"Dataset and images saved to {dataset_dir}")
```

## Next Steps

- Verify the downloaded images in the `./data/DataSetName/images/` directory
- Check the generated `Dataset.csv` file for the filtered data with local image paths
- Adapt column names (`'imageurl'`, `'product_code'`) to match your specific dataset schema

This approach ensures you have a clean, local dataset ready for machine learning tasks, with all images successfully downloaded and accessible.