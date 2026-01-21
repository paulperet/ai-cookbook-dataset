# OpenAI Usage & Costs API Tutorial: Building a Custom Analytics Dashboard

## Overview

For most OpenAI users, the [default usage and cost dashboards](https://platform.openai.com/usage) provide sufficient insights. However, when you need more detailed data or want to build custom analytics, the Completions Usage and Costs APIs offer powerful programmatic access.

This tutorial guides you through retrieving, processing, and visualizing OpenAI usage data. You'll learn how to:

1. Set up API authentication and retrieve paginated data
2. Parse JSON responses into structured DataFrames
3. Visualize token usage trends over time
4. Analyze usage patterns by model and project
5. Retrieve and visualize cost data

## Prerequisites

Before starting, ensure you have:

- An OpenAI organization with admin access
- An Admin API key (available at https://platform.openai.com/settings/organization/admin-keys)
- Python 3.8+ installed

## Setup and Installation

First, install the required Python libraries:

```bash
pip install requests pandas numpy matplotlib
```

Then import the necessary modules:

```python
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import json
```

## Step 1: Configure API Authentication

Create a reusable function to handle paginated API requests. Replace `'PLACEHOLDER'` with your actual Admin API key (consider using environment variables for production):

```python
def get_data(url, params):
    """Retrieve paginated data from OpenAI APIs"""
    OPENAI_ADMIN_KEY = 'PLACEHOLDER'
    
    headers = {
        "Authorization": f"Bearer {OPENAI_ADMIN_KEY}",
        "Content-Type": "application/json",
    }
    
    all_data = []
    page_cursor = None
    
    while True:
        if page_cursor:
            params["page"] = page_cursor
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data_json = response.json()
            all_data.extend(data_json.get("data", []))
            
            page_cursor = data_json.get("next_page")
            if not page_cursor:
                break
        else:
            print(f"Error: {response.status_code}")
            break
    
    if all_data:
        print("Data retrieved successfully!")
    else:
        print("Issue: No data available to retrieve.")
    
    return all_data
```

## Step 2: Retrieve Completions Usage Data

Now let's fetch usage data for the last 30 days. The API supports various filtering options that you can customize:

```python
# Define the API endpoint
url = "https://api.openai.com/v1/organization/usage/completions"

# Calculate start time: 30 days ago from now
days_ago = 30
start_time = int(time.time()) - (days_ago * 24 * 60 * 60)

# Define parameters with all available options
params = {
    "start_time": start_time,  # Required: Start time (Unix seconds)
    # "end_time": end_time,  # Optional: End time (Unix seconds)
    "bucket_width": "1d",  # Optional: '1m', '1h', or '1d' (default '1d')
    # "project_ids": ["proj_example"],  # Optional: List of project IDs
    # "user_ids": ["user_example"],     # Optional: List of user IDs
    # "api_key_ids": ["key_example"],   # Optional: List of API key IDs
    # "models": ["o1-2024-12-17", "gpt-4o-2024-08-06", "gpt-4o-mini-2024-07-18"],
    # "batch": False,             # Optional: True for batch jobs, False for non-batch
    # "group_by": ["model"],     # Optional: Fields to group by
    "limit": 7,  # Optional: Number of buckets to return
}

usage_data = get_data(url, params)
```

## Step 3: Parse and Structure the Data

The API returns JSON data that we'll parse into a pandas DataFrame for easier analysis:

```python
# Initialize a list to hold parsed records
records = []

# Iterate through the data to extract bucketed data
for bucket in usage_data:
    start_time = bucket.get("start_time")
    end_time = bucket.get("end_time")
    for result in bucket.get("results", []):
        records.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "input_cached_tokens": result.get("input_cached_tokens", 0),
                "input_audio_tokens": result.get("input_audio_tokens", 0),
                "output_audio_tokens": result.get("output_audio_tokens", 0),
                "num_model_requests": result.get("num_model_requests", 0),
                "project_id": result.get("project_id"),
                "user_id": result.get("user_id"),
                "api_key_id": result.get("api_key_id"),
                "model": result.get("model"),
                "batch": result.get("batch"),
            }
        )

# Create a DataFrame from the records
df = pd.DataFrame(records)

# Convert Unix timestamps to datetime for readability
df["start_datetime"] = pd.to_datetime(df["start_time"], unit="s")
df["end_datetime"] = pd.to_datetime(df["end_time"], unit="s")

# Reorder columns for better readability
df = df[
    [
        "start_datetime",
        "end_datetime",
        "start_time",
        "end_time",
        "input_tokens",
        "output_tokens",
        "input_cached_tokens",
        "input_audio_tokens",
        "output_audio_tokens",
        "num_model_requests",
        "project_id",
        "user_id",
        "api_key_id",
        "model",
        "batch",
    ]
]

# Display the first few rows
df.head()
```

## Step 4: Visualize Token Usage Over Time

Create a bar chart to compare input and output token usage across time buckets:

```python
if not df.empty:
    plt.figure(figsize=(12, 6))
    
    # Create bar charts for input and output tokens
    width = 0.35  # width of the bars
    indices = range(len(df))
    
    plt.bar(indices, df["input_tokens"], width=width, label="Input Tokens", alpha=0.7)
    plt.bar(
        [i + width for i in indices],
        df["output_tokens"],
        width=width,
        label="Output Tokens",
        alpha=0.7,
    )
    
    # Set labels and ticks
    plt.xlabel("Time Bucket")
    plt.ylabel("Number of Tokens")
    plt.title("Daily Input vs Output Token Usage Last 30 Days")
    plt.xticks(
        [i + width / 2 for i in indices],
        [dt.strftime("%Y-%m-%d") for dt in df["start_datetime"]],
        rotation=45,
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No data available to plot.")
```

## Step 5: Analyze Usage by Model and Project

To get more granular insights, you can group data by multiple dimensions. Note that without the `group_by` parameter, fields like `project_id` and `model` will return as `null`:

```python
# Calculate start time: 30 days ago from now
days_ago = 30
start_time = int(time.time()) - (days_ago * 24 * 60 * 60)

# Define parameters with grouping by model and project_id
params = {
    "start_time": start_time,
    "bucket_width": "1d",
    "group_by": ["model", "project_id"],  # Group data by model and project_id
    "limit": 7,
}

# Retrieve grouped data
all_group_data = get_data(url, params)

# Parse the grouped data
records = []
for bucket in all_group_data:
    start_time = bucket.get("start_time")
    end_time = bucket.get("end_time")
    for result in bucket.get("results", []):
        records.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "input_tokens": result.get("input_tokens", 0),
                "output_tokens": result.get("output_tokens", 0),
                "input_cached_tokens": result.get("input_cached_tokens", 0),
                "input_audio_tokens": result.get("input_audio_tokens", 0),
                "output_audio_tokens": result.get("output_audio_tokens", 0),
                "num_model_requests": result.get("num_model_requests", 0),
                "project_id": result.get("project_id", "N/A"),
                "user_id": result.get("user_id", "N/A"),
                "api_key_id": result.get("api_key_id", "N/A"),
                "model": result.get("model", "N/A"),
                "batch": result.get("batch", "N/A"),
            }
        )

# Create and format the DataFrame
df = pd.DataFrame(records)
df["start_datetime"] = pd.to_datetime(df["start_time"], unit="s", errors="coerce")
df["end_datetime"] = pd.to_datetime(df["end_time"], unit="s", errors="coerce")

df = df[
    [
        "start_datetime",
        "end_datetime",
        "start_time",
        "end_time",
        "input_tokens",
        "output_tokens",
        "input_cached_tokens",
        "input_audio_tokens",
        "output_audio_tokens",
        "num_model_requests",
        "project_id",
        "user_id",
        "api_key_id",
        "model",
        "batch",
    ]
]

df.head()
```

## Step 6: Create a Stacked Bar Chart by Model and Project

Visualize how model requests are distributed across different projects:

```python
# Group data by model and project_id
grouped_by_model_project = (
    df.groupby(["model", "project_id"])
    .agg(
        {
            "num_model_requests": "sum",
        }
    )
    .reset_index()
)

# Determine unique models and project IDs
models = sorted(grouped_by_model_project["model"].unique())
project_ids = sorted(grouped_by_model_project["project_id"].unique())

# Create color mapping for projects
distinct_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
project_color_mapping = {
    pid: distinct_colors[i % len(distinct_colors)] for i, pid in enumerate(project_ids)
}

# Calculate total requests per project for legend
project_totals = (
    grouped_by_model_project.groupby("project_id")["num_model_requests"]
    .sum()
    .sort_values(ascending=False)
)

# Set up bar positions
n_models = len(models)
bar_width = 0.6
x = np.arange(n_models)

plt.figure(figsize=(12, 6))

# Plot stacked bars for each model
for model_idx, model in enumerate(models):
    model_data = grouped_by_model_project[grouped_by_model_project["model"] == model]
    bottom = 0
    
    # Stack segments for each project ID
    for _, row in model_data.iterrows():
        color = project_color_mapping[row["project_id"]]
        plt.bar(
            x[model_idx],
            row["num_model_requests"],
            width=bar_width,
            bottom=bottom,
            color=color,
        )
        bottom += row["num_model_requests"]

# Labeling and styling
plt.xlabel("Model")
plt.ylabel("Number of Model Requests")
plt.title("Total Model Requests by Model and Project ID Last 30 Days")
plt.xticks(x, models, rotation=45, ha="right")

# Create a sorted legend with totals
handles = [
    mpatches.Patch(color=project_color_mapping[pid], label=f"{pid} (Total: {total})")
    for pid, total in project_totals.items()
]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()
```

## Step 7: Create a Pie Chart of Model Request Distribution

Visualize the proportion of model requests by project:

```python
# Prepare data for pie chart
records = []
for bucket in all_group_data:
    for result in bucket.get("results", []):
        records.append(
            {
                "project_id": result.get("project_id", "N/A"),
                "num_model_requests": result.get("num_model_requests", 0),
            }
        )

df = pd.DataFrame(records)
grouped_by_project = (
    df.groupby("project_id").agg({"num_model_requests": "sum"}).reset_index()
)

if not grouped_by_project.empty:
    # Filter out rows with zero requests
    filtered_grouped_by_project = grouped_by_project[
        grouped_by_project["num_model_requests"] > 0
    ]
    
    total_requests = filtered_grouped_by_project["num_model_requests"].sum()
    
    if total_requests > 0:
        # Calculate percentages
        filtered_grouped_by_project["percentage"] = (
            filtered_grouped_by_project["num_model_requests"] / total_requests
        ) * 100
        
        # Separate "Other" projects (below 5%)
        other_projects = filtered_grouped_by_project[
            filtered_grouped_by_project["percentage"] < 5
        ]
        main_projects = filtered_grouped_by_project[
            filtered_grouped_by_project["percentage"] >= 5
        ]
        
        # Sum up "Other" projects
        if not other_projects.empty:
            other_row = pd.DataFrame(
                {
                    "project_id": ["Other"],
                    "num_model_requests": [other_projects["num_model_requests"].sum()],
                    "percentage": [other_projects["percentage"].sum()],
                }
            )
            filtered_grouped_by_project = pd.concat(
                [main_projects, other_row], ignore_index=True
            )
        
        # Sort by number of requests
        filtered_grouped_by_project = filtered_grouped_by_project.sort_values(
            by="num_model_requests", ascending=False
        )
        
        # Create main pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(
            filtered_grouped_by_project["num_model_requests"],
            labels=filtered_grouped_by_project["project_id"],
            autopct=lambda p: f"{p:.1f}%\n({int(p * total_requests / 100):,})",
            startangle=140,
            textprops={"fontsize": 10},
        )
        plt.title("Distribution of Model Requests by Project ID", fontsize=14)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()
        
        # Create secondary pie chart for "Other" projects if needed
        if not other_projects.empty:
            other_total_requests = other_projects["num_model_requests"].sum()
            
            plt.figure(figsize=(10, 8))
            plt.pie(
                other_projects["num_model_requests"],
                labels=other_projects["project_id"],
                autopct=lambda p: f"{p:.1f}%\n({int(p * other_total_requests / 100):,})",
                startangle=140,
                textprops={"fontsize": 10},
            )
            plt.title('Breakdown of "Other" Projects by Model Requests', fontsize=14)
            plt.axis("equal")
            plt.tight_layout()
            plt.show()
    else:
        print("Total model requests is zero. Pie chart will not be rendered.")
else:
    print("No grouped data available for pie chart.")
```

## Step 8: Retrieve and Analyze Cost Data

Finally, let's work with the Costs API to retrieve and visualize cost data:

```python
# Calculate start time: 30 days ago from now
days_ago = 30
start_time = int(time.time()) - (days_ago * 24 * 60 * 60)

# Define the Costs API endpoint
costs_url = "https://api.openai.com/v1/organization/costs"

costs_params = {
    "start_time": start_time,  # Required: Start time (Unix seconds)
    "bucket_width": "1d",  # Optional: Currently only '1d' is supported
    "limit": 7,
}

# Retrieve cost data
costs_data = get_data(costs_url, costs_params)

# Parse and visualize cost data (similar structure to usage data)
# You can adapt the parsing and visualization code from previous steps
```

## Conclusion

You've now built a comprehensive analytics pipeline for OpenAI usage and cost data. Key takeaways:

1. **API Flexibility**: The OpenAI APIs support extensive filtering and grouping options
2. **Pagination Handling**: Always implement pagination logic for complete data retrieval
3. **Data Visualization**: Combine different chart types (bar, stacked bar, pie) for comprehensive insights
4. **Custom Dashboards**: You can extend this foundation to build custom monitoring dashboards

For production use, consider:
- Storing API keys securely in environment variables
- Implementing error handling and retry logic
- Caching API responses to reduce load
- Scheduling regular data collection for trend analysis

This tutorial provides a foundation you can adapt to your specific monitoring and analytics needs.