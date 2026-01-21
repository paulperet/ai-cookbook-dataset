# Holistic Trace Analysis (HTA) Tutorial: Analyzing Distributed Training Performance

**Author:** [Anupam Bhatnagar](https://github.com/anupambhatnagar)

This tutorial demonstrates how to use Holistic Trace Analysis (HTA) to identify performance bottlenecks in distributed PyTorch training jobs by analyzing execution traces.

## Prerequisites & Installation

We recommend using a Conda environment. If you don't have Anaconda installed, follow the [official installation guide](https://docs.anaconda.com/anaconda/install/index.html).

### Step 1: Create and Activate a Conda Environment (Optional but Recommended)

```bash
# Create a new environment named 'hta_env'
conda create -n hta_env python=3.9 -y

# Activate the environment
conda activate hta_env
```

### Step 2: Install HTA

Install the HTA package via pip:

```bash
pip install HolisticTraceAnalysis
```

## Getting Started with Trace Analysis

First, launch a Jupyter notebook or Python script and import the necessary module. You'll need to point the analyzer to your trace directory.

```python
from hta.trace_analysis import TraceAnalysis

# Set this to the path containing your trace files
trace_dir = "/path/to/folder/with/traces"
analyzer = TraceAnalysis(trace_dir=trace_dir)
```

## Step 1: Analyze Temporal Breakdown

Understanding how GPUs spend their time is critical for optimization. The temporal breakdown categorizes time into:
- **Compute Time:** GPU is performing matrix multiplications or vector operations.
- **Non-Compute Time:** GPU is engaged in communication or memory events.
- **Idle Time:** GPU is not executing any kernels.

High training efficiency requires maximizing compute time and minimizing idle and non-compute time.

```python
# Generate a DataFrame with the temporal breakdown per rank
time_spent_df = analyzer.get_temporal_breakdown()
print(time_spent_df.head())
```

To visualize this data as a bar chart, set the `visualize` argument to `True`:

```python
time_spent_df = analyzer.get_temporal_breakdown(visualize=True)
```

## Step 2: Dive Deeper into Idle Time

A GPU is idle when no kernel is running. HTA categorizes idle time into three types to guide optimization:

1.  **Host Wait:** GPU stalls because the CPU isn't enqueuing kernels fast enough.
    *   *Optimization:* Examine slow CPU operators, increase batch size, or apply operator fusion.
2.  **Kernel Wait:** Brief overhead between launching consecutive kernels.
    *   *Optimization:* Use CUDA Graph optimizations.
3.  **Other Wait:** Idle time from synchronization (e.g., CUDA events) or insufficient trace information.

By default, the analysis runs for rank 0. Use the `ranks` argument to analyze other ranks.

```python
# Get idle time breakdown (returns a tuple of DataFrames)
idle_time_df = analyzer.get_idle_time_breakdown()

# To see summary statistics of idle intervals, set show_idle_interval_stats=True
idle_time_df = analyzer.get_idle_time_breakdown(show_idle_interval_stats=True)
```

**Tip:** By default, the visualization shows percentages. To display absolute time on the y-axis, set `visualize_pctg=False`.

## Step 3: Examine Kernel Breakdown

This feature breaks down GPU time by kernel type: Computation (COMP), Communication (COMM), and Memory (MEM).

```python
# Returns two DataFrames: kernel type metrics and detailed kernel metrics
kernel_type_metrics_df, kernel_metrics_df = analyzer.get_gpu_kernel_breakdown()
print(kernel_type_metrics_df.head())
```

The `kernel_metrics_df` contains duration statistics (count, min, max, average, sum) for each kernel per rank, which HTA uses to generate visualizations like:
*   Pie charts of the top kernels for each type per rank.
*   Bar graphs showing the average duration of top kernels across all ranks.

You can configure these charts:
*   `num_kernels`: Controls how many top kernels to display (default is 5).
*   `duration_ratio`: Sets the minimum percentage of total time a kernel must consume to be included.

```python
# Example: Show top 3 kernels that make up at least 1% of the time
kernel_type_metrics_df, kernel_metrics_df = analyzer.get_gpu_kernel_breakdown(
    num_kernels=3,
    duration_ratio=0.01
)
```

**Note for JupyterLab Users:** To render plots correctly, set `image_renderer="jupyterlab"` when calling `get_gpu_kernel_breakdown()`.

## Step 4: Measure Communication-Computation Overlap

In distributed training, efficient overlap between communication and computation is key to keeping GPUs busy. The overlap percentage is calculated as:
`(Time spent computing while communicating) / (Total communication time)`

A higher percentage is better.

```python
# Calculate overlap percentage for each rank
overlap_df = analyzer.get_comm_comp_overlap()
print(overlap_df)

# To generate a bar chart visualization
overlap_df = analyzer.get_comm_comp_overlap(visualize=True)
```

## Step 5: Utilize Augmented Counters

HTA can augment your trace files with two helpful counters:

1.  **Memory Bandwidth:** Tracks data transfer rates for H2D, D2H, and D2D copies (memcpy/memset).
2.  **Queue Length:** Shows the number of outstanding operations on each CUDA stream. A queue length of 1024+ can cause CPU stalls.

### Generate a New Trace File with Counters

By default, this processes rank 0 and creates a new file with the suffix `_with_counters`.

```python
analyzer.generate_trace_with_counters()

# To process multiple ranks (e.g., 0 and 1)
analyzer.generate_trace_with_counters(ranks=[0, 1])
```

### Analyze Counter Summaries and Time Series

```python
# Get summary statistics
mem_bw_summary = analyzer.get_memory_bw_summary()
queue_len_summary = analyzer.get_queue_length_summary()

# Get time-series data (returns a dict keyed by rank)
mem_bw_series = analyzer.get_memory_bw_time_series() # Default is rank 0
queue_len_series = analyzer.get_queue_length_time_series(ranks=[0, 1]) # Specify ranks
```

## Step 6: Analyze CUDA Kernel Launch Statistics

HTA links CPU scheduling events (e.g., `CudaLaunchKernel`) to their corresponding GPU kernels using correlation IDs. This analysis helps identify:
*   **Short GPU Kernels:** Kernels whose execution time is less than their CPU launch time.
*   **Runtime Event Outliers:** Excessively long CPU runtime events.
*   **Launch Delay Outliers:** Kernels with long scheduling delays.

```python
# Get launch statistics DataFrame
kernel_info_df = analyzer.get_cuda_kernel_launch_stats()
print(kernel_info_df.head())
```

You can customize the outlier detection thresholds:

```python
# Adjust cutoffs for runtime and launch delay outliers (values in microseconds)
kernel_info_df = analyzer.get_cuda_kernel_launch_stats(
    runtime_cutoff=100,   # Classify CPU runtime > 100μs as outlier
    launch_delay_cutoff=50 # Classify launch delay > 50μs as outlier
)
```

## Conclusion

You've now learned the core features of Holistic Trace Analysis for profiling distributed training jobs. HTA helps you pinpoint inefficiencies in GPU utilization, kernel execution, communication overlap, and kernel launch overhead.

To continue learning, explore the [Trace Diff tutorial](https://pytorch.org/tutorials/beginner/hta_trace_diff_tutorial.html) to compare traces and analyze performance regressions.