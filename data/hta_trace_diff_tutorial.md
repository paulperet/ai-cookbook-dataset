# Holistic Trace Analysis (HTA) Guide: Comparing PyTorch Traces with TraceDiff

## Introduction
When optimizing PyTorch code, developers often need to understand how changes affect runtime behavior. The Holistic Trace Analysis (HTA) library provides a `TraceDiff` class to compare two sets of execution traces—typically a *control* (baseline) trace and a *test* (modified) trace. This guide walks you through using HTA's trace comparison feature to identify added/removed operators and kernels, analyze frequency changes, and visualize differences.

## Prerequisites
Ensure you have HTA installed. If not, install it via pip:

```bash
pip install hta
```

## Step 1: Import and Initialize TraceDiff
First, import the `TraceDiff` class and create an instance. You'll need paths to your control and test trace directories.

```python
from hta.trace_diff import TraceDiff

# Paths to your trace directories
control_trace_dir = "./path/to/control_traces"
test_trace_dir = "./path/to/test_traces"

# Initialize TraceDiff
trace_diff = TraceDiff(control_trace_dir, test_trace_dir)
```

## Step 2: Compare Traces
Use the `compare_traces()` method to compute differences in frequency and total duration for CPU operators and GPU kernels between the two trace sets.

```python
# Compare traces
comparison_df = trace_diff.compare_traces()
print(comparison_df.head())
```

The output DataFrame includes columns for:
- `control_counts` / `test_counts`: Frequency of each operator/kernel.
- `control_duration` / `test_duration`: Cumulative duration.
- `diff_counts` / `diff_duration`: Differences between test and control.

## Step 3: Analyze Differences in Operators and Kernels
The `ops_diff()` method categorizes changes into five groups: **added**, **deleted**, **increased**, **decreased**, and **unchanged**.

```python
# Get categorized differences
diff_categories = trace_diff.ops_diff(comparison_df)

# Example: View operators added in the test trace
print("Added operators:", diff_categories["added"])
```

This helps pinpoint exactly which operators/kernels were affected by your code changes.

## Step 4: Visualize Frequency Changes
To visualize the top operators with increased frequency, sort the comparison DataFrame and use `visualize_counts_diff()`.

```python
# Get top 10 operators by frequency increase
top_increases = comparison_df.sort_values(by="diff_counts", ascending=False).head(10)

# Visualize
trace_diff.visualize_counts_diff(top_increases)
```

The chart will show operators with the largest frequency gains, helping identify potential performance regressions or optimizations.

## Step 5: Visualize Duration Changes
Similarly, visualize operators with the largest duration changes. Filter out "ProfilerStep" to focus on meaningful operators.

```python
# Get top 10 operators by duration increase (excluding ProfilerStep)
duration_changes = comparison_df.sort_values(by="diff_duration", ascending=False)
filtered_changes = duration_changes.loc[~duration_changes.index.str.startswith("ProfilerStep")].head(10)

# Visualize
trace_diff.visualize_duration_diff(filtered_changes)
```

This highlights operators whose cumulative execution time changed most significantly.

## Conclusion
HTA's `TraceDiff` class provides a powerful way to compare PyTorch traces and understand the impact of code changes. By following these steps, you can:
- Quantify differences in operator/kernel frequency and duration.
- Categorize changes as added, deleted, increased, decreased, or unchanged.
- Visualize key differences to guide optimization efforts.

For a complete example, refer to the [trace_diff_demo notebook](https://github.com/facebookresearch/HolisticTraceAnalysis/blob/main/examples/trace_diff_demo.ipynb) in the HTA repository.