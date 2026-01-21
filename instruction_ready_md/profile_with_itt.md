# Profiling PyTorch Workloads with the Instrumentation and Tracing Technology (ITT) API

This guide demonstrates how to use the Instrumentation and Tracing Technology (ITT) API integrated into PyTorch to profile your models and visualize their execution in Intel® VTune™ Profiler. You will learn how to label custom code regions and interpret the profiling timeline to identify performance bottlenecks.

## Prerequisites

Before you begin, ensure you have the following installed:

*   **PyTorch 1.13 or later**: The ITT API is integrated into PyTorch from version 1.13 onwards. Follow the [official installation instructions](https://pytorch.org/get-started/locally/).
*   **Intel® VTune™ Profiler**: This is the performance analysis tool used to visualize the ITT traces. You can find more information and a Getting Started guide on the [Intel website](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html).

## Understanding the Tools

### Intel® VTune™ Profiler
Intel® VTune™ Profiler is a powerful performance analysis tool for serial and multithreaded applications. It provides a rich set of metrics to help you understand how your application executes on Intel hardware, making it easier to locate performance bottlenecks.

### The ITT API in PyTorch
The Instrumentation and Tracing Technology (ITT) API allows your application to generate and control trace data during execution. The key advantage is the ability to label the time span of individual PyTorch operators and custom code regions directly on the VTune Profiler GUI. This makes it straightforward to pinpoint which operator or section of code is behaving unexpectedly.

> **Note**: The ITT API is integrated directly into PyTorch. You do not need to call the original C/C++ APIs; you only need to use the Python APIs provided by PyTorch, as documented in the [PyTorch Profiler documentation](https://pytorch.org/docs/stable/profiler.html#intel-instrumentation-and-tracing-technology-apis).

PyTorch provides two ways to use the ITT feature:
1.  **Implicit Invocation**: By default, all PyTorch operators registered via the standard mechanism are automatically labeled when ITT is enabled.
2.  **Explicit Invocation**: You can use specific PyTorch ITT APIs to manually label any custom region of your code for finer-grained analysis.

## Step 1: Prepare Your Script for Profiling

To enable explicit ITT labeling, wrap the code you want to profile within a `torch.autograd.profiler.emit_itt()` context manager.

Create a Python script (`sample.py`) with the following content. This script defines a simple model and runs inference for three iterations, labeling each iteration explicitly.

```python
# sample.py
import torch
import torch.nn as nn

class ITTSample(nn.Module):
    def __init__(self):
        super(ITTSample, self).__init__()
        self.conv = nn.Conv2d(3, 5, 3)
        self.linear = nn.Linear(292820, 1000)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

def main():
    m = ITTSample()
    # Uncomment the following line for XPU (GPU) profiling
    # m = m.to("xpu")

    x = torch.rand(10, 3, 244, 244)
    # Uncomment the following line for XPU (GPU) profiling
    # x = x.to("xpu")

    with torch.autograd.profiler.emit_itt():
        for i in range(3):
            # Method 1: Label a region using push/pop
            # torch.profiler.itt.range_push(f'iteration_{i}')
            # m(x)
            # torch.profiler.itt.range_pop()

            # Method 2: Label a region using a context manager (recommended)
            with torch.profiler.itt.range(f'iteration_{i}'):
                m(x)

if __name__ == '__main__':
    main()
```

## Step 2: Create a Launch Script for VTune Profiler

For ease of profiling, it's recommended to wrap your environment setup and script execution into a bash script. This script will be the target application for VTune Profiler.

Create a bash script named `launch.sh`:

```bash
#!/bin/bash

# Retrieve the directory containing this script and sample.py
BASEFOLDER=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Activate your Python environment here (e.g., conda, venv)
# Example: source /path/to/your/venv/bin/activate

cd ${BASEFOLDER}
python sample.py
```

Make the script executable:
```bash
chmod +x launch.sh
```

## Step 3: Configure and Run Profiling in VTune Profiler

### Launch VTune Profiler
Start the VTune Profiler GUI. You can launch it from your application menu or via command line. For remote or web-based profiling, you can use the VTune Profiler Web Server.

### Create a New Analysis
1.  Click the blue **Configure Analysis...** button.
2.  In the configuration window, you will see three main sections: **WHERE**, **WHAT**, and **HOW**.

### Configure the Analysis Settings
*   **WHERE**: Select the local machine or a remote target for profiling.
*   **WHAT**: Set the application to profile. Point to your `bash` executable and provide the full path to your `launch.sh` script as the argument. For example:
    *   Application: `/bin/bash`
    *   Application parameters: `/full/path/to/launch.sh`
*   **HOW**: Choose the analysis type.
    *   For **CPU profiling**, select **Hotspots**.
    *   For **XPU (GPU) profiling**, select **GPU Offload**.

Click **Start** to begin the profiling run. VTune will execute your script and collect performance data.

## Step 4: Interpret the Profiling Results

After the analysis completes, open the **Platform** tab to view the timeline.

### CPU Profiling Timeline
The timeline displays:
*   The main Python thread at the top.
*   Individual OpenMP threads below it.
*   Labeled regions appear in the main thread row.
    *   Operators prefixed with `aten::` are labeled implicitly by PyTorch's ITT integration.
    *   Regions labeled `iteration_N` are from your explicit `range()` calls.
    *   Additional labels (e.g., `convolution`, `reorder`) may come from libraries like Intel oneDNN.

The colored portions in each thread row represent CPU usage. This visualization helps you assess:
*   CPU core utilization per thread.
*   Load balance across threads.
*   Synchronization of OpenMP threads.

### XPU (GPU) Profiling Timeline
For GPU profiles, the timeline also shows:
*   The GPU Computing Queue at the top.
*   Various XPU kernels dispatched into the queue, alongside the Python thread activity.

## Summary

You have successfully learned how to:
1.  Instrument a PyTorch script using the integrated ITT APIs to label custom code regions.
2.  Configure and run a profiling session in Intel VTune Profiler for both CPU and XPU targets.
3.  Navigate to the Platform timeline to visualize the execution flow of PyTorch operators and your custom labels.

This workflow enables you to move from observing a performance issue to precisely identifying the responsible operators or code sections, which is the first critical step in optimization. For deeper analysis, explore the other tabs and metrics provided by VTune Profiler, as detailed in the [Intel VTune Profiler User Guide](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/analyze-performance.html).