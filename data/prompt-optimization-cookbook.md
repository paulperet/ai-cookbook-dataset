# Guide: Migrating and Optimizing Prompts for GPT-5 using the Prompt Optimizer

## Introduction
GPT-5 represents a significant advancement in model capabilities, particularly for agentic tasks, coding, and steerability. To help you leverage these improvements, OpenAI provides the **GPT-5 Prompt Optimizer**—a tool designed to refine your prompts according to best practices and remove common failure modes.

This guide walks you through using the Prompt Optimizer to migrate and improve a prompt for a coding task, demonstrating measurable performance gains.

## Prerequisites
Ensure you have:
1. An OpenAI API key set as the environment variable `OPENAI_API_KEY`.
2. Access to the GPT-5 model.

### Setup
First, verify your API key and install the required packages.

```python
import os

# Check for the required environment variable
required = ('OPENAI_API_KEY',)
missing = [k for k in required if not os.getenv(k)]
if missing:
    print('Missing environment variable: ' + ', '.join(missing) + '. Please set them before running the workflow.')
else:
    print('OPENAI_API_KEY is set!')
```

```bash
pip install -r requirements.txt --quiet
```

## Part 1: The Coding Task - Streaming Top-K Frequent Words
We will optimize a prompt for a coding task where the model must generate a Python script to compute the exact Top-K most frequent tokens from a large text stream. The evaluation criteria are strict:
- **Compilation/Execution Success:** Over 30 runs.
- **Average Runtime & Peak Memory:** For successful runs.
- **Exactness:** Output must match the ground-truth Top-K with specific tie-breaking (by count descending, then token ascending).

### 1.1 The Baseline Prompt
Our starting point is a typical prompt with several contradictions and ambiguities.

```python
baseline_prompt = """
Write Python to solve the task on a MacBook Pro (M4 Max). Keep it fast and lightweight.

- Prefer the standard library; use external packages if they make things simpler.
- Stream input in one pass to keep memory low; reread or cache if that makes the solution clearer.
- Aim for exact results; approximate methods are fine when they don't change the outcome in practice.
- Avoid global state; expose a convenient global like top_k so it's easy to check.
- Keep comments minimal; add brief explanations where helpful.
- Sort results in a natural, human-friendly way; follow strict tie rules when applicable.

Output only a single self-contained Python script inside one Python code block, with all imports, ready to run.
"""
```

**Why This Prompt is Problematic:**
- **Contradictory Constraints:** Encourages the standard library but allows external packages, leading to non-portable solutions.
- **Ambiguous Memory Guidance:** Suggests single-pass streaming but permits rereading/caching, which can defeat memory constraints.
- **Vague Correctness:** Allows approximate methods "when they don't change the outcome," a judgment call the model cannot reliably make.
- **Mixed Interface:** Advises against global state while suggesting a global variable (`top_k`), confusing the function's contract.
- **Unclear Sorting:** Requests "natural, human-friendly" sorting alongside "strict tie rules," which may conflict.

These soft constraints introduce variability, causing the model to produce different solution families across runs, impacting correctness and performance.

### 1.2 Generating and Evaluating Baseline Scripts
We will generate 30 scripts using the baseline prompt and evaluate them.

```python
from scripts.gen_baseline import generate_baseline_topk

MODEL = "gpt-5"
N_RUNS = 30
CONCURRENCY = 10
OUTPUT_DIR = "results_topk_baseline"

USER_PROMPT = """
Task:
Given globals text (str) and k (int), produce the Top-K most frequent tokens.

Tokenization:
- Case-insensitive tokenization using an ASCII regex; produce lowercase tokens. Whole-string lowercasing is not required.
- Tokens are ASCII [a-z0-9]+ sequences; treat all other characters as separators.

Output:
- Define top_k as a list of (token, count) tuples.
- Sort by count desc, then token asc.
- Length = min(k, number of unique tokens).

Notes:
- Run as-is with the provided globals; no file or network I/O.
"""

generate_baseline_topk(
    model=MODEL,
    n_runs=N_RUNS,
    concurrency=CONCURRENCY,
    output_dir=OUTPUT_DIR,
    dev_prompt=baseline_prompt,
    user_prompt=USER_PROMPT,
)
```

After generation, evaluate the scripts:

```python
from scripts.topk_eval import evaluate_folder

evaluate_folder(
    folder_path="results_topk_baseline",
    k=500,
    scale_tokens=5_000_000,
    csv_path="run_results_topk_baseline.csv",
)
```

## Part 2: Optimizing the Prompt
Now, we'll use the **GPT-5 Prompt Optimizer** to refine our prompt.

### 2.1 Using the Optimizer
1. Navigate to the [OpenAI Optimize Playground](https://platform.openai.com/chat/edit?optimize=true).
2. Paste the baseline prompt into the "Developer Message" section.
3. Click **Optimize**. The tool will analyze the prompt and suggest improvements based on best practices for the target model and task.
4. You can provide specific feedback (e.g., "Enforce single-pass streaming") and re-optimize iteratively.
5. Once satisfied, save the optimized prompt as a **Prompt Object** for version management and reuse.

### 2.2 The Optimized Prompt
Here is the resulting optimized prompt after refinement.

```python
optimized_prompt = """
# Objective
Generate a single, self-contained Python script that exactly solves the specified task on a MacBook Pro (M4 Max).

# Hard requirements
- Use only Python stdlib. No approximate algorithms.
- Tokenization: ASCII [a-z0-9]+ on the original text; match case-insensitively and lowercase tokens individually. Do NOT call text.lower() on the full string.
- Exact Top‑K semantics: sort by count desc, then token asc. No reliance on Counter.most_common tie behavior.
- Define `top_k` as a list of (token, count) tuples with length = min(k, number of unique tokens).
- When globals `text` (str) and `k` (int) exist, do not reassign them; set `top_k` from those globals. If you include a `__main__` demo, guard it to run only when globals are absent.
- No file I/O, stdin, or network access, except optionally printing `top_k` as the last line.

# Performance & memory constraints
- Do NOT materialize the entire token stream or any large intermediate list.
- Do NOT sort all unique (token, count) items unless k >= 0.3 * number_of_unique_tokens.
- When k < number_of_unique_tokens, compute Top‑K using a bounded min‑heap of size k over counts.items(), maintaining the correct tie-break (count desc, then token asc).
- Target peak additional memory beyond the counts dict to O(k). Avoid creating `items = sorted(counts.items(), ...)` for large unique sets.

# Guidance
- Build counts via a generator over re.finditer with re.ASCII | re.IGNORECASE; lowercase each matched token before counting.
- Prefer heapq.nsmallest(k, cnt.items(), key=lambda kv: (-kv[1], kv[0])) for exact selection without full sort; avoid heapq.nlargest.
- Do NOT wrap tokens in custom comparator classes (e.g., reverse-lex __lt__) or rely on tuple tricks for heap ordering.
- Keep comments minimal; include a brief complexity note (time and space).

# Output format
- Output only one Python code block; no text outside the block.

# Examples
```python
import re, heapq
from collections import Counter
from typing import List, Tuple, Iterable

_TOKEN = re.compile(r"[a-z0-9]+", flags=re.ASCII | re.IGNORECASE)

def _tokens(s: str) -> Iterable[str]:
    # Case-insensitive match; lowercase per token to avoid copying the whole string
    for m in _TOKEN.finditer(s):
        yield m.group(0).lower()

def top_k_tokens(text: str, k: int) -> List[Tuple[str, int]]:
    if k <= 0:
        return []
    cnt = Counter(_tokens(text))
    u = len(cnt)
    key = lambda kv: (-kv[1], kv[0])
    if k >= u:
        return sorted(cnt.items(), key=key)
    # Exact selection with bounded memory
    return heapq.nsmallest(k, cnt.items(), key=key)

# Compute from provided globals when available; demo only if missing and running as main
try:
    text; k  # type: ignore[name-defined]
except NameError:
    if __name__ == "__main__":
        demo_text = "A a b b b c1 C1 c1 -- d! d? e"
        demo_k = 3
        top_k = top_k_tokens(demo_text, demo_k)
        print(top_k)
else:
    top_k = top_k_tokens(text, k)  # type: ignore[name-defined]
# Complexity: counting O(N tokens), selection O(U log k) via heapq.nsmallest; extra space O(U + k)
```
"""
```

**Key Improvements:**
- **Clear, Unambiguous Requirements:** Hard constraints replace soft suggestions.
- **Specific Algorithmic Guidance:** Directs the model to use a bounded min-heap for memory efficiency.
- **Precise Tokenization Rules:** Eliminates ambiguity about case handling.
- **Strict Output Format:** Ensures a single, runnable code block.

### 2.3 Generating and Evaluating Optimized Scripts
Generate 30 scripts with the optimized prompt.

```python
from scripts.gen_optimized import generate_optimized_topk

MODEL = "gpt-5"
N_RUNS = 30
CONCURRENCY = 10
OUTPUT_DIR = "results_topk_optimized"

generate_optimized_topk(
    model=MODEL,
    n_runs=N_RUNS,
    concurrency=CONCURRENCY,
    output_dir=OUTPUT_DIR,
    dev_prompt=optimized_prompt,
    user_prompt=USER_PROMPT,  # Same USER_PROMPT as before
)
```

Evaluate the new scripts.

```python
evaluate_folder(
    folder_path="results_topk_optimized",
    k=500,
    scale_tokens=5_000_000,
    csv_path="run_results_topk_optimized.csv",
)
```

## Part 3: Qualitative Evaluation with LLM-as-a-Judge
Beyond quantitative metrics, we can assess code quality and task adherence using an LLM judge.

```python
from scripts.llm_judge import judge_folder

# Judge baseline results
judge_folder(
    results_dir="results_topk_baseline",
    out_dir=None,  # Auto-maps to results_llm_as_judge_baseline
    model="gpt-5",
    system_prompt_path="llm_as_judge.txt",
    task_text=None,
    concurrency=6,
)

# Judge optimized results
judge_folder(
    results_dir="results_topk_optimized",
    out_dir=None,  # Auto-maps to results_llm_as_judge_optimized
    model="gpt-5",
    system_prompt_path="llm_as_judge.txt",
    task_text=None,
    concurrency=6,
)
```

## Part 4: Results Summary
Finally, we compile and compare the results from both the quantitative evaluation and the LLM judge.

```python
from pathlib import Path
import importlib
import scripts.results_summarizer as rs
from IPython.display import Markdown, display

importlib.reload(rs)

# Render comparison charts
fig = rs.render_charts(
    quant_baseline=Path("results_topk_baseline")/"run_results_topk_baseline.csv",
    quant_optimized=Path("results_topk_optimized")/"run_results_topk_optimized.csv",
    judge_baseline=Path("results_llm_as_judge_baseline")/"judgement_summary.csv",
    judge_optimized=Path("results_llm_as_judge_optimized")/"judgement_summary.csv",
    auto_display=True,
    close_after=True,
)

# Build a markdown summary
md = rs.build_markdown_summary(
    quant_baseline=Path("results_topk_baseline")/"run_results_topk_baseline.csv",
    quant_optimized=Path("results_topk_optimized")/"run_results_topk_optimized.csv",
    judge_baseline=Path("results_llm_as_judge_baseline")/"judgement_summary.csv",
    judge_optimized=Path("results_llm_as_judge_optimized")/"judgement_summary.csv",
)

display(Markdown(md))
```

### Prompt Optimization Results - Coding Tasks

| Metric                      | Baseline | Optimized | Δ (Opt − Base) |
|----------------------------|---------:|----------:|---------------:|
| Avg Time (s)                |    7.906 |     6.977 |        -0.929 |
| Peak Memory (KB)            |   3626.3 |     577.5 |       -3048.8 |
| Exact (%)                   |    100.0 |     100.0 |           0.0 |
| Sorted (%)                  |    [Value]|    [Value]|         [Δ] |

## Conclusion
Using the GPT-5 Prompt Optimizer, we transformed a vague, contradictory prompt into a precise, high-performance specification. The optimized prompt yielded:
- **Faster Execution:** ~12% reduction in average runtime.
- **Significantly Lower Memory Usage:** ~84% reduction in peak memory.
- **Maintained 100% Exactness.**

This demonstrates the tangible benefits of prompt optimization for complex tasks like coding. Remember, prompting is iterative—use the Optimizer to refine your prompts, save them as reusable objects, and continuously test to achieve the best results for your specific use case.