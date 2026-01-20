# GPT-5 Prompt Migration and Improvement using the new prompt optimizer

The GPT-5 Family of models are the smartest models we’ve released to date, representing a step change in the models’ capabilities across the board. GPT-5 is particularly specialized in agentic task performance, coding, and steerability, making it a great fit for everyone from curious users to advanced researchers.

GPT-5 will benefit from all the traditional [prompting best practices](https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide), but to make optimizations and migrations easier, we are introducing the **[GPT-5  Prompt Optimizer](https://platform.openai.com/chat/edit?optimize=true)** in our Playground to help users get started on **improving existing prompts** and **migrating prompts** for GPT-5 and other OpenAI models.

In this cookbook we will show you how to use the Prompt Optimzer to get spun up quickly to solve your tasks with GPT-5, while demonstrating how prompt optimize can have measurable improvements.

### Migrating and Optimizing Prompts

Crafting effective prompts is a critical skill when working with LLMs. The goal of the Prompt Optimizer is to give your prompt the best practices and formatting most effective for our models. The Optimizer also removes common prompting failure modes such as:

•   Contradictions in the prompt instructions
•	Missing or unclear format specifications
•	Inconsistencies between the prompt and few-shot examples

Along with tuning the prompt for the target model, the Optimizer is cognizant of the specific task you are trying to accomplish and can apply crucial practices to boost performance in Agentic Workflows, Coding and Multi-Modality. Let's walk through some before-and-afters to see where prompt optimization shines.

> Remember that prompting is not a one-size-fits-all experience, so we recommend running thorough experiments and iterating to find the best solution for your problem.

> Ensure you have set up your OpenAI API Key set as `OPENAI_API_KEY` and have access to GPT-5

```python
import os

required = ('OPENAI_API_KEY',)
missing = [k for k in required if not os.getenv(k)]
print('OPENAI_API_KEY is set!' if not missing else 'Missing environment variable: ' + ', '.join(missing) + '. Please set them before running the workflow.')
```

```python
## Let's install our required packages
%pip install -r requirements.txt --quiet
```

----------------

### Coding and Analytics: Streaming Top‑K Frequent Words

We start with a task in a field that model has seen significant improvements: Coding and Analytics. We will ask the model to generate a Python script that computes the exact Top‑K most frequent tokens from a large text stream using a specific tokenization spec. Tasks like these are highly sensitive to poor prompting as they can push the model toward the wrong algorithms and approaches (approximate sketches vs multi‑pass/disk‑backed exact solutions), dramatically affecting accuracy and runtime.

For this task, we will evaluate:
1. Compilation/Execution success over 30 runs
2. Average runtime (successful runs)
3. Average peak memory (successful runs)
4. Exactness: output matches ground‑truth Top‑K with tie‑break: by count desc, then token asc

Note: Evaluated on an M4 Max MacBook Pro; adjust constraints if needed.

### Our Baseline Prompt
For our example, let's look at a typical starting prompt with some minor **contradictions in the prompt**, and **ambiguous or underspecified instructions**. Contradictions in instructions often reduce performance and increase latency, especially in reasoning models like GPT-5, and ambiguous instructions can cause unwanted behaviors.

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

This baseline prompt is something that you could expect from asking ChatGPT to write you a prompt, or talking to a friend who is knowledgeable about coding but not particularly invested in your specific use case. Our baseline prompt is intentionally shorter and friendlier, but it hides mixed signals that can push the model into inconsistent solution families.

First, we say to prefer the standard library, then immediately allow external packages “if they make things simpler.” That soft permission can nudge the model toward non‑portable dependencies or heavier imports that change performance and even execution success across environments.

Next, we encourage single‑pass streaming to keep memory low, but we also say it’s fine to reread or cache “if that makes the solution clearer.” That ambiguity opens the door to multi‑pass designs or in‑memory caches that defeat the original streaming constraint and can alter runtime and memory profiles.

We also ask for exact results while permitting approximate methods “when they don’t change the outcome in practice.” This is a judgment call the model can’t reliably verify. It may introduce sketches or heuristics that subtly shift counts near the Top‑K boundary, producing results that look right but fail strict evaluation.

We advise avoiding global state, yet suggest exposing a convenient global like `top_k`. That mixes interface contracts: is the function supposed to return data, or should callers read globals? Models may implement both, causing side effects that complicate evaluation and reproducibility.

Documentation guidance is similarly split: “keep comments minimal” but “add brief explanations.” Depending on how the model interprets this, you can get under‑explained code or prose interleaved with logic, which sometimes leaks outside the required output format.

Finally, we ask for “natural, human‑friendly” sorting while also mentioning strict tie rules. These aren’t always the same. The model might pick convenience ordering (e.g., `Counter.most_common`) and drift from the evaluator’s canonical `(-count, token)` sort, especially on ties—leading to subtle correctness misses.

**Why this matters**: the softened constraints make the prompt feel easy to satisfy, but they create forks in the road. The model may pick different branches across runs—stdlib vs external deps, one‑pass vs reread/cache, exact vs approximate—yielding variability in correctness, latency, and memory.

**Our evaluator remains strict**: fixed tokenization `[a-z0-9]+` on lowercased text and deterministic ordering by `(-count, token)`. Any divergence here will penalize exactness even if the rest of the solution looks reasonable.

### Let's see how it performs: Generating 30 code scripts with the baseline prompt

Using the OpenAI Responses API we'll invoke the model 30 times with our baseline prompt and save each response as a Python file in the `results_topk_baseline`. This may take some time.

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

### Evaluate Generated Scripts - Baseline Prompt

We then benchmark every script in ``results_topk_baseline`` On larger datasets this evaluation is intentionally heavy and can take several minutes.

```python
from scripts.topk_eval import evaluate_folder

evaluate_folder(
    folder_path="results_topk_baseline",
    k=500,
    scale_tokens=5_000_000,
    csv_path="run_results_topk_baseline.csv",
)
```

### Optimizing our Prompt

Now let's use the prompt optimization tool in the console to improve our prompt and then review the results. We can start by going to the [OpenAI Optimize Playground](https://platform.openai.com/chat/edit?optimize=true), and pasting our existing prompt in the Developer Message section.

From there press the **Optimize** button. This will open the optimization panel. At this stage, you can either provide specific edits you'd like to see reflected in the prompt or simply press **Optimize** to have it refined according to best practices for the target model and task. To start let's do just this.

Once it's completed you'll see the result of the prompt optimization. In our example below you'll see many changes were made to the prompt. It will also give you snippets of what it changed and why the change was made. You can interact with these by opening the comments up or using the inline reviewer mode.

We'll add an additional change we'd like which include:

- Enforcing the single-pass streaming

This is easy using the iterative process of the Prompt Optimizer.

Once we are happy with the optimized version of our prompt, we can save it as a [Prompt Object](#https://platform.openai.com/docs/guides/prompt-engineering#reusable-prompts) using a button on the top right of the optimizer. We can use this object within our API Calls which can help with future iteration, version management, and reusability across different applications.

###  Let's see how it performs: Evaluating our improved prompt

For visibility we will provide our new optimized prompt here, but you can also pass the ``prompt_id`` and ``version``. Let's start by writing out our optimized prompt.

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

### Generating 30 code scripts with the Optimized prompt

```python
from scripts.gen_optimized import generate_optimized_topk

MODEL = "gpt-5"
N_RUNS = 30
CONCURRENCY = 10
OUTPUT_DIR = "results_topk_optimized"

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

generate_optimized_topk(
    model=MODEL,
    n_runs=N_RUNS,
    concurrency=CONCURRENCY,
    output_dir=OUTPUT_DIR,
    dev_prompt=optimized_prompt,
    user_prompt=USER_PROMPT,
)
```

### Evaluate Generated Scripts - Optimized Prompt

We run the same evaluation as above, but now with our optimized prompt to see if there were any improvements

```python
from scripts.topk_eval import evaluate_folder

evaluate_folder(
    folder_path="results_topk_optimized",
    k=500,
    scale_tokens=5_000_000,
    csv_path="run_results_topk_optimized.csv",
)
```

### Adding LLM-as-a-Judge Grading

Along with more quantitative evaluations we can measure the models performance on more qualitative metrics like code quality, and task adherence. We have created a sample prompt for this called ``llm_as_judge.txt``.

```python
from scripts.llm_judge import judge_folder
```

```python

# Run LLM-as-judge for baseline results
judge_folder(
    results_dir="results_topk_baseline",
    out_dir=None,  # auto-map to results_llm_as_judge_baseline
    model="gpt-5",
    system_prompt_path="llm_as_judge.txt",
    task_text=None,  # use default task description
    concurrency=6,
)
```

```python

# Run LLM-as-judge for optimized results
judge_folder(
    results_dir="results_topk_optimized",
    out_dir=None,  # auto-map to results_llm_as_judge_optimized
    model="gpt-5",
    system_prompt_path="llm_as_judge.txt",
    task_text=None,
    concurrency=6,
)
```

### Summarizing the results

We can now demonstrate from both a quantitative standpoint, along with a qualitative standpoint from our LLM as Judge results.

```python
from pathlib import Path
import importlib
import scripts.results_summarizer as rs
from IPython.display import Markdown, display

importlib.reload(rs)

fig = rs.render_charts(
    quant_baseline=Path("results_topk_baseline")/"run_results_topk_baseline.csv",
    quant_optimized=Path("results_topk_optimized")/"run_results_topk_optimized.csv",
    judge_baseline=Path("results_llm_as_judge_baseline")/"judgement_summary.csv",
    judge_optimized=Path("results_llm_as_judge_optimized")/"judgement_summary.csv",
    auto_display=True,
    close_after=True,
)
md = rs.build_markdown_summary(
    quant_baseline=Path("results_topk_baseline")/"run_results_topk_baseline.csv",
    quant_optimized=Path("results_topk_optimized")/"run_results_topk_optimized.csv",
    judge_baseline=Path("results_llm_as_judge_baseline")/"judgement_summary.csv",
    judge_optimized=Path("results_llm_as_judge_optimized")/"judgement_summary.csv",
)

display(Markdown(md))

print(md)
```

### Prompt Optimization Results - Coding Tasks

| Metric                      | Baseline | Optimized | Δ (Opt − Base) |
|----------------------------|---------:|----------:|---------------:|
| Avg Time (s)                |    7.906 |     6.977 |        -0.929 |
| Peak Memory (KB)            |   3626.3 |     577.5 |       -3048.8 |
| Exact (%)                   |    100.0 |     100.0 |           0.0 |
| Sorted (%)