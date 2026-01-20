# 1. Executive Summary
---

## 1.1. Purpose and Audience

This notebook provides a hands-on guide for building **temporally-aware knowledge graphs** and performing **multi-hop retrieval directly over those graphs**. 

It's designed for engineers, architects, and analysts working on temporally-aware knowledge graphs. Whether you’re prototyping, deploying at scale, or exploring new ways to use structured data, you’ll find practical workflows, best practices, and decision frameworks to accelerate your work.

This cookbook presents two hands-on workflows you can use, extend, and deploy right away:

1. **Temporally-aware knowledge graph (KG) construction**
A key challenge in developing knowledge-driven AI systems is maintaining a database that stays current and relevant. While much attention is given to boosting retrieval accuracy with techniques like semantic similarity and re-ranking, this guide focuses on a fundamental—yet frequently overlooked—aspect: *systematically updating and validating your knowledge base as new data arrives*.
No matter how advanced your retrieval algorithms are, their effectiveness is limited by the quality and freshness of your database. This cookbook demonstrates how to routinely validate and update knowledge graph entries as new data arrives, helping ensure that your knowledge base remains accurate and up to date.

2. **Multi-hop retrieval using knowledge graphs**
Learn how to combine OpenAI models (such as o3, o4-mini, GPT-4.1, and GPT-4.1-mini) with structured graph queries via tool calls, enabling the model to traverse your graph in multiple steps across entities and relationships.
This method lets your system answer complex, multi-faceted questions that require reasoning over several linked facts, going well beyond what single-hop retrieval can accomplish.

Inside, you'll discover:

* **Practical decision frameworks** for choosing models and prompting techniques at each stage
* **Plug-and-play code examples** for easy integration into your ML and data pipelines
* **Links to in-depth resources** on OpenAI tool use, fine-tuning, graph backend selection, and more
* **A clear path from prototype to production**, with actionable best practices for scaling and reliability

> **Note:** All benchmarks and recommendations are based on the best available models and practices as of June 2025. As the ecosystem evolves, periodically revisit your approach to stay current with new capabilities and improvements.

## 1.2. Key takeaways

### Creating a Temporally-Aware Knowledge Graph with a Temporal Agent
1. **Why make your knowledge graph temporal?**
Traditional knowledge graphs treat facts as static, but real-world information evolves constantly. What was true last quarter may be outdated today, risking errors or misinformed decisions if the graph does not capture change over time. Temporal knowledge graphs allow you to precisely answer questions like “What was true on a given date?” or analyse how facts and relationships have shifted, ensuring decisions are always based on the most relevant context.

2. **What is a Temporal Agent?**
A Temporal Agent is a pipeline component that ingests raw data and produces time-stamped triplets for your knowledge graph. This enables precise time-based querying, timeline construction, trend analysis, and more.

3. **How does the pipeline work?**
The pipeline starts by semantically chunking your raw documents. These chunks are decomposed into statements ready for our Temporal Agent, which then creates time-aware triplets. An Invalidation Agent can then perform temporal validity checks, spotting and handling any statements that are invalidated by new statements that are incident on the graph.

### Multi-Step Retrieval Over a Knowledge Graph
1. **Why use multi-step retrieval?**
Direct, single-hop queries frequently miss salient facts distributed across a graph's topology. Multi-step (multi-hop) retrieval enables iterative traversal, following relationships and aggregating evidence across several hops. This methodology surfaces complex dependencies and latent connections that would remain hidden with one-shot lookups, providing more comprehensive and nuanced answers to sophisticated queries.

2. **Planners**
Planners orchestrate the retrieval process. *Task-orientated* planners decompose queries into concrete, sequential subtasks. *Hypothesis-orientated* planners, by contrast, propose claims to confirm, refute, or evolve. Choosing the optimal strategy depends on where the problem lies on the spectrum from deterministic reporting (well-defined paths) to exploratory research (open-ended inference).

3. **Tool Design Paradigms**
Tool design spans a continuum: *Fixed tools* provide consistent, predictable outputs for specific queries (e.g., a service that always returns today’s weather for San Francisco). At the other end, *Free-form tools* offer broad flexibility, such as code execution or open-ended data retrieval. *Semi-structured tools* fall between these extremes, restricting certain actions while allowing tailored flexibility—specialized sub-agents are a typical example. Selecting the appropriate paradigm is a trade-off between control, adaptability, and complexity.

4. **Evaluating Retrieval Systems**
High-fidelity evaluation hinges on expert-curated "golden" answers, though these are costly and labor-intensive to produce. Automated judgments, such as those from LLMs or tool traces, can be quickly generated to supplement or pre-screen, but may lack the precision of human evaluation. As your system matures, transition towards leveraging real user feedback to measure and optimize retrieval quality in production.
A proven workflow: Start with synthetic tests, benchmark on your curated human-annotated "golden" dataset, and iteratively refine using live user feedback and ratings.

### Prototype to Production
1. **Keep the graph lean**
Established archival policies and assign numeric relevance scores to each edge (e.g., recency x trust x query-frequency). Automate the archival or sparsification of low-value nodes and edges, ensuring only the most critical and frequently accessed facts remain for rapid retrieval.

2. **Parallelize the ingestion pipeline**
Transition from a linear document → chunk → extraction → resolution pipeline to a staged, asynchronous architecture. Assign each processing phase its own queue and dedicated worker pool. Apply clustering or network-based batching for invalidation jobs to maximize efficiency. Batch external API requests (e.g., OpenAI) and database writes wherever possible. This design increases throughput, introduces backpressure for reliability, and allows you to scale each pipeline stage independently.

3. **Integrate Robust Production Safeguards**
Enforce rigorous output validation: standardise temporal fields (e.g., ISO-8601 date formatting), constrain entity types to your controlled vocabulary, and apply lightweight model-based sanity checks for output consistency. Employ structured logging with traceable identifiers and monitor real-time quality and performance metrics in real lime to proactively detect data drift, regressions, or pipeline anomalised before they impact downstream applications.

# 2. How to Use This Cookbook
---

This cookbook is designed for flexible engagement:

1. Use it as a comprehensive technical guide—read from start to finish for a deep understanding of temporally-aware knowledge graph systems.
2. Skim for advanced concepts, methodologies, and implementation patterns if you prefer a high-level overview.
3. Jump into any of the three modular sections; each is self-contained and directly applicable to real-world scenarios.

Inside, you'll find:

1. **Creating a Temporally-Aware Knowledge Graph with a Temporal Agent**
Build a pipeline that extracts entities and relations from unstructured text, resolves temporal conflicts, and keeps your graph up-to-date as new information arrives.

2. **Multi-Step Retrieval Over a Knowledge Graph**
Use structured queries and language model reasoning to chain multiple hops across your graph and answer complex questions.

3. **Prototype to Production**
Move from experimentation to deployment. This section covers architectural tips, integration patterns, and considerations for scaling reliably.


## 2.1. Pre-requisites

Before diving into building temporal agents and knowledge graphs, let's set up your environment. Install all required dependencies with pip, and set your OpenAI API key as an environment variable. Python 3.12 or later is required.


```python
!python -V
%pip install --upgrade pip
%pip install -qU chonkie datetime ipykernel jinja2 matplotlib networkx numpy openai plotly pydantic rapidfuzz scipy tenacity tiktoken pandas
%pip install -q "datasets<3.0"
```

    Python 3.12.8
    Requirement already satisfied: pip in ./.venv/lib/python3.12/site-packages (25.1.1)
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.



```python
import os

if "OPENAI_API_KEY" not in os.environ:
    import getpass
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Paste your OpenAI API key here: ")
```

# 3. Creating a Temporally-Aware Knowledge Graph with a Temporal Agent
---

**Accurate data is the foundation of any good business decision.** 
OpenAI’s latest models like o3, o4-mini, and the GPT-4.1 family are enabling businesses to build state-of-the-art retrieval systems for their most important workflows. However, information evolves rapidly: facts ingested confidently yesterday may already be outdated today.

Without the ability to track when each fact was valid, retrieval systems risk returning answers that are outdated, non-compliant, or misleading. The consequences of missing temporal context can be severe in any industry, as illustrated by the following examples.

| Industry | Example question | Risk if database is not temporal |
|----------|------------------|----------------------------------|
| **Financial Services** | *"How has Moody’s long‑term rating for Bank YY evolved since Feb 2023?"* | Mispricing credit risk by mixing historical & current ratings |
| | *"Who was the CFO of Retailer ZZ when the FY‑22 guidance was issued?"* | Governance/insider‑trading analysis may blame the wrong executive |
| | *"Was Fund AA sanctioned under Article BB at the time it bought Stock CC in Jan 2024?"* | Compliance report could miss an infraction if rules changed later |
| **Manufacturing / Automotive** | *"Which ECU firmware was deployed in model Q3 cars shipped between 2022‑05 and 2023‑03?"* | Misdiagnosing field failures due to firmware drift |
| | *"Which robot‑controller software revision ran on Assembly Line 7 during Lot 8421?"* | Root‑cause analysis may blame the wrong software revision |
| | *"What torque specification applied to steering‑column bolts in builds produced in May 2024?"* | Safety recall may miss affected vehicles |

While we've called out some specific examples here, this theme is true across many industries including pharmaceuticals, law, consumer goods, and more.

**Looking beyond standard retrieval**

A temporally-aware knowledge graph allows you to go beyond static fact lookup. It enables richer retrieval workflows such as factual Q&A grounded in time, timeline generation, change tracking, counterfactual analysis, and more. We dive into these in more detail in our retrieval section later in the cookbook.

## 3.1. Introducing our Temporal Agent
---

A **temporal agent** is a specialized pipeline that converts raw, free-form statements into time-aware triplets ready for ingesting into a knowledge graph that can then be queried with the questions of the character *“What was true at time T?”*. 

Triplets are the basic building blocks of knowledge graphs. It's a way to represent a single fact or piece of knowledge using three parts (hence, *"triplet"*): 
- **Subject** - the entity you are talking about
- **Predicate** - the type of relationship or property
- **Object** - the value or other entity that the subject is connected to

You can thinking of this like a sentence with a structure `[Subject] - [Predicate] - [Object]`. As a more clear example:
```
"London" - "isCapitalOf" - "United Kingdom"
```

The Temporal Agent implemented in this cookbook draws inspiration from [Zep](https://arxiv.org/abs/2501.13956) and [Graphiti](https://github.com/getzep/graphiti), while introducing tighter control over fact invalidation and a more nuanced approach to episodic typing.

### 3.1.1. Key enhancements introduced in this cookbook

1. **Temporal validity extraction**
Builds on Graphiti's prompt design to identify temporal spans and episodic context without requiring auxiliary reference statements.

2. **Fact invalidation logic**
Introduces bidirectionality checks and constrains comparisons by episodic type. This retains Zep's non-lossy approach while reducing unnecessary evaluations.

3. **Temporal & episodic typing**
Differentiates between `Fact`, `Opinion`, `Prediction`, as well as between temporal classes `Static`, `Dynamic`, `Atemporal`.

4. **Multi‑event extraction**
Handles compound sentences and nested date references in a single pass.

This process allows us to update our sources of truth efficiently and reliably:

> **Note**: While the implementation in this cookbook is focused on a graph-based implementation, this approach is generalizable to other knowledge base structures e.g., pgvector-based systems.
---

### 3.1.2. The Temporal Agent Pipeline

The Temporal Agent processes incoming statements through a three-stage pipeline:

1. **Temporal Classification**
Labels each statement as **Atemporal**, **Static**, or **Dynamic**:
- *Atemporal* statements never change (e.g., “The speed of light in a vaccuum is ≈3×10⁸ m s⁻¹”).
- *Static* statements are valid from a point in time but do not change afterwards (e.g., "Person YY was CEO of Company XX on October 23rd 2014.").
- *Dynamic* statements evolve (e.g., "Person YY is CEO of Company XX.").

2. **Temporal Event Extraction**
Identifies relative or partial dates (e.g., “Tuesday”, “three months ago”) and resolves them to an absolute date using the document timestamp or fallback heuristics (e.g., default to the 1st or last of the month if only the month is known).

3. **Temporal Validity Check**
Ensures every statement includes a `t_created` timestamp and, when applicable, a `t_expired` timestamp. The agent then compares the candidate triplet to existing knowledge graph entries to:
- Detect contradictions and mark outdated entries with `t_invalid`
- Link newer statements to those they invalidate with `invalidated_by`

### 3.1.3. Selecting the right model for a Temporal Agent
When building systems with LLMs, it is a good practice to [start with larger models then later look to optimize and shrink](https://platform.openai.com/docs/guides/model-selection). 

The GPT-4.1 series is particularly well-suited for building Temporal Agents due to its strong instruction-following ability. On benchmarks like Scale’s MultiChallenge, [GPT-4.1 outperforms GPT-4o by $10.5\%_{abs}$](https://openai.com/index/gpt-4-1/), demonstrating superior ability to maintain context, reason in-conversation, and adhere to instructions - key traits for extracting time-stamped triplets. These capabilities make it an excellent choice for prototyping agents that rely on time-aware data extraction.

#### Recommended development workflow
1. **Prototype with GPT-4.1**
Maximize correctness and reduce prompt-debug time while you build out the core pipeline logic.

2. **Swap to GPT-4.1-mini or GPT-4.1-nano**
Once prompts and logic are stable, switch to smaller variants for lower latency and cost-effective inference.

3. **Distill onto GPT-4.1-mini or GPT-4.1-nano**
Use [OpenAI's Model Distillation](https://platform.openai.com/docs/guides/distillation) to train smaller models with high-quality outputs from a larger 'teacher' model such as GPT-4.1, preserving (or even improving) performance relative to GPT-4.1.

| Model | Relative cost | Relative latency | Intelligence | Ideal Role in Workflow |
|-------|---------------|------------------|--------------|------------------------|
| *GPT-4.1* | ★★★ | ★★ | ★★★ *(highest)* | Ground-truth prototyping, generating data for distillation |
| *GPT-4.1-mini* | ★★ | ★ | ★★ | Balanced cost-performance, mid to large scale production systems |
| *GPT-4.1-nano* | ★ *(lowest)* | ★ *(fastest)* | ★ | Cost-sensitive and ultra-large scale bulk processing |

> In practice, this looks like: prototype with GPT-4.1 → measure quality → step down the ladder until the trade-offs no longer meet your needs.

## 3.2. Building our Temporal Agent Pipeline
---
Before diving into the implementation details, it's useful to understand the ingestion pipeline at a high level:

1. **Load transcripts**

2. **Creating a Semantic Chunker**

3. **Laying the Foundations for our Temporal Agent**

4. **Statement Extraction**

5. **Temporal Range Extraction**

6. **Creating our Triplets**

7. **Temporal Events**

8. **Defining our Temporal Agent**

9. **Entity Resolution**

10. **Invalidation Agent**

11. **Building our pipeline**

### Architecture diagram

### 3.2.1. Load transcripts
For the purposes of this cookbook, we have selected the ["Earnings Calls Dataset" (jlh-ibm/earnings_call)](https://huggingface.co/datasets/jlh-ibm/earnings_call) which is made available under the Creative Commons Zero v1.0 license. This dataset contains a collection of 188 earnings call transcripts originating in the period 2016-2020 in relation to the NASDAQ stock market. We believe this dataset is a good choice for this cookbook as extracting information from - and subsequently querying information from - earnings call transcripts is a common problem in many financial institutions around the world. 

Moreover, the often variable character of statements and topics from the same company across multiple earnings calls provides a useful vector through which to demonstrate the temporal knowledge graph concept. 

Despite this dataset's focus on the financial world, we build up the Temporal Agent in a general structure, so it will be quick to adapt to similar problems in other industries such as pharmaceuticals, law, automotive, and more. 

For the purposes of this cookbook we are limiting the processing to two companies - AMD and Nvidia - though in practice this pipeline can easily be scaled to any company. 

Let’s start by loading the dataset from HuggingFace.


```python
from datasets import load_dataset

hf_dataset_name = "jlh-ibm/earnings_call"
subset_options = ["stock_prices", "transcript-sentiment", "transcripts"]

hf_dataset = load_dataset(hf_dataset_name, subset_options[2])
my_dataset = hf_dataset["train"]
```


```python
my_dataset
```




    Dataset({
        features: ['company', 'date', 'transcript'],
        num_rows: 150
    })




```python
row = my_dataset[0]
row["company"], row["date"], row["transcript"][:200]
```


```python
from collections import Counter

company_counts = Counter(my_dataset["company"])
company_counts
```

**Database Set-up**


Before we get to processing this