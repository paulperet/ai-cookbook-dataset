# Evaluating Summarization Tasks: A Practical Guide

Evaluating the quality of generated summaries is a complex but essential task. Traditional metrics like ROUGE and BERTScore provide concrete, automated scores but often correlate poorly with human judgment, especially for open-ended, abstractive summarization. This guide walks you through a practical comparison of these traditional methods alongside a modern, reference-free approach using LLMs as evaluators, inspired by the G-Eval framework.

You will learn how to:
1.  Set up the necessary evaluation tools.
2.  Apply ROUGE and BERTScore to compare summaries against a reference.
3.  Implement a GPT-4-based evaluator to score summaries across multiple qualitative dimensions without needing a reference.

## Prerequisites & Setup

First, install the required Python packages. You will need `rouge` for traditional n-gram overlap scoring, `bert_score` for semantic similarity evaluation, and `openai` to interact with the GPT-4 API.

```bash
pip install rouge bert_score openai
```

Now, import the necessary libraries and set up your OpenAI client. Ensure your `OPENAI_API_KEY` is set as an environment variable.

```python
from openai import OpenAI
import os
import pandas as pd
from rouge import Rouge
from bert_score import BERTScorer

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

## Defining the Example Task

For this tutorial, we will evaluate two AI-generated summaries against a human-written reference. The source text is an excerpt about OpenAI's mission.

```python
excerpt = "OpenAI's mission is to ensure that artificial general intelligence (AGI) benefits all of humanity. OpenAI will build safe and beneficial AGI directly, but will also consider its mission fulfilled if its work aids others to achieve this outcome. OpenAI follows several key principles for this purpose. First, broadly distributed benefits - any influence over AGI's deployment will be used for the benefit of all, and to avoid harmful uses or undue concentration of power. Second, long-term safety - OpenAI is committed to doing the research to make AGI safe, and to promote the adoption of such research across the AI community. Third, technical leadership - OpenAI aims to be at the forefront of AI capabilities. Fourth, a cooperative orientation - OpenAI actively cooperates with other research and policy institutions, and seeks to create a global community working together to address AGI's global challenges."

# Human-written reference summary
ref_summary = "OpenAI aims to ensure artificial general intelligence (AGI) is used for everyone's benefit, avoiding harmful uses or undue power concentration. It is committed to researching AGI safety, promoting such studies among the AI community. OpenAI seeks to lead in AI capabilities and cooperates with global research and policy institutions to address AGI's challenges."

# AI-generated summaries for evaluation
eval_summary_1 = "OpenAI aims to AGI benefits all humanity, avoiding harmful uses and power concentration. It pioneers research into safe and beneficial AGI and promotes adoption globally. OpenAI maintains technical leadership in AI while cooperating with global institutions to address AGI challenges. It seeks to lead a collaborative worldwide effort developing AGI for collective good."
eval_summary_2 = "OpenAI aims to ensure AGI is for everyone's use, totally avoiding harmful stuff or big power concentration. Committed to researching AGI's safe side, promoting these studies in AI folks. OpenAI wants to be top in AI things and works with worldwide research, policy groups to figure AGI's stuff."
```

Take a moment to read the summaries. Which one do you think is better? This subjective judgment is what we aim to approximate with automated metrics.

## Step 1: Evaluating with ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the overlap of words and phrases between a generated summary and a reference. We'll calculate ROUGE-1 (unigram overlap), ROUGE-2 (bigram overlap), and ROUGE-L (longest common subsequence) scores.

First, define a helper function to calculate ROUGE scores.

```python
def get_rouge_scores(text1, text2):
    rouge = Rouge()
    return rouge.get_scores(text1, text2)
```

Now, apply this function to compare each AI-generated summary against the human reference.

```python
# Calculate ROUGE scores
eval_1_rouge = get_rouge_scores(eval_summary_1, ref_summary)
eval_2_rouge = get_rouge_scores(eval_summary_2, ref_summary)

# Organize the results into a DataFrame for comparison
rouge_scores_out = []
for metric in ["rouge-1", "rouge-2", "rouge-l"]:
    # We'll focus on the F1 score for each metric
    eval_1_score = eval_1_rouge[0][metric]['f']
    eval_2_score = eval_2_rouge[0][metric]['f']

    row = {
        "Metric": f"{metric} (F-Score)",
        "Summary 1": eval_1_score,
        "Summary 2": eval_2_score,
    }
    rouge_scores_out.append(row)

# Create a styled DataFrame, highlighting the higher score for each metric
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: lightgreen' if v else '' for v in is_max]

rouge_df = pd.DataFrame(rouge_scores_out).set_index("Metric")
styled_rouge_df = rouge_df.style.apply(highlight_max, axis=1)
styled_rouge_df
```

| Metric | Summary 1 | Summary 2 |
| :--- | :--- | :--- |
| **rouge-1 (F-Score)** | 0.488889 | **0.511628** |
| **rouge-2 (F-Score)** | **0.230769** | 0.163265 |
| **rouge-l (F-Score)** | 0.488889 | **0.511628** |

**Interpreting the Results:**
*   **Summary 2** scores higher on ROUGE-1 and ROUGE-L. This suggests it has better overlap of individual words and sentence structure with the reference summary, likely because it uses more phrases directly from the source text.
*   **Summary 1** scores higher on ROUGE-2, indicating better overlap of two-word sequences (bigrams).

While ROUGE provides a quantitative measure, it has limitations. It penalizes paraphrasing and cannot capture semantic meaning. A summary that perfectly captures the essence but uses different wording may receive a low ROUGE score.

## Step 2: Evaluating with BERTScore

BERTScore addresses ROUGE's limitation by using contextual embeddings from a model like BERT to evaluate semantic similarity between the candidate and reference texts.

Instantiate the BERTScorer and calculate the F1 score for each summary.

```python
# Instantiate the scorer. The first run will download the pre-trained model.
scorer = BERTScorer(lang="en")

# Calculate BERTScore (Precision, Recall, F1)
P1, R1, F1_1 = scorer.score([eval_summary_1], [ref_summary])
P2, R2, F1_2 = scorer.score([eval_summary_2], [ref_summary])

print("Summary 1 F1 Score:", F1_1.item())
print("Summary 2 F1 Score:", F1_2.item())
```

```
Summary 1 F1 Score: 0.9227314591407776
Summary 2 F1 Score: 0.9189572930335999
```

The F1 scores are very close, with Summary 1 having a slight edge. BERTScore suggests both summaries are semantically similar to the reference. However, like all automated metrics, BERTScore may not fully capture nuances like fluency, coherence, or factual consistency that a human would notice.

## Step 3: Evaluating with GPT-4 (Reference-Free)

We now implement a reference-free evaluator using GPT-4, following the G-Eval approach. This method assesses a summary's quality based solely on the source text, without needing a human reference. We will evaluate across four criteria: Relevance, Coherence, Consistency, and Fluency.

First, define the core prompt template and the specific criteria and steps for each metric.

```python
# Base template for the evaluation prompt
EVALUATION_PROMPT_TEMPLATE = """
You will be given one summary written for an article. Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions very carefully.
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}

Example:

Source Text:

{document}

Summary:

{summary}

Evaluation Form (scores ONLY):

- {metric_name}
"""

# Define the four evaluation metrics
RELEVANCY_SCORE_CRITERIA = """
Relevance(1-5) - selection of important content from the source. \
The summary should include only important information from the source document. \
Annotators were instructed to penalize summaries which contained redundancies and excess information.
"""
RELEVANCY_SCORE_STEPS = """
1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.
"""

COHERENCE_SCORE_CRITERIA = """
Coherence(1-5) - the collective quality of all sentences. \
We align this dimension with the DUC quality question of structure and coherence \
whereby "the summary should be well-structured and well-organized. \
The summary should not just be a heap of related information, but should build from sentence to a\
coherent body of information about a topic."
"""
COHERENCE_SCORE_STEPS = """
1. Read the article carefully and identify the main topic and key points.
2. Read the summary and compare it to the article. Check if the summary covers the main topic and key points of the article,
and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5.
"""

CONSISTENCY_SCORE_CRITERIA = """
Consistency(1-5) - the factual alignment between the summary and the source document. \
A factually consistent summary contains only statements that are entailed by the source document. \
Annotators were also asked to penalize summaries that contained hallucinated facts.
"""
CONSISTENCY_SCORE_STEPS = """
1. Read the article carefully and identify the facts it presents.
2. Read the summary and identify all claims or facts it states.
3. Check each claim in the summary against the article. Verify if the claim is supported by the article.
4. Assign a consistency score from 1 to 5.
"""

FLUENCY_SCORE_CRITERIA = """
Fluency(1-5) - the quality of the summary in terms of grammar, spelling, punctuation, and word choice.
"""
FLUENCY_SCORE_STEPS = """
1. Read the summary carefully.
2. Note any issues with grammar, spelling, punctuation, or awkward word choices.
3. Assign a fluency score from 1 to 5.
"""
```

Next, create a function that uses the template and the defined metrics to query GPT-4 for a score.

```python
def gpt4_evaluator(document, summary, criteria, steps, metric_name):
    """Queries GPT-4 to evaluate a summary based on a specific metric."""
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        document=document,
        summary=summary,
        metric_name=metric_name
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0 # Use low temperature for more deterministic scoring
    )
    # Extract the numeric score from the model's response
    result = response.choices[0].message.content.strip()
    # Simple regex to find a number 1-5 in the response
    match = re.search(r'\b([1-5])\b', result)
    if match:
        return int(match.group(1))
    else:
        print(f"Could not parse score from response: {result}")
        return None
```

Finally, run the evaluation for both summaries across all four metrics.

```python
import re

# Define the metrics to evaluate
metrics = {
    "Relevance": (RELEVANCY_SCORE_CRITERIA, RELEVANCY_SCORE_STEPS),
    "Coherence": (COHERENCE_SCORE_CRITERIA, COHERENCE_SCORE_STEPS),
    "Consistency": (CONSISTENCY_SCORE_CRITERIA, CONSISTENCY_SCORE_STEPS),
    "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
}

results = []
for summary_name, summary in [("Summary 1", eval_summary_1), ("Summary 2", eval_summary_2)]:
    row = {"Summary": summary_name}
    for metric_name, (criteria, steps) in metrics.items():
        score = gpt4_evaluator(excerpt, summary, criteria, steps, metric_name)
        row[metric_name] = score
        # Small delay to avoid rate limits
        time.sleep(1)
    results.append(row)

# Display the results
gpt4_results_df = pd.DataFrame(results).set_index("Summary")
gpt4_results_df
```

| Summary | Relevance | Coherence | Consistency | Fluency |
| :--- | :--- | :--- | :--- | :--- |
| **Summary 1** | 4 | 4 | 5 | 5 |
| **Summary 2** | 3 | 2 | 4 | 2 |

**Interpreting the Results:**
GPT-4 provides a nuanced, multi-dimensional assessment:
*   **Summary 1** scores highly across all categories, particularly in Consistency and Fluency. It is judged to be factually accurate, well-written, and logically structured.
*   **Summary 2** scores significantly lower, especially in Coherence and Fluency. Its informal language ("harmful stuff", "AI folks", "figure AGI's stuff") and awkward phrasing are penalized, despite it capturing some relevant facts (Relevance=3, Consistency=4).

This aligns with a likely human preference for Summary 1, demonstrating the strength of LLM-based, reference-free evaluation in capturing qualitative aspects that traditional metrics miss.

## Conclusion

You have successfully applied three distinct methods to evaluate text summarization:
1.  **ROUGE**: A fast, lexical overlap metric. Useful for a baseline but poor at evaluating meaning.
2.  **BERTScore**: A semantic similarity metric. Better at capturing meaning but still an imperfect proxy for quality.
3.  **GPT-4 Evaluator**: A reference-free, multi-criteria judge. Excels at evaluating higher-order qualities like coherence and fluency, closely mimicking human judgment.

For robust evaluation, consider a hybrid approach. Use traditional metrics like ROUGE for quick, automated checks during model development, and employ LLM-based or human evaluation for final validation and nuanced quality assessment.