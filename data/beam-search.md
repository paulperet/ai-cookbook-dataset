# A Practical Guide to Beam Search for Sequence Generation

## Introduction

In sequence-to-sequence models, generating output sequences requires a decoding strategy. While greedy search is computationally efficient, it often produces suboptimal results. Exhaustive search guarantees the best sequence but is computationally infeasible. Beam search strikes a practical balance between these extremes. This guide walks you through the concepts and mathematics behind these strategies.

## Prerequisites

This tutorial assumes familiarity with:
- Sequence-to-sequence architectures
- Conditional probability and sequence generation
- Basic algorithmic complexity concepts

## Understanding the Search Space

Before diving into strategies, let's establish our notation:

- **Vocabulary (Y)**: The set of all possible output tokens, including the special end-of-sequence token `<eos>`
- **Maximum sequence length (T')**: The upper bound on output sequence length
- **Context variable (c)**: The encoded representation of the input sequence

The total number of possible output sequences is approximately |Y|^T', which grows exponentially with sequence length.

## Strategy 1: Greedy Search

### How Greedy Search Works

At each time step t', greedy search selects the token with the highest conditional probability:

```python
y_t' = argmax_{y ∈ Y} P(y | y_1, ..., y_{t'-1}, c)
```

The process continues until `<eos>` is generated or the maximum length T' is reached.

### Computational Cost

Greedy search has a computational complexity of O(|Y|T'), making it extremely efficient.

### The Greedy Search Problem

While efficient, greedy search doesn't guarantee the most probable sequence. Consider this example:

**Example 1: Greedy Path**
- Time step 1: Select A (P=0.5)
- Time step 2: Select B (P=0.4)
- Time step 3: Select C (P=0.4)
- Time step 4: Select `<eos>` (P=0.6)

Sequence probability: 0.5 × 0.4 × 0.4 × 0.6 = 0.048

**Example 2: Alternative Path**
- Time step 1: Select A (P=0.5)
- Time step 2: Select C (P=0.3) ← Second best at step 2
- Time step 3: Select B (P=0.6)
- Time step 4: Select `<eos>` (P=0.6)

Sequence probability: 0.5 × 0.3 × 0.6 × 0.6 = 0.054

The alternative path (0.054) has higher probability than the greedy path (0.048), demonstrating that greedy search can miss better sequences.

## Strategy 2: Exhaustive Search

### How Exhaustive Search Works

Exhaustive search evaluates all possible sequences and selects the one with the highest overall probability:

```python
best_sequence = argmax_{y_1,...,y_L} ∏_{t'=1}^L P(y_t' | y_1, ..., y_{t'-1}, c)
```

### Computational Cost

The complexity is O(|Y|^T'), which becomes prohibitive quickly:
- For |Y| = 10,000 and T' = 10: 10000^10 = 10^40 sequences
- This is computationally infeasible for practical applications

## Strategy 3: Beam Search

### The Beam Search Algorithm

Beam search maintains k candidate sequences at each time step, where k is the beam size.

**Step 1: Initialize**
At time step 1, select the k tokens with highest P(y₁ | c)

**Step 2: Expand Candidates**
For each subsequent time step:
1. For each of the k current candidates, compute probabilities for all |Y| possible next tokens
2. Select the top k sequences from the k × |Y| possibilities

**Step 3: Terminate**
Continue until all sequences end with `<eos>` or reach maximum length T'

### Example Walkthrough

Let's trace through an example with:
- Vocabulary: Y = {A, B, C, D, E, `<eos>`}
- Beam size: k = 2
- Maximum length: 3

**Time Step 1:**
- Top 2 tokens: A (P=0.6), C (P=0.4)
- Candidates: {A}, {C}

**Time Step 2:**
- Expand A: Compute P(A,y₂|c) for all y₂ ∈ Y
- Expand C: Compute P(C,y₂|c) for all y₂ ∈ Y
- Top 2 sequences: {A,B} (P=0.3), {C,E} (P=0.25)

**Time Step 3:**
- Expand {A,B}: Compute P(A,B,y₃|c) for all y₃ ∈ Y
- Expand {C,E}: Compute P(C,E,y₃|c) for all y₃ ∈ Y
- Top 2 sequences: {A,B,D} (P=0.15), {C,E,D} (P=0.12)

**Final candidates:** A, C, AB, CE, ABD, CED

### Sequence Scoring

To compare sequences of different lengths, use normalized log probability:

```
score = (1 / L^α) × Σ_{t'=1}^L log P(y_t' | y_1, ..., y_{t'-1}, c)
```

Where:
- L: Sequence length
- α: Length penalty (typically 0.75)
- The L^α term penalizes longer sequences

### Computational Cost

Beam search complexity: O(k|Y|T')
- For |Y| = 10,000, T' = 10, k = 5: 5 × 10,000 × 10 = 500,000 computations
- This is feasible while providing better results than greedy search

## Practical Considerations

### Choosing Beam Size
- **k = 1**: Equivalent to greedy search (fastest, lowest quality)
- **Small k (2-10)**: Good balance for most applications
- **Large k (50+)**: Approaches exhaustive search quality (slower)

### Length Penalty (α)
- α = 0: No length normalization
- α = 1: Linear length normalization
- α = 0.75: Common default (moderate penalty for longer sequences)

## Summary

| Strategy | Complexity | Quality | Practical Use |
|----------|------------|---------|---------------|
| Greedy Search | O(|Y|T') | Low | Fast inference, low-resource settings |
| Exhaustive Search | O(|Y|^T') | Optimal | Theoretically optimal, impractical |
| Beam Search | O(k|Y|T') | High | Standard for production systems |

## Key Takeaways

1. **Greedy search** is efficient but can produce suboptimal sequences
2. **Exhaustive search** guarantees the best sequence but is computationally impossible
3. **Beam search** provides a tunable trade-off between quality and computation
4. The **beam size k** controls this trade-off
5. **Length normalization** is essential for comparing sequences of different lengths

## Exercises for Practice

1. **Theoretical Understanding**: Can exhaustive search be considered a special case of beam search? What would the beam size need to be?

2. **Practical Application**: Implement beam search for a machine translation task. Experiment with different beam sizes (1, 5, 10, 50) and observe:
   - How translation quality changes
   - How inference speed scales with beam size
   - The point of diminishing returns

3. **Strategy Analysis**: In text generation with prefix completion, which search strategy is typically used? How could beam search improve results while maintaining reasonable speed?

## Further Reading

For implementation details and community discussions, visit the [D2L discussion forum](https://discuss.d2l.ai/t/338).

---

*Note: Beam search is the standard decoding algorithm for most sequence generation tasks in production systems, including machine translation, text summarization, and dialogue generation. The choice of beam size represents a key engineering trade-off between quality and latency.*