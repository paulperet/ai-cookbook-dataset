# A Guide to Understanding and Handling Distribution Shift in Machine Learning

## Introduction

In previous sections, we've built machine learning models on various datasets, but we haven't stopped to consider where our data comes from or how our models will be used in practice. Many failed machine learning deployments trace back to this oversight.

Models can perform well on test data but fail catastrophically when deployed if the data distribution shifts. Even worse, deploying a model can sometimes cause the very distribution shifts that break it.

Consider a loan approval model that learns to associate footwear with default risk (Oxfords = repayment, sneakers = default). If we start denying loans to sneaker-wearers, applicants will quickly adapt by wearing Oxfords, without any actual improvement in creditworthiness. The model becomes useless because our decisions changed the environment.

This guide will help you understand different types of distribution shift, recognize them in practice, and implement strategies to handle them.

## Types of Distribution Shift

### Covariate Shift
Covariate shift occurs when the distribution of inputs changes over time, but the relationship between inputs and outputs remains constant. Statisticians call this "covariate shift" because the covariates (features) shift while the conditional distribution P(y|x) stays the same.

**Example**: Training a cat/dog classifier on photos but deploying it on cartoon images. The features (pixel patterns) differ between training and test data, but "cat" and "dog" still mean the same thing.

### Label Shift
Label shift is the opposite problem: the distribution of labels P(y) changes, but the conditional distribution P(x|y) remains fixed. This often happens when y causes x.

**Example**: Medical diagnosis where disease prevalence changes over time. The symptoms for each disease remain the same, but some diseases become more or less common.

### Concept Shift
Concept shift occurs when the very definitions of labels change over time.

**Example**: Regional differences in terminology. What's called "pop" in some parts of the US is called "soda" in others. If you build a machine translation system, P(y|x) might differ by location.

## Real-World Examples of Distribution Shift

### Medical Diagnostics Failure
A startup developed a blood test for a disease affecting older men. For healthy controls, they used blood from university students. Their classifier achieved near-perfect accuracy distinguishing patients from students, but for all the wrong reasons: it learned to detect age, hormone levels, and lifestyle differences rather than disease markers. When deployed on real patients of similar ages, it failed completely.

### Self-Driving Car Pitfalls
A company trained a roadside detector using synthetic data from a game engine. The detector worked perfectly on synthetic test data but failed in real cars because it had learned to recognize the simplistic, uniform textures used in the game rather than actual roadside features.

The US Army had a similar experience trying to detect tanks in forests. Their classifier learned to distinguish morning shadows from noon shadows (when the photos were taken) rather than actual tanks.

### Nonstationary Distributions
Many real-world distributions change slowly:
- Advertising models that don't account for new products
- Spam filters that can't adapt to new spam techniques  
- Recommendation systems recommending Christmas items year-round

## Mathematical Framework

### Empirical Risk vs. True Risk
During training, we minimize empirical risk:

```
minimize_f (1/n) Σ l(f(x_i), y_i)
```

where l is our loss function. This approximates the true risk:

```
E_p(x,y)[l(f(x), y)] = ∫∫ l(f(x), y) p(x, y) dx dy
```

The empirical risk is what we can compute with our training data; the true risk is what we actually care about for deployment.

## Correcting Distribution Shift

### Covariate Shift Correction
When we have labeled data from source distribution q(x) but need to perform well on target distribution p(x), with P(y|x) constant, we can reweight training examples by:

```
β_i = p(x_i) / q(x_i)
```

Then we perform weighted empirical risk minimization:

```
minimize_f (1/n) Σ β_i l(f(x_i), y_i)
```

**Implementation Algorithm**:
1. Create binary classification dataset: {(x_i, -1)} from training, {(u_i, 1)} from test
2. Train logistic regression classifier h to distinguish source from target
3. Compute weights β_i = exp(h(x_i)) (clipped if necessary)
4. Train main model with weighted loss using β_i

**Important Assumption**: Every test example must have non-zero probability in the training distribution. If p(x) > 0 but q(x) = 0, the weight becomes infinite.

### Label Shift Correction
For label shift where P(x|y) is constant but P(y) changes, we weight by:

```
β_i = p(y_i) / q(y_i)
```

To estimate p(y) without test labels:
1. Compute confusion matrix C on validation data
2. Compute mean predictions μ(ŷ) on test data  
3. Solve linear system: C p(y) = μ(ŷ)
4. Use solution p(y) = C^(-1) μ(ŷ) to compute weights

### Concept Shift Correction
Concept shift is harder to correct. Often the best approach is to:
- Continuously collect new labeled data
- Update models incrementally rather than retraining from scratch
- Monitor performance drift over time

## Learning Problem Taxonomy

### Batch Learning
Traditional approach: train once on fixed dataset, deploy static model.

### Online Learning
Data arrives sequentially. For each x_t, we:
1. Make prediction f_t(x_t)
2. Observe true label y_t  
3. Update model to f_{t+1} based on loss l(y_t, f_t(x_t))

### Bandits
Special case with finite action space. Stronger theoretical guarantees possible.

### Reinforcement Learning
Environment responds to our actions. Need to consider long-term consequences and opponent strategies.

## Ethical Considerations

When deploying ML systems, you're not just optimizing predictions—you're automating decisions that affect people's lives.

**Key Issues**:
- **Fairness**: Systems may work differently for different populations
- **Feedback Loops**: Predictions can influence future training data
- **Accuracy vs. Impact**: Low accuracy might be acceptable if errors are harmless; high accuracy unacceptable if errors cause serious harm

**Example - Predictive Policing**:
1. High-crime neighborhoods get more patrols
2. More crimes are discovered there (due to increased surveillance)
3. Model predicts even more crime for these neighborhoods
4. More patrols are deployed, creating a self-reinforcing cycle

## Summary

- **Distribution shift** occurs when training and test data come from different distributions
- **Covariate shift**: Input distribution changes, P(y|x) constant
- **Label shift**: Label distribution changes, P(x|y) constant  
- **Concept shift**: Definitions of labels change
- **Empirical risk minimization** approximates true risk minimization
- **Reweighting techniques** can correct for certain types of shift
- **Ethical considerations** are crucial when models influence real-world decisions

## Exercises

1. Consider a search engine that changes its ranking algorithm. How might users adapt their search behavior? How might advertisers adapt?
2. Implement a covariate shift detector using a binary classifier.
3. Implement the covariate shift correction algorithm described above.
4. Besides distribution shift, what other factors affect how well empirical risk approximates true risk? (Hint: consider finite sample sizes, model complexity, etc.)