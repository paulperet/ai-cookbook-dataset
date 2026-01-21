# A Practical Guide to Statistical Inference for Machine Learning

## Introduction

Statistics is the science of collecting, processing, analyzing, and interpreting data. For machine learning practitioners, statistical inference—the process of deducing population characteristics from samples—is essential for evaluating model improvements, testing hypotheses, and quantifying uncertainty.

This guide covers three fundamental statistical inference methods:
1. **Evaluating and comparing estimators**
2. **Conducting hypothesis tests**
3. **Constructing confidence intervals**

## Prerequisites

We'll use common deep learning frameworks for our examples. Choose one and install the necessary libraries.

```bash
# For MXNet
pip install mxnet

# For PyTorch
pip install torch

# For TensorFlow
pip install tensorflow
```

## 1. Evaluating and Comparing Estimators

An estimator is a function of samples used to estimate a true population parameter θ. For example, the sample mean estimates the population mean. We evaluate estimators using three key metrics.

### 1.1 Mean Squared Error (MSE)

MSE measures the average squared deviation between an estimator and the true parameter:

```python
import numpy as np

def mse(data, true_theta):
    return np.mean(np.square(data - true_theta))
```

### 1.2 Statistical Bias

Bias measures the systematic error of an estimator:

```python
def stat_bias(true_theta, est_theta):
    return np.mean(est_theta) - true_theta
```

### 1.3 Variance and Standard Deviation

Variance measures the estimator's fluctuation around its own expected value:

```python
def estimator_variance(estimates):
    return np.var(estimates)
```

### 1.4 The Bias-Variance Trade-off

The MSE decomposes into three components:
```
MSE = Bias² + Variance + Irreducible Error
```

Let's verify this with a simulation:

```python
# Generate samples from a normal distribution
theta_true = 1
sigma = 4
sample_len = 10000
samples = np.random.normal(theta_true, sigma, sample_len)

# Use sample mean as our estimator
theta_est = np.mean(samples)

# Calculate MSE
mse_value = mse(samples, theta_true)

# Calculate bias² + variance
bias = stat_bias(theta_true, theta_est)
variance = np.var(samples)
bias_variance_sum = bias**2 + variance

print(f"MSE: {mse_value:.4f}")
print(f"Bias² + Variance: {bias_variance_sum:.4f}")
print(f"Difference: {abs(mse_value - bias_variance_sum):.6f}")
```

The two values should agree to numerical precision, demonstrating the bias-variance decomposition.

## 2. Conducting Hypothesis Tests

Hypothesis testing evaluates evidence against a default statement (null hypothesis) about a population.

### 2.1 Key Concepts

- **Null Hypothesis (H₀)**: Default statement to test
- **Alternative Hypothesis (H₁)**: Contrary statement
- **Statistical Significance (1-α)**: Probability of not rejecting H₀ when it's true
- **Statistical Power (1-β)**: Probability of rejecting H₀ when it's false
- **p-value**: Probability of observing a test statistic at least as extreme as the one calculated, assuming H₀ is true

### 2.2 General Steps for Hypothesis Testing

1. **State the question** and establish null hypothesis H₀
2. **Set significance level α** (commonly 0.05) and desired power (commonly 0.8)
3. **Obtain samples** through experiments
4. **Calculate test statistic** and p-value
5. **Make decision**: Reject H₀ if p-value ≤ α

### 2.3 Example: Testing a New Teaching Method

Imagine testing whether a new teaching method improves test scores:

```python
import scipy.stats as stats

# Simulated test scores
control_scores = np.random.normal(75, 10, 100)  # Traditional method
treatment_scores = np.random.normal(78, 10, 100)  # New method

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(treatment_scores, control_scores)

print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject null hypothesis: The new method has a significant effect.")
else:
    print("Fail to reject null hypothesis: No significant effect detected.")
```

## 3. Constructing Confidence Intervals

A confidence interval provides a range of plausible values for a parameter with a specified level of confidence.

### 3.1 Definition

A (1-α)% confidence interval Cₙ satisfies:
```
P(Cₙ contains θ) ≥ 1 - α for all θ
```

### 3.2 Common Misinterpretations to Avoid

1. **False**: "There's a 95% probability that θ is in the interval"
2. **False**: "Values inside the interval are more likely to be θ"
3. **False**: "A particular 95% CI has 95% probability of containing θ"

The correct interpretation: If we repeated the experiment many times, (1-α)% of the constructed intervals would contain θ.

### 3.3 Gaussian Mean Confidence Interval

For a Gaussian distribution with unknown mean and variance:

```python
def confidence_interval(samples, confidence=0.95):
    """Calculate confidence interval for the mean"""
    n = len(samples)
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)  # Sample standard deviation
    
    # For large n, use z-score (1.96 for 95% confidence)
    # For small n, use t-distribution critical value
    if n >= 30:
        critical_value = 1.96  # z-score for 95% confidence
    else:
        # Use t-distribution with n-1 degrees of freedom
        critical_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    
    margin_of_error = critical_value * std / np.sqrt(n)
    return (mean - margin_of_error, mean + margin_of_error)

# Generate sample data
np.random.seed(42)
samples = np.random.normal(loc=0, scale=1, size=1000)

# Calculate 95% confidence interval
ci_low, ci_high = confidence_interval(samples)
print(f"95% Confidence Interval: [{ci_low:.3f}, {ci_high:.3f}]")
print(f"Sample Mean: {np.mean(samples):.3f}")
print(f"True Mean (0) in interval: {ci_low <= 0 <= ci_high}")
```

## Practical Example: Complete Analysis Workflow

Let's walk through a complete example analyzing whether a new algorithm improves processing time.

```python
def complete_statistical_analysis():
    # Step 1: Generate simulated data
    np.random.seed(42)
    n_samples = 50
    
    # Current algorithm processing times (seconds)
    current_times = np.random.exponential(scale=2.0, size=n_samples)
    
    # New algorithm processing times (10% faster on average)
    new_times = np.random.exponential(scale=1.8, size=n_samples)
    
    # Step 2: Calculate point estimates
    current_mean = np.mean(current_times)
    new_mean = np.mean(new_times)
    improvement = current_mean - new_mean
    
    print(f"Current algorithm mean: {current_mean:.3f}s")
    print(f"New algorithm mean: {new_mean:.3f}s")
    print(f"Mean improvement: {improvement:.3f}s ({improvement/current_mean*100:.1f}%)")
    
    # Step 3: Hypothesis test
    t_stat, p_value = stats.ttest_ind(current_times, new_times)
    print(f"\nHypothesis Test Results:")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Conclusion: Significant difference detected (reject H₀)")
    else:
        print("Conclusion: No significant difference detected (fail to reject H₀)")
    
    # Step 4: Confidence intervals
    ci_current = confidence_interval(current_times)
    ci_new = confidence_interval(new_times)
    
    print(f"\n95% Confidence Intervals:")
    print(f"Current algorithm: [{ci_current[0]:.3f}, {ci_current[1]:.3f}]")
    print(f"New algorithm: [{ci_new[0]:.3f}, {ci_new[1]:.3f}]")
    
    # Step 5: Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(current_times) + np.var(new_times)) / 2)
    cohens_d = improvement / pooled_std
    print(f"\nEffect Size (Cohen's d): {cohens_d:.3f}")
    
    if abs(cohens_d) < 0.2:
        print("Effect size: Very small")
    elif abs(cohens_d) < 0.5:
        print("Effect size: Small")
    elif abs(cohens_d) < 0.8:
        print("Effect size: Medium")
    else:
        print("Effect size: Large")

complete_statistical_analysis()
```

## Key Takeaways

1. **Estimator Evaluation**: Use MSE, bias, and variance to compare different estimators. Remember the bias-variance trade-off in model selection.

2. **Hypothesis Testing**: Follow the five-step process to test claims about your data. Always pre-specify your significance level and consider statistical power when designing experiments.

3. **Confidence Intervals**: Provide more information than point estimates by indicating uncertainty. Interpret them correctly: they're about the procedure, not a particular interval.

4. **Practical Application**: Combine these methods to make data-driven decisions. For example, when A/B testing a new feature, use hypothesis testing to determine if there's a significant effect and confidence intervals to understand the magnitude and precision of your estimate.

## Exercises for Practice

1. **Estimator Comparison**: Generate 1000 samples from Uniform(0, θ) with θ=5. Compare the estimators max(X₁,...,Xₙ) and 2×mean(X₁,...,Xₙ) using bias, variance, and MSE.

2. **Power Analysis**: For the chemist example in the introduction, calculate the required sample size to detect a 10% improvement with 80% power and 95% significance.

3. **Confidence Interval Simulation**: Generate 100 datasets with N=2 from N(0,1). Calculate 50% confidence intervals (t_star=1.0) and plot them. Observe how many contain the true mean (0) and reflect on the interpretation of confidence intervals.

Remember: Statistics provides the tools to quantify uncertainty and make informed decisions from data. While machine learning focuses on prediction, statistics helps us understand the reliability and significance of our results.