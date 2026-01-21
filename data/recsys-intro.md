# Recommender Systems: An Introduction

## Overview

In the past decade, the Internet has transformed into a platform for large-scale online services, fundamentally changing how we communicate, consume news, shop, and watch entertainment. With an unprecedented number of items available online—movies, news articles, books, products—we need systems to help us discover content we prefer. Recommender systems are powerful information filtering tools that provide personalized services and tailored experiences. They play a pivotal role in managing the vast amount of data available to make choices manageable.

Today, recommender systems are core components of major online service providers like Amazon, Netflix, and YouTube. The benefits are twofold:
- **For users:** They reduce the effort required to find relevant items and alleviate information overload.
- **For businesses:** They add significant value and are an important revenue source.

This guide introduces the fundamental concepts, classic models, and recent advances in recommender systems, with a focus on deep learning approaches.

## Core Concepts

### Collaborative Filtering

Collaborative Filtering (CF) is a foundational concept in recommender systems. The term was first coined by the Tapestry system, referring to "people collaborate to help one another perform the filtering process." In a broad sense, CF involves filtering information or patterns using techniques that leverage collaboration among multiple users, agents, and data sources.

CF techniques are generally categorized into three types:

1. **Memory-based CF:** Uses the entire user-item dataset to make predictions. Examples include nearest neighbor-based approaches like user-based CF and item-based CF.
2. **Model-based CF:** Uses machine learning models to learn patterns from the data. A classic example is matrix factorization.
3. **Hybrid CF:** Combines memory-based and model-based approaches.

Memory-based CF can struggle with sparse and large-scale data because it computes similarity based on common items. Model-based methods often handle sparsity and scalability better. Many model-based CF approaches can be enhanced with neural networks, leading to more flexible and scalable models.

CF primarily relies on user-item interaction data. Other approaches, like content-based and context-based recommender systems, incorporate additional information such as item descriptions, user profiles, timestamps, and locations.

### Explicit vs. Implicit Feedback

To learn user preferences, systems collect feedback, which can be **explicit** or **implicit**.

- **Explicit Feedback:** Direct input from users indicating their preferences (e.g., star ratings on IMDb, thumbs-up/down on YouTube). Gathering explicit feedback requires proactive user action and is not always readily available.
- **Implicit Feedback:** Inferred from user behavior (e.g., purchase history, clicks, browsing history, watch time). Implicit feedback is more readily available but is inherently noisy—observing a behavior doesn't always indicate a positive preference.

### Common Recommendation Tasks

Various recommendation tasks have been explored, often defined by the application domain or the type of data used:

- **Rating Prediction:** Predicts explicit ratings (e.g., stars).
- **Top-N Recommendation (Item Ranking):** Ranks all items for a user based on implicit feedback.
- **Sequence-Aware Recommendation:** Incorporates temporal data (timestamps) to model user behavior over time.
- **Click-Through Rate (CTR) Prediction:** Predicts the likelihood of a user clicking on an item, often using various categorical features.
- **Cold-Start Recommendation:** Addresses scenarios involving new users or new items with little to no interaction history.

## Summary

- Recommender systems are crucial for both users and industries, with Collaborative Filtering being a key underlying concept.
- Feedback can be explicit (direct ratings) or implicit (inferred from behavior), each with its own advantages and challenges.
- A variety of recommendation tasks exist, tailored to different domains and data types.

## Exercises

1. How do recommender systems influence your daily online activities?
2. What novel recommendation tasks or challenges do you think are worth investigating?

---
*Next: We will dive into implementing a basic collaborative filtering model using matrix factorization.*