# Naive Bayes Classifier

Naive Bayes is a probabilistic machine learning algorithm based on Bayes' theorem, which describes the probability of an event based on prior knowledge of conditions that might be related to the event. The "naive" assumption in Naive Bayes is that features used to describe instances are conditionally independent, given the class label. Despite its simplicity, Naive Bayes is effective for various classification tasks, especially in text and document classification.

## Key Concepts

### Bayes' Theorem

The algorithm is rooted in Bayes' theorem, which calculates the probability of a hypothesis given the evidence:

\[ P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)} \]

where:

- \( P(A | B) \) is the probability of hypothesis A given evidence B.
- \( P(B | A) \) is the probability of evidence B given hypothesis A.
- \( P(A) \) is the prior probability of hypothesis A.
- \( P(B) \) is the prior probability of evidence B.

### Conditional Independence

Naive Bayes assumes that the features used to describe an instance are conditionally independent given the class label. This simplifies the calculations and makes the algorithm computationally efficient.

## Applications

Naive Bayes is commonly used in various applications, including:

- **Text Classification:** Spam filtering, sentiment analysis, and topic categorization.
- **Medical Diagnosis:** Predicting the likelihood of a disease based on symptoms.
- **Recommendation Systems:** Classifying user preferences for personalized recommendations.

## Types of Naive Bayes Classifiers

1. **Gaussian Naive Bayes:** Assumes that features follow a normal distribution.
2. **Multinomial Naive Bayes:** Used for discrete data, often applied in text classification.
3. **Bernoulli Naive Bayes:** Suitable for binary feature variables.
