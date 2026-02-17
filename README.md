

# Hotel Review Sentiment & Rating Modeling

## Overview

This project explores transformer-based sentiment modeling for hotel review data, with the goal of predicting ordinal star ratings (1–5) from review text and analyzing the semantic structure underlying consumer feedback.

Using a fine-tuned **RoBERTa-base** model, we evaluate rating prediction performance under multiple modeling frameworks, including:

* Five-class ordinal classification
* Class-weighted training to address imbalance
* Three-class reframing (Low / Mid / High)
* Embedding-based clustering for semantic segmentation

The analysis demonstrates how label engineering and sentiment structure significantly influence classification performance and interpretability.

---

## Objectives

1. Establish a baseline RoBERTa model for 1–5 star rating prediction.
2. Evaluate the impact of class imbalance using weighted loss.
3. Reframe ordinal ratings into a three-class structure to improve separability.
4. Use transformer embeddings to identify latent consumer sentiment segments.
5. Compare modeling approaches and interpret class-level performance.

---

## Dataset

The dataset consists of hotel review text paired with numerical ratings (1–5 stars).

* Review: Free-text consumer feedback
* Rating: Ordinal scale from 1 (lowest) to 5 (highest)

Exploratory Data Analysis revealed class imbalance, with higher ratings more prevalent than lower ones.

---

## Modeling Approaches

### 1. Five-Class Ordinal Classification (1–5)

* Model: RoBERTa-base fine-tuned for multi-class classification
* Loss: Cross-entropy
* Metrics: Accuracy, Macro F1

**Results:**

* Accuracy ≈ 0.68
* Macro F1 ≈ 0.63
* Strong performance at rating extremes
* Significant confusion in mid-range ratings (2–3–4)

---

### 2. Weighted Five-Class Classification

* Introduced inverse-frequency class weights

**Results:**

* Minimal improvement in Macro F1
* Ordinal ambiguity dominated over imbalance effects

---

### 3. Three-Class Reframing (Low / Mid / High)

Mapping:

* Low: 1–2
* Mid: 3
* High: 4–5

**Results:**

* Accuracy ≈ 0.88
* Macro F1 ≈ 0.76
* High F1 ≈ 0.95
* Low F1 ≈ 0.84
* Mid F1 ≈ 0.50

This reframing significantly improved separability by collapsing noisy ordinal boundaries.

---

## Embedding-Based Sentiment Segmentation

Using RoBERTa embeddings:

1. Extracted CLS token embeddings.
2. Clustered within rating buckets.
3. Applied Bucket–Cluster Contrast scoring to identify distinctive lexical signals.

### Key Findings

Low ratings segmented into:

* Severe dissatisfaction (e.g., "dirty", "worst", "avoid")
* Mild disappointment
* Contextual trade-offs

Mid ratings segmented into:

* Budget-value framing
* Neutral adequacy
* Complaint-driven subsegments

High ratings segmented into:

* Emphatic praise ("wonderful", "fantastic")
* Value-oriented satisfaction
* Qualified approval

This revealed that rating categories are not homogeneous but contain meaningful consumer subtypes.

---

## Key Insights

* Extreme ratings (low and high) exhibit strong lexical polarity and high predictability.
* Middle ratings reflect inherent human subjectivity and sentiment ambiguity.
* Class weighting does not resolve ordinal noise.
* Label engineering materially improves performance.
* Embedding-based clustering enhances interpretability beyond raw classification.

---

## Repository Structure

```
roberta_hotel_review_rating_model.ipynb
README.md
```

---

## Dependencies

* transformers
* datasets
* scikit-learn
* torch
* nbformat (for notebook validation)

---

## Future Improvements

* Ordinal regression modeling
* DeBERTa comparison
* UMAP visualization of embedding clusters
* SHAP-based feature attribution
* Threshold optimization for high-satisfaction detection

---

## Author

Christopher Overton
M.S. Data Science


* A more research-style README
* A version emphasizing ML engineering
* Or a version optimized for recruiter readability
