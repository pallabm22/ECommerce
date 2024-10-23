# E-Commerce Product Classification Dataset

## Overview

This project focuses on classifying product descriptions into various categories using a machine learning model. The dataset consists of product names or descriptions labeled under categories such as "Household," "Electronics," "Clothing & Accessories," and "Books." This classification task can be helpful in building recommendation systems, automating product categorization for e-commerce platforms, and improving search results.

### Dataset Information

- **Total Rows:** 24,000
- **Columns:**
  - `Text`: The product description or name.
  - `Label`: The product category label, such as "Household," "Electronics," "Clothing & Accessories," or "Books."

### Example Rows:

| Text                                                                | Label                 |
|---------------------------------------------------------------------|-----------------------|
| Urban Ladder Eisner Low Back Study-Office Computer Chair (Brown)     | Household             |
| Contrast Living Wooden Decorative Box, Painted Multicolour           | Household             |
| IO Crest SY-PCI40010 PCI RAID Host Controller                        | Electronics           |
| ISAKAA Baby Socks from Just Born to 8 Years - Pack of 5 Pairs        | Clothing & Accessories|
| Indira Designer Women's Art Mysore Silk Saree (Green)                | Clothing & Accessories|

---

## Machine Learning Pipeline

The classification model is built using a **TF-IDF Vectorizer** and a **K-Nearest Neighbors (KNN)** classifier. Below is a brief description of the pipeline:

1. **TF-IDF Vectorizer**: 
   - Converts product descriptions into numerical feature vectors based on term frequency-inverse document frequency (TF-IDF).
   
2. **K-Nearest Neighbors Classifier**: 
   - A supervised machine learning algorithm used to predict the product category based on the product description's TF-IDF vector. It classifies a product based on the categories of its nearest neighbors in the feature space.

---

## Model Performance

The model was trained and evaluated on a subset of the dataset, and the following classification metrics were observed:

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.96      | 0.95   | 0.95     | 1206    |
| 1     | 0.97      | 0.96   | 0.97     | 1209    |
| 2     | 0.98      | 0.97   | 0.97     | 1208    |
| 3     | 0.95      | 0.97   | 0.96     | 1177    |

**Overall Metrics:**

- **Accuracy**: 96%
- **Macro Average Precision, Recall, F1-Score**: 0.96
- **Weighted Average Precision, Recall, F1-Score**: 0.96


---
