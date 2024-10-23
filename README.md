# E-Commerce Product Classification Dataset

## Overview

This project classifies product descriptions into various categories using a machine learning model. The dataset contains product names or descriptions labeled under categories such as "Household," "Electronics," "Clothing & Accessories," and "Books."

### Dataset Information:
- **Total Rows:** 24,000
- **Columns:**
  - `Text`: The product description/name.
  - `Label`: The product category label.

### Example Rows:

| Text | Label |
| --- | --- |
| Urban Ladder Eisner Low Back Study-Office Computer Chair (Brown) | Household |
| Contrast Living Wooden Decorative Box, Painted Multicolour | Household |
| IO Crest SY-PCI40010 PCI RAID Host Controller | Electronics |
| ISAKAA Baby Socks from Just Born to 8 Years - Pack of 5 Pairs | Clothing & Accessories |
| Indira Designer Women's Art Mysore Silk Saree (Green) | Clothing & Accessories |

## Machine Learning Pipeline

The classification model is built using a **TF-IDF Vectorizer** and a **K-Nearest Neighbors (KNN)** classifier. The TF-IDF vectorizer converts the product descriptions into numerical features based on the frequency of words, and the KNN classifier is used to predict the category of the product based on these features.

### Pipeline:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer()),    
    ('KNN', KNeighborsClassifier())
])

Model Performance
The model was trained and evaluated on the dataset with the following results:
              precision    recall  f1-score   support

           0       0.96      0.95      0.95      1206
           1       0.97      0.96      0.97      1209
           2       0.98      0.97      0.97      1208
           3       0.95      0.97      0.96      1177

    accuracy                           0.96      4800
   macro avg       0.96      0.96      0.96      4800
weighted avg       0.96      0.96      0.96      4800
