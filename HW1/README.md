# Sentiment Classification and Feature Engineering

#### Project done by Kiara LaRocca | UCSC | klarocca@ucsc.edu

# Sentiment Classification and Feature Engineering

## Overview

This repository contains code and resources for **binary sentiment classification** using machine learning models on text data. The project explores feature engineering techniques, compares classifier performance, and includes a custom implementation of a Naive Bayes classifier. It is divided into three parts:

1. **Part A**: Feature engineering and classifier comparison on Amazon Book Reviews.
2. **Part B**: Sentiment analysis on the Stanford Movie Review dataset.
3. **Part C**: Custom Naive Bayes classifier replicating Part A.

## Datasets

### Amazon Book Reviews
- **Dataset**: `small-books-rating.csv`
- **Labeling**: Reviews with scores ≥4 are positive (`1`), and scores ≤2 are negative (`0`). Reviews with a score of 3 are ignored.
  
### Stanford Movie Review Dataset
- **Dataset**: Includes 25,000 positive and 25,000 negative reviews.
  
## Features and Classifiers

### Feature Engineering
Three feature extraction methods are used:
1. **Bag of Words (BoW)**: Converts text into word frequency counts.
2. **TF-IDF**: Adds weight to rare words while penalizing frequent ones.
3. **N-grams**: Captures sequences of words (e.g., bigrams and trigrams) to add contextual information.

### Classifiers
In Parts A and B, the following classifiers are used:
- **Naive Bayes**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Logistic Regression** (Part B only)
- **Random Forest** (Part B only)
- **Gradient Boosting** (Part B only)
- **Linear Regression** (Part B only)

Part C implements a custom Naive Bayes classifier to compare with scikit-learn’s version.

This project is licensed under the MIT License.
