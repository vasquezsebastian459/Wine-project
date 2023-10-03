# 🍷 Wine Quality Prediction

This repository is dedicated to predicting wine quality based on its physicochemical properties. By using various machine learning models, we evaluate their performance in terms of their accuracy, classification metrics, and more. Visual analytics also play a significant role in our data exploration process.

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Technologies & Libraries](#technologies--libraries)
- [Data Exploration & Visualization](#data-exploration--visualization)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Visualizations](#visualizations)
- [Getting Started](#getting-started)

## Dataset Overview

The dataset used in this repository is `winequality-red.csv`, which contains various physicochemical properties of red wines alongside their corresponding quality ratings.

- **Features**: Physicochemical tests results (e.g., acidity, sugar level).
- **Target Variable**: Quality rating of the wine.

## Technologies & Libraries

The analysis and model training make use of several Python libraries:

- **scikit-learn**: For machine learning and data preprocessing.
- **pandas**: For data manipulation.
- **numpy**: For numerical operations.
- **seaborn & matplotlib**: For data visualization.

## Data Exploration & Visualization

Before diving into modeling, we first explore and visualize the data to understand its structure, distribution, and relationships:

- **Histograms**: These plots offer insights into the distribution of individual features.
- **Scatter Plots**: Used to observe relationships or patterns, like the one between total sulfur dioxide and wine quality.
- **Bar Plots**: These highlight relationships between quality and other features like volatile acidity and citric acid.
- **Correlation Heatmap**: Provides a quick overview of how features relate to one another.

## Model Training and Evaluation

We've employed three diverse machine learning models:

1. **Random Forest Classifier**: An ensemble model consisting of numerous decision trees.
2. **SGD Classifier**: Linear classifiers (SVM, logistic regression, etc.) with SGD training.
3. **Logistic Regression**: A classic choice for binary classification problems.

For each of the models, the performance metrics include:

- **Accuracy Score**: Provides a quick insight into the overall correctness of predictions.
- **Classification Report**: Highlights precision, recall, and F1-score.
- **ROC Curve & AUC Score**: Shows trade-off between sensitivity and specificity.
- **Confusion Matrix**: A detailed view of true positive, true negative, false positive, and false negative predictions.

## Visualizations

Visualizations offer an intuitive way to comprehend model performances and data structures:

- `images` folder contains saved visualizations:
  - Feature distribution histograms.
  - Scatter and bar plots.
  - Correlation matrix heatmap.
  - ROC curves and confusion matrices for each model.


