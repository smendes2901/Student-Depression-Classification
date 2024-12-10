# Student Depression Classification

This project focuses on predicting the likelihood of depression among students based on a dataset from [Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset). The classification task is achieved using various machine learning models, including Logistic Regression, Random Forest, Decision Tree, Naive Bayes, Support Vector Machine (SVM), and Artificial Neural Networks (ANN).

## Table of Contents
- [Student Depression Classification](#student-depression-classification)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dataset](#dataset)
  - [Features](#features)
  - [Models](#models)
  - [Results](#results)
  - [Requirements](#requirements)

## Introduction
The objective of this project is to identify students experiencing depression based on a set of features, allowing for early intervention and mental health support. The dataset contains responses to a questionnaire aimed at identifying symptoms of depression.

## Dataset
The dataset is obtained from [Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) and includes the following key attributes:
- Demographic details (age, gender, etc.)
- Academic and social factors
- Mental health indicators based on survey responses

The dataset is cleaned and preprocessed before training the models.

## Features
The key features:
- Suicidal Thoughts
- Academic Pressure
- Financial Stress
- Study Satisfaction

## Models
The following models were trained and evaluated for the classification task:
1. **Logistic Regression**: A baseline linear model for binary classification.
2. **Random Forest**: A tree-based ensemble learning method.
3. **Decision Tree**: A simple, interpretable tree-based classifier.
4. **Naive Bayes**: A probabilistic model based on Bayes' theorem.
5. **Support Vector Machine (SVM)**: A robust classifier for non-linear data.
6. **Artificial Neural Network (ANN)**: A deep learning approach for complex relationships.

## Results
Each model's performance was evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1-Score

| Model                     | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| ------------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression       | 0.84     | 0.83      | 0.84   | 0.84     | 0.92    |
| SVM                       | 0.84     | 0.83      | 0.84   | 0.84     | 0.92    |
| Decision Tree             | 0.83     | 0.81      | 0.82   | 0.82     | 0.88    |
| Random Forest             | 0.84     | 0.84      | 0.84   | 0.84     | 0.92    |
| Naive Bayes               | 0.84     | 0.83      | 0.84   | 0.85     | 0.92    |
| K-Nearest Neighbour       | 0.82     | 0.79      | 0.80   | 0.81     | 0.92    |
| Artificial Neural Network | 0.84     | 0.84      | 0.84   | 0.84     | 0.92    |
| XGBoost                   | 0.84     | 0.83      | 0.84   | 0.84     | 0.92    |

## Requirements
Install the necessary libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt