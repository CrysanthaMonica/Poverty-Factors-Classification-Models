# Poverty Factors Classification using Machine Learning

## Overview
This project analyzes poverty factors in Indonesia using various machine learning classification models. It aims to classify regions in Indonesia into two categories—"poverty" and "no poverty"—based on the Global Multidimensional Poverty Index (MPI). By utilizing several models such as Logistic Regression, KNN, Naive Bayes, Decision Tree, and Gradient Boosting, the goal is to identify the most effective model in predicting poverty rates across different regions.

## Table of Contents
- [Introduction](#introduction)
- [Models Used](#models-used)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Challenges Faced](#challenges-faced)
- [Results](#results)
- [Further Improvements](#further-improvements)
- [Conclusion](#conclusion)

## Introduction
Poverty is a persistent issue globally, and Indonesia ranks 91st among the world’s poorest countries as of 2023. The poverty rate in Indonesia reached 9.36% in March 2023, affecting approximately 25.9 million people. Understanding and predicting poverty based on various social and economic factors can help in designing more effective interventions.

This project leverages machine learning models to analyze poverty data and classify regions based on their poverty index.

## Models Used
The following models were trained and evaluated to classify regions based on their poverty status:
1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Naive Bayes**
4. **Decision Tree**
5. **Gradient Boosting Classifier**

Each model's performance was evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Dataset
The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/ophi/mpi) and contains data related to the Multidimensional Poverty Index (MPI) of various regions across Indonesia. The dataset includes:
- **MPI National**: The national-level multidimensional poverty index.
- **MPI Regional**: The regional-level multidimensional poverty index.
- **Headcount Ratio Regional**: The percentage of the population considered poor at the regional level.
- **Intensity of Deprivation Regional**: The average distance below the poverty line for those considered poor at the regional level.

### Key Features
1. **ISO**: Unique country identifier.
2. **Country Name**: The name of the respective country.
3. **Sub-national region**: Regions within the country.
4. **Headcount Ratio Regional**: Percentage of the population considered poor at the regional level.
5. **MPI Regional**: Regional poverty index score.
6. **Label**: Assigned based on the MPI—less than 0.10 is classified as "No Poverty," and greater than 0.10 is classified as "Poverty."

## Methodology
1. **Data Preparation**: Feature engineering was applied to classify data based on the poverty level. The target variable ("Label") was assigned based on the MPI value.
2. **Model Selection**: Various classification algorithms were employed to predict whether a region falls under the "poverty" or "no poverty" category.
   - **Logistic Regression**: A statistical model that uses a logistic function to model a binary dependent variable.
   - **K-Nearest Neighbors (KNN)**: A non-parametric method used for classification by comparing data points based on their proximity.
   - **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
   - **Decision Tree**: A model that predicts the value of a target variable by learning decision rules inferred from data features.
   - **Gradient Boosting Classifier**: An ensemble learning method that builds models sequentially, reducing error in each iteration.
3. **Model Training and Evaluation**: Models were trained using the dataset and evaluated based on accuracy, precision, recall, and F1-score.

## Challenges Faced
### 1. Dataset Limitations
- **Limited Sample Size**: The dataset contained only 33 samples of regional data, making it difficult to generalize the results. Despite attempting oversampling, some models showed unreliable results due to the small dataset size.
  
- **Imbalance in Classes**: There was a notable imbalance between "poverty" and "no poverty" classes, with far fewer instances of "no poverty." This led to skewed results in some models, where the classifier overly favored the majority class.

### 2. Model Overfitting
- **Logistic Regression and KNN**: These models achieved perfect accuracy (1.00) across all evaluation metrics, which likely indicates overfitting due to the limited and imbalanced dataset. The perfect scores are unrealistic and suggest that the model learned patterns specific to the training data rather than generalizable features.

- **Naive Bayes and Decision Tree**: These models showed slightly lower accuracy (~0.89), but still exhibited signs of overfitting, as their performance on unseen data was not optimal. 

### 3. Model Robustness
- **Gradient Boosting Classifier**: While the Gradient Boosting Classifier showed more realistic results (0.90 accuracy) and balanced performance across metrics, it still struggled with the imbalanced dataset, especially in predicting the "poverty" class where the recall score fluctuated between 0.67 and 1.00.

## Results
The performance of the models was as follows:

| Model                | Accuracy | Precision (Poverty) | Recall (Poverty) | F1-Score (Poverty) |
|----------------------|----------|---------------------|------------------|--------------------|
| Logistic Regression   | 1.00     | 1.00                | 1.00             | 1.00               |
| KNN                  | 1.00     | 1.00                | 1.00             | 1.00               |
| Naive Bayes           | 0.89     | 0.86                | 0.67             | 0.80               |
| Decision Tree         | 0.89     | 0.86                | 0.67             | 0.80               |
| Gradient Boosting     | 0.90     | 0.88                | 0.67             | 0.80               |

The **Gradient Boosting Classifier** was the most reliable model, as it handled the data better than other models and provided a more realistic evaluation.

## Further Improvements
### 1. **Addressing Dataset Imbalance**
  - **Synthetic Data Generation**: One solution is to use techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic data points for under-represented classes. This could help mitigate the imbalance in "no poverty" samples and provide more data points for training, reducing the likelihood of overfitting.

  - **Class Weight Adjustment**: Adjusting the class weights in the model to penalize misclassifications in the minority class more heavily could improve the model's ability to learn from the imbalanced data.

### 2. **Model Generalization**
  - **Cross-validation**: Implementing k-fold cross-validation would help ensure that the model generalizes well to unseen data. Given the small sample size, using cross-validation can better evaluate model performance and reduce overfitting.
  
  - **Regularization Techniques**: Using techniques such as L2 regularization could reduce overfitting in models like **Logistic Regression** and **KNN**, which have demonstrated suspiciously high performance metrics.

### 3. **Feature Selection and Engineering**
  - **Enhanced Feature Engineering**: More advanced feature engineering could be performed on the dataset to create more informative features. For instance, creating interaction terms between features or deriving new variables based on regional socio-economic data could improve model performance.

  - **Dimensionality Reduction**: Applying PCA (Principal Component Analysis) or feature importance ranking from tree-based models could help reduce noise in the data, leading to better model accuracy and faster training times.

### 4. **Model Tuning and Ensemble Methods**
  - **Hyperparameter Tuning**: The models, particularly Gradient Boosting, could benefit from fine-tuning hyperparameters such as the learning rate, max depth, and number of estimators. This would likely improve the model's ability to distinguish between "poverty" and "no poverty" classes.
  
  - **Ensemble Learning**: Combining multiple models through ensemble methods like **stacking** could leverage the strengths of each model and produce a more robust classifier, especially for small and imbalanced datasets.

## Conclusion
The Gradient Boosting Classifier performed the best in classifying poverty status based on social and economic factors in Indonesia. However, due to the small and imbalanced dataset, all models are prone to overfitting and require further optimization. Implementing the suggested improvements, especially focusing on addressing the class imbalance and improving generalization techniques, could lead to more accurate and reliable predictions.
