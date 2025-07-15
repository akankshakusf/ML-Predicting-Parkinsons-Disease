# 2025_IA651_Mbuwayesango_Moyo

# Predicting Parkinson’s Disease Progression Using Machine Learning

## Project overview 
Parkinson’s Disease is a progressive neurological disorder that affects movement, speech, and various motor functions. Early and accurate tracking of its progression is critical for effective treatment and patient care. In this project, we leverage machine learning techniques to analyze a dataset containing voice measurements and clinical attributes from individuals with Parkinson’s Disease.

The primary objective is to build predictive models that estimate the progression of the disease using biomedical voice features and patient demographics. Through exploratory data analysis, feature engineering, and model development, we aim to identify patterns and markers that are indicative of disease advancement. This work not only showcases the power of machine learning in medical diagnostics but also contributes toward the development of non-invasive tools for disease monitoring.

## Dataset
The dataset used in this project is centered on analyzing the progression of Parkinson’s Disease based on various clinical, functional, and lifestyle-related features. It is sourced from Kaggle - Parkinson’s Disease Progression Dataset, which provides anonymized health records of 500 patients living with Parkinson’s Disease.

The dataset includes the following fields:

Patient_ID: Unique identifier for each patient

Age: Patient’s age

Gender: Biological sex of the patient (M/F)

Years_Since_Diagnosis: Time (in years) since Parkinson’s diagnosis

UPDRS_Score: Unified Parkinson's Disease Rating Scale score

Tremor_Severity: Severity of tremors (scale: 0–5)

Motor_Function: Motor function impairment rating (scale: 0–5)

Speech_Difficulty: Degree of speech difficulty (scale: 0–5)

Balance_Problems: Level of balance issues experienced (scale: 0–5)

Medications: Primary medication prescribed (e.g., Levodopa, Amantadine, Ropinirole, Pramipexole)

Exercise_Level: Self-reported physical activity level (Low, Moderate, High)

Disease_Progression: Disease progression severity (target variable, scale: 1–3)

This dataset supports the development of machine learning models that can predict disease progression based on physical, behavioral, and treatment-related attributes. Such models can aid in early intervention, personalized care, and informed decision-making for both clinicians and patients.


## Process Overview
This project was an iterative process of exploration and refinement. We started with data cleaning and EDA to understand key patterns and feature relationships. Early modeling attempts using basic classifiers revealed the need for better preprocessing, so we incorporated feature scaling and PCA to improve performance and interpretability.

Along the way, we adjusted our approach based on insights—giving more weight to clinically relevant features like UPDRS_Score and Exercise_Level. These pivots helped us build a more effective model and strengthened our understanding of the data and its real-world implications.

## Exploratory Data Analysis (EDA)

The dataset consists of 500 observations and 11 relevant features (excluding the Patient_ID). The X variables include demographic and clinical attributes such as Age, Gender, Tremor_Severity, Motor_Function, Speech_Difficulty, Exercise_Level, and Medications. The target variable (Y) is Disease_Progression, which classifies patients into three severity levels: mild, moderate, and severe. This defines our problem as a multi-class classification task. With a strong feature-to-observation ratio, the dataset is well-suited for building and evaluating predictive models.


## Features
The dataset includes the following features (columns):
![image](https://github.com/user-attachments/assets/f199d598-1a65-4ead-b71f-75200eb9f043)

## Target Variable 
The target variable in this dataset is Disease_Progression, which classifies patients into three categories based on the severity of Parkinson’s Disease: 1 for Mild, 2 for Moderate, and 3 for Severe. It represents the progression stage of the disease and is used for multi-class classification. Predicting this variable accurately is essential for identifying patient needs, tailoring treatment plans, and enabling early interventions in managing Parkinson’s Disease.

## Feature distribution 
![image](https://github.com/user-attachments/assets/2ff82ca5-367c-4e16-aeda-40b7ea2cda5b)



## Distribution of continuous variables 
![image](https://github.com/user-attachments/assets/d0a054fb-8944-4b15-9af2-1140b888006e)

## Correlation Analysis
The correlation heatmap shows generally weak relationships between most numeric features. Years_Since_Diagnosis has a slight positive correlation with Motor_Function and Balance_Problems, while Tremor_Severity is slightly negatively correlated with Age. Notably, UPDRS_Score shows minimal correlation with Disease_Progression, suggesting that linear relationships alone may not fully explain progression severity. Overall, no strong multicollinearity was observed, and all features were retained for modeling.

![image](https://github.com/user-attachments/assets/ab5110a8-e969-45a1-9fb2-6698c36b3a4e)

## Feature Engineering 

To enhance model performance and interpretability, we performed several feature engineering steps:

Dropped non-informative columns: Removed Patient_ID as it does not carry predictive value.

Categorical Encoding: Converted categorical features like Gender, Medications, and Exercise_Level into numerical format using label encoding or one-hot encoding depending on the model.

Feature Scaling: Applied standardization (e.g., StandardScaler) to continuous variables such as Age and UPDRS_Score to ensure uniform scale across features.

Dimensionality Reduction: Performed PCA (Principal Component Analysis) to reduce noise and better visualize feature structure during EDA.

Correlation Check: Examined feature correlation to identify redundant variables, though no features were removed due to weak correlations.

Target Encoding Insight: Ensured the target variable Disease_Progression was treated as a categorical class for classification models.

## Principal Component Analysis (PCA)
PCA was performed to reduce the dimensionality of the dataset and to visualize the variance explained by each component. The following chart shows the explained variance by each principal component:

![image](https://github.com/user-attachments/assets/554d7ce1-ccc2-4737-b251-2df19d81806d)

![image](https://github.com/user-attachments/assets/92448d00-b3de-427a-ac60-cef678ce8120)

![image](https://github.com/user-attachments/assets/d363a1b7-43b4-4041-a70f-fa4a83a694c0)

The PCA results indicated that approximately 90% of the variance in the dataset is explained by the first 9 principal components, suggesting that these components capture most of the meaningful information. This dimensionality reduction helped uncover the underlying structure of the data, reduced noise, and provided insights into which features contributed most to patient variability. These insights were valuable in guiding feature selection and improving the efficiency of our modeling process.

## Model Fitting
To assess the predictive performance of various classification algorithms, the dataset was split into training and testing subsets using an 80/20 stratified split. Stratification ensured that all classes of the target variable, Disease_Progression, were proportionally represented in both subsets. This approach preserved class distribution while providing sufficient data for both model training and unbiased performance evaluation.

### Data preprocessing involved:

Encoding categorical variables using one-hot encoding

Standardizing continuous features with StandardScaler to normalize scale-sensitive models

For model development, we implemented a mix of linear and non-linear classification techniques:

Logistic Regression (Multinomial) and Support Vector Machine (SVC) were tuned using GridSearchCV to optimize key hyperparameters.

Decision Tree and Random Forest classifiers were trained using manually selected or default hyperparameters.

Additional ensemble models combining SVC, Random Forest, and Gradient Boosting were also evaluated to explore hybrid strategies.

Model evaluation was based on key classification metrics: accuracy, precision, recall, F1-score (macro), and confusion matrices. Cross-validation was applied during training to assess model robustness, and we reported the mean and standard deviation of macro F1-scores across folds for comparative insight.

As the dataset is non-temporal, no special handling for time-based data leakage was required. Nonetheless, strict separation between training and testing data was maintained throughout preprocessing and model fitting to prevent any form of data leakage.

The rationale for evaluating multiple models and hyperparameter tuning was to benchmark both linear and non-linear methods and determine the most effective classifier for predicting the progression of Parkinson’s Disease based on clinical and lifestyle predictors.


## Models
### Logistic Regression 
The confusion matrix for the Logistic Regression (Multinomial) model shows that the model struggles to accurately classify the three categories (Slow, Moderate, Fast).

![confusion matrix for logistic regression](https://github.com/user-attachments/assets/85c1c610-fc36-45f5-9209-b247760d7ae8)

The high number of misclassifications, especially predicting "Slow" and "Moderate" as "Fast", indicates that the model has difficulty distinguishing between the classes, particularly between the slower and faster progression categories. This suggests that the model may be biased towards the "Fast" class, possibly due to an imbalance in the dataset or model limitations.

### Descision Tree 
The confusion matrix for the Decision Tree Classifier shows moderate performance in distinguishing between the three classes (Slow, Moderate, Fast).


![confusion matrix -descision tree](https://github.com/user-attachments/assets/e3ca896a-4e3e-41aa-8065-76736c29cd4c)


Compared to the Logistic Regression model, the Decision Tree shows improved accuracy for the "Slow" class but continues to struggle with correctly identifying "Moderate" and "Fast" instances.The model has a tendency to misclassify "Moderate" cases as either "Slow" or "Fast", indicating difficulty in capturing the nuanced differences between the progression speeds.This pattern suggests that while the Decision Tree has slightly better classification of the "Slow" class compared to Logistic Regression, it still lacks consistency in correctly classifying the other classes.

### SVC 
The confusion matrix for the SVC model shows a clear pattern where the model is heavily biased towards predicting a single class.

![confusion matrix -SVC](https://github.com/user-attachments/assets/729958ac-edd4-4c8c-9a12-a1afcb5f67da)

### Random Forrest 

![random forrest - feature importance](https://github.com/user-attachments/assets/9059f4ad-cbcd-45e0-a9a8-274834cdceb6)
![random forrest - confusion matrix](https://github.com/user-attachments/assets/4a9d4c19-8c39-4d18-a4bb-b0a80f03cf36)

### Ensamble -RF + SVC + XGBoost

The confusion matrix for the Ensemble Model (Random Forest + SVC + XGBoost) shows a mixed performance in distinguishing between the three classes (Slow, Moderate, Fast).

![image](https://github.com/user-attachments/assets/31447f2c-3b85-4888-b370-1a7e7ba37345)

The ensemble approach managed to reduce the extreme bias seen in the standalone SVC model but at the cost of introducing more confusion between the moderate and fast classes. This indicates that while the ensemble approach improves generalization, it may also blur the distinction between similar classes.


## ROC-AUC Curves
The Multiclass ROC Curve plot shows the Receiver Operating Characteristic (ROC) curves for each classification model tested on the dataset. The ROC curve visually represents the trade-off between the True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity) for each model

![image](https://github.com/user-attachments/assets/289eefac-a4b7-4f2e-a7ee-8de5d98e03ff)


## Consolidated Model Metrics

![Screenshot 2025-05-06 195234](https://github.com/user-attachments/assets/4e596292-ca84-4606-b848-7d047a39cc21)

### Accuracy

The SVC model has the highest accuracy (0.36), indicating that it correctly predicts the class more often compared to other models.

The Decision Tree and Ensemble methods both achieved 0.35 accuracy, suggesting that they performed similarly in classification tasks.

### F1 Score (Macro)

Decision Tree Classifier and Ensemble Model (RF + SVC + XGB) both have the highest F1 score (0.3500), indicating better balance between precision and recall compared to other models.

The SVC model has a significantly lower F1 score (0.1800), reflecting its poor performance in predicting minority classes despite having the highest accuracy.

### Cross-Validation Mean F1 (CV Mean F1)

The Decision Tree model has the highest CV Mean F1 (0.3729), indicating that it consistently performs well across cross-validation folds.

The Ensemble model also shows a strong CV Mean F1 of 0.3619, suggesting good generalization.

### Cross-Validation Standard Deviation (CV Std F1)

The SVC model shows the lowest standard deviation (0.0018), indicating stable performance across different data splits.

In contrast, the Ensemble model shows a relatively higher standard deviation (0.0608), indicating that its performance varies more depending on the training data.


## Conclusion
The evaluation of multiple classification models for predicting Parkinson's Disease progression revealed important insights into their respective performances.

The Decision Tree Classifier and Ensemble Model (RF + SVC + XGB) demonstrated the most consistent performance in terms of F1 score, indicating their ability to balance precision and recall effectively across all classes. In contrast, the SVC model, despite achieving the highest accuracy (0.36), exhibited significant challenges in handling class imbalance, resulting in a considerably lower F1 score (0.18). This discrepancy suggests that while SVC accurately predicts the majority class, it struggles with correctly classifying minority classes.

The Ensemble method successfully integrates the strengths of Random Forest, SVC, and XGBoost, yielding a more balanced and robust performance. By combining multiple learning algorithms, the ensemble approach mitigates the bias inherent in individual models, thereby offering a more generalized predictive capability. This ensemble strategy helps address the limitations of single models, particularly in complex, multi-class classification problems.

## 🏆 Final Verdict:
Taking into account both accuracy and F1 score, the Ensemble Model (RF + SVC + XGB) emerges as the most balanced and reliable model for predicting Parkinson’s Disease progression. While SVC shows promising accuracy, its inability to handle class imbalance makes it less suitable for this task compared to the ensemble approach.

In conclusion, leveraging an ensemble of diverse classifiers proves to be a more effective strategy in addressing the inherent challenges of multi-class classification within this dataset. Further improvements may include optimizing the ensemble weights or incorporating additional models to enhance predictive performance.
























