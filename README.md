# Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to churn based on customer profile, service usage, contract, and billing information.

## Project Overview

Customer churn is an important business problem for telecom companies because losing existing customers can directly affect revenue.

This project uses Machine Learning to:

- Analyze customer data
- Identify important churn-related features
- Train classification models
- Compare Logistic Regression and Random Forest
- Predict customer churn probability
- Provide an interactive Streamlit web application
- Generate business insights for customer retention

## Business Problem

The objective is to predict whether a customer is likely to leave the telecom service.

The prediction can help businesses identify customers who may be at risk and support data-driven customer retention strategies.

## Dataset

The project uses a Telco Customer Churn dataset containing customer demographic, service, contract, and billing information.

Important features include:

- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Internet Service
- Payment Method
- Online Security
- Tech Support
- Gender
- Senior Citizen
- Partner
- Dependents
- Phone Service
- Streaming Services

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## Machine Learning Workflow

### 1. Data Loading

The dataset is loaded using Pandas and examined for structure, data types, missing values, and relevant variables.

### 2. Data Preprocessing

The data preprocessing workflow includes:

- Handling missing values
- Converting categorical variables
- Encoding categorical features
- Feature selection
- Train-test split
- Feature scaling where required

### 3. Exploratory Data Analysis

EDA was performed to understand customer behavior and identify patterns related to churn.

The analysis included:

- Distribution analysis
- Categorical feature analysis
- Churn-related patterns
- Feature relationships
- Model feature importance

### 4. Model Training

Two classification algorithms were trained:

- Logistic Regression
- Random Forest

### 5. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Model Comparison

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.8038 | 0.6091 | 0.8357 |
| Random Forest | 0.7683 | 0.5995 | 0.8204 |

Logistic Regression achieved higher accuracy, F1 Score, and ROC-AUC in this evaluation.

## Feature Importance

The Random Forest model was also used to examine feature importance.

The top important features identified were:

1. TotalCharges
2. Tenure
3. MonthlyCharges
4. Contract - Two year
5. InternetService - Fiber optic
6. PaymentMethod - Electronic check
7. Contract - One year
8. OnlineSecurity - Yes
9. Gender - Male
10. TechSupport - Yes

These features provide useful information about the factors contributing to the model's churn predictions.

## Business Insights

Based on the model analysis:

- Total customer charges were one of the most important features in the Random Forest model.
- Customer tenure was among the strongest predictive features.
- Monthly charges had a significant contribution to the model.
- Contract type, internet service, and payment method also contributed to churn prediction.
- Customer retention strategies can focus on identifying and supporting higher-risk customer segments.

## Streamlit Application

An interactive Streamlit application was developed for real-time customer churn prediction.

The application allows users to enter:

### Customer Information

- Senior Citizen
- Gender
- Tenure
- Partner
- Dependents
- Phone Service
- Internet Service
- Contract
- Monthly Charges

### Additional Services

- Online Security
- Device Protection
- Online Backup
- Tech Support
- Streaming TV
- Streaming Movies

### Billing Information

- Payment Method
- Paperless Billing

After entering the customer information, the application generates:

- Churn prediction
- Churn probability
- Customer risk message

## Project Structure

```text
Customer_Churn_Prediction/
│
├── app/
│   ├── app.py
│   └── requirements.txt
│
├── Data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── 01_data_loading.py
│
├── feature_columns.pkl
├── logistic_model.pkl
├── random_forest_model.pkl
├── scaler.pkl
├── .gitignore
└── README.md
```

## Saved Machine Learning Artifacts

The trained models and preprocessing objects are saved using Joblib:

logistic_model.pkl
random_forest_model.pkl
scaler.pkl
feature_columns.pkl

This allows the Streamlit application to load the trained models without retraining them every time.

How to Run the Project

1. Clone the Repository
git clone <your-github-repository-url>

2. Navigate to the Project
cd Customer_Churn_Prediction

3. Install Dependencies
pip install -r app/requirements.txt

4. Run the Streamlit Application
streamlit run app/app.py

The application will open in your browser at:

http://localhost:8501

Project Highlights
End-to-end Machine Learning project
Binary classification
Logistic Regression implementation
Random Forest implementation
Model comparison using multiple evaluation metrics
Feature importance analysis
Model persistence using Joblib
Interactive Streamlit application
Business-oriented churn insights
Future Improvements

Possible future enhancements include:

Hyperparameter tuning
Cross-validation
Advanced ensemble models
SHAP-based model explainability
Customer risk segmentation
Churn monitoring dashboard
Deployment to a cloud platform


Conclusion

This project demonstrates an end-to-end Machine Learning workflow for customer churn prediction,
from data preprocessing and exploratory analysis to model development, 
evaluation, model persistence, and deployment through an interactive Streamlit application.