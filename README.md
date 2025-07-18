**Personal Expense Tracker and Forecasting Tool**

**PLS USE DARK MODE!!!

Project Streamlit Link: https://kcqueso-expense-tracker.streamlit.app/

This project is an interactive Streamlit application that allows users to upload personal expense data in CSV format, analyze trends, detect anomalies, and forecast future spending using machine learning models.

=================================================================

**Features:**

CSV Upload and Validation - Upload a CSV file with standardized columns and receive immediate validation for formatting and data integrity.

Visual Analysis
  - Monthly and daily spending trends
  - Category-based summaries
  - Spending breakdown by necessity

Machine Learning Models
  - Compares multiple regression models: Linear Regression, Random Forest, KNN, and XGBoost
  - Automatically selects the best-performing model based on Mean Squared Error

Forecasting
  - Predicts spending for the next week using recent data trends and lag-based features
  - Calculates the probability of overspending next week

Anomaly Detection - Flags unusually high or low transactions via Z-score 

Budgeting Suggestions - Offers personalized monthly caps for non-essential categories based on past 3 months of spending

Actual vs Predicted Comparison - Displays prediction accuracy for a sample of test data

=================================================================

**Required CSV Format**

Your CSV file must contain exactly these columns in the same order:

DATE, CATEGORY, DESCRIPTION, AMOUNT, PAYMENT_METHOD, MERCHANT, IS_NECESSARY

  - DATE must be in YYYY-MM-DD format

  - IS_NECESSARY must be either yes or no
  
  - No missing values allowed
  
  - You will be prompted if the format is incorrect.

Moreover, you can base on format.png so that it will be clearer

=================================================================

**Tech Stack**
  - Python
  
  - Streamlit
  
  - pandas, numpy
  
  - seaborn, matplotlib
  
  - scikit-learn
  
  - XGBoost
