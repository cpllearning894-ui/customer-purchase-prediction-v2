Customer Purchase Prediction

Project Overview

This project is a Machine Learning application that predicts whether a customer is likely to purchase a product based on customer information.

The model analyzes customer behavior and characteristics such as:

- Age
- Income
- Website visits
- Previous purchases
- Discount

The prediction result helps businesses identify customers who are more likely to make a purchase.

---

Business Problem

Businesses often have many customers, but not all customers are equally likely to purchase a product.

This project uses Machine Learning to predict customer purchase behavior.

The goal is to help a business answer the question:

«Which customers are most likely to purchase?»

This information could help a business improve marketing campaigns and focus on potential customers.

---

Dataset

The project uses a customer dataset containing 200 customer records.

Features

Feature| Description
Age| Customer age
Income| Customer income
Website Visits| Number of website visits
Previous Purchases| Number of previous purchases
Discount| Discount offered to the customer
Target| Purchase decision

Target Values

- "0" = Did Not Purchase
- "1" = Purchased

The dataset was divided into:

- Training data: 80%
- Testing data: 20%

---

Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the customer dataset.

The analysis included:

- Dataset inspection
- Missing value checking
- Duplicate checking
- Statistical summary
- Target distribution analysis
- Feature comparison
- Data visualization

Generated reports include:

- Customer purchase distribution
- Website visits vs purchase
- Previous purchases vs purchase
- Income vs purchase
- Discount vs purchase

---

Data Preprocessing

The dataset preprocessing process included:

- Checking missing values
- Removing duplicate records
- Separating features and target
- Preparing processed data for Machine Learning

The processed dataset was saved in:

data/processed/customers_processed.csv

---

Machine Learning Model

The project uses a Logistic Regression model.

The model predicts whether a customer is likely to purchase.

Input features:

- Age
- Income
- Website visits
- Previous purchases
- Discount

Output:

- "0" → Did Not Purchase
- "1" → Purchased

The trained model is saved using Joblib.

models/logistic_regression_model.pkl

---

Model Evaluation

The model was evaluated using the test dataset.

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

Model Results

Metric| Score
Accuracy| 0.925
Precision| 0.9394
Recall| 0.9688
F1 Score| 0.9538

The model achieved strong performance on the testing dataset.

---

Prediction Example

Example customer:

Age: 35
Income: 65000
Website Visits: 12
Previous Purchases: 4
Discount: 15

Prediction:

Purchased

Purchase probability:

0.9943

---

FastAPI Deployment

The Machine Learning model is deployed using FastAPI.

The API allows users or applications to send customer information and receive a prediction.

Example API response:

{
  "prediction": 1,
  "prediction_label": "Purchased",
  "purchase_probability": 0.9943
}

---

Project Structure

customer_purchase_prediction/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── logistic_regression_model.pkl
│
├── notebooks/
│   ├── 01_eda.py
│   └── 02_feature_analysis.py
│
├── reports/
│
├── src/
│
├── tests/
│
├── requirements.txt
│
└── README.md

---

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- FastAPI
- Uvicorn

---

Installation

Clone the repository:

git clone YOUR_REPOSITORY_URL

Move into the project folder:

cd customer_purchase_prediction

Install the required libraries:

pip install -r requirements.txt

---

Run the Machine Learning Pipeline

Create the dataset:

python src/create_dataset.py

Perform Exploratory Data Analysis:

python notebooks/01_eda.py

Perform Feature Analysis:

python notebooks/02_feature_analysis.py

Preprocess the data:

python src/preprocess_data.py

Split the dataset:

python src/train_test_split.py

Train the model:

python src/train_model.py

Evaluate the model:

python src/evaluate_model.py

---

Run the API

Start the FastAPI server:

uvicorn api.main:app --reload

Open the API documentation:

http://127.0.0.1:8000/docs

---

Project Workflow

Customer Data
      ↓
Exploratory Data Analysis
      ↓
Feature Analysis
      ↓
Data Preprocessing
      ↓
Train/Test Split
      ↓
Machine Learning Model
      ↓
Model Evaluation
      ↓
Predictions
      ↓
FastAPI Deployment

---

Author

CPL Learning

Aspiring Data Scientist