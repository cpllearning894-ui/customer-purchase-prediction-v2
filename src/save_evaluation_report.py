import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------
# Load testing data
# --------------------------------

X_test = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "data/processed/"
    "X_test.csv"
)

y_test = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "data/processed/"
    "y_test.csv"
).squeeze()

print("Testing data loaded successfully!")


# --------------------------------
# Load trained model
# --------------------------------

model = joblib.load(
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "models/"
    "logistic_regression_model.pkl"
)

print("Trained model loaded successfully!")


# --------------------------------
# Create predictions
# --------------------------------

y_pred = model.predict(X_test)

print("Predictions created successfully!")


# --------------------------------
# Calculate evaluation metrics
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    y_pred
)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)


# --------------------------------
# Create report text
# --------------------------------

report_text = f"""
CUSTOMER PURCHASE PREDICTION
MODEL EVALUATION REPORT

----------------------------------------
MODEL PERFORMANCE
----------------------------------------

Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}


----------------------------------------
CLASSIFICATION REPORT
----------------------------------------

{report}


----------------------------------------
CONFUSION MATRIX
----------------------------------------

{matrix}


----------------------------------------
BUSINESS INTERPRETATION
----------------------------------------

This model predicts whether a customer is
likely to make a purchase.

The model achieved an accuracy of
{accuracy:.2%} on the testing dataset.

The model correctly identified
{recall:.2%} of customers who actually
made a purchase.
"""


# --------------------------------
# Save report
# --------------------------------

report_path = (
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "reports/"
    "model_evaluation_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(report_text)


print()
print("Evaluation report saved successfully!")
print(report_path)