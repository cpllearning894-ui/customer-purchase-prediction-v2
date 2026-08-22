import os
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
# Create confusion matrix
# --------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print()
print("Confusion Matrix:")
print(cm)


# --------------------------------
# Create visualization
# --------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Did Not Purchase",
        "Purchased"
    ]
)

display.plot()

plt.title("Customer Purchase Prediction - Confusion Matrix")

plt.tight_layout()


# --------------------------------
# Save chart
# --------------------------------

chart_path = (
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "reports/"
    "confusion_matrix.png"
)

plt.savefig(
    chart_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------
# Check file
# --------------------------------

print()
print("Confusion matrix chart saved successfully!")
print(chart_path)

print(
    "Chart exists:",
    os.path.exists(chart_path)
)