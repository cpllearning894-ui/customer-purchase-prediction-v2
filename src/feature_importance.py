import os
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# --------------------------------
# Load training features
# --------------------------------

X_train = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "data/processed/"
    "X_train.csv"
)

print("Training features loaded successfully!")
print()

print("Feature names:")
print(list(X_train.columns))
print()


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
print()


# --------------------------------
# Get feature importance
# --------------------------------

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_[0]
})

# Calculate absolute importance
importance_df["Importance"] = importance_df["Coefficient"].abs()

# Sort from most important to least important
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("Feature importance:")
print(importance_df)
print()


# --------------------------------
# Create feature importance chart
# --------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.title("Customer Purchase Prediction - Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=30)

plt.tight_layout()


# --------------------------------
# Save chart
# --------------------------------

chart_path = (
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "reports/"
    "feature_importance.png"
)

plt.savefig(
    chart_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# --------------------------------
# Check saved file
# --------------------------------

print("Feature importance chart saved successfully!")
print(chart_path)

print(
    "Chart exists:",
    os.path.exists(chart_path)
)

print()
print("Feature importance analysis completed!")