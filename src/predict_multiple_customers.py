import pandas as pd
import joblib


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
# Create multiple new customers
# --------------------------------

new_customers = pd.DataFrame({
    "age": [25, 35, 45, 55, 30],
    "income": [30000, 65000, 80000, 50000, 40000],
    "website_visits": [2, 12, 15, 5, 8],
    "previous_purchases": [0, 4, 7, 2, 1],
    "discount": [5, 15, 20, 10, 8]
})

print("New customers:")
print(new_customers)
print()


# --------------------------------
# Make predictions
# --------------------------------

predictions = model.predict(new_customers)

probabilities = model.predict_proba(new_customers)


# --------------------------------
# Add prediction results
# --------------------------------

new_customers["prediction"] = predictions

new_customers["purchase_probability"] = probabilities[:, 1]


# --------------------------------
# Create readable prediction label
# --------------------------------

new_customers["prediction_label"] = new_customers["prediction"].map({
    0: "Did Not Purchase",
    1: "Purchased"
})


# --------------------------------
# Show final results
# --------------------------------

print("FINAL PREDICTION RESULTS:")
print()

print(new_customers)
print()


# --------------------------------
# Save prediction results
# --------------------------------

output_path = (
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "reports/"
    "multiple_customer_predictions.csv"
)

new_customers.to_csv(
    output_path,
    index=False
)

print("Prediction results saved successfully!")
print(output_path)