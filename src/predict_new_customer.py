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
# Create new customer data
# --------------------------------

new_customer = pd.DataFrame({
    "age": [35],
    "income": [65000],
    "website_visits": [12],
    "previous_purchases": [4],
    "discount": [15]
})

print("New customer information:")
print(new_customer)
print()


# --------------------------------
# Predict purchase
# --------------------------------

prediction = model.predict(new_customer)

purchase_probability = model.predict_proba(new_customer)

print("Prediction result:")
print(prediction)
print()

print("Purchase probabilities:")
print(purchase_probability)
print()


# --------------------------------
# Show readable result
# --------------------------------

if prediction[0] == 1:
    print("FINAL RESULT:")
    print("Customer is likely to PURCHASE!")
else:
    print("FINAL RESULT:")
    print("Customer is NOT likely to purchase.")