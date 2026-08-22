import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression


# --------------------------------
# Load training data
# --------------------------------

X_train = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "data/processed/"
    "X_train.csv"
)

y_train = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "data/processed/"
    "y_train.csv"
)


print("Training data loaded successfully!")
print()

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print()


# --------------------------------
# Convert target to 1D
# --------------------------------

y_train = y_train.squeeze()


# --------------------------------
# Create Logistic Regression model
# --------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# --------------------------------
# Train the model
# --------------------------------

model.fit(X_train, y_train)


print("Model training completed successfully!")
print()


# --------------------------------
# Show model information
# --------------------------------

print("Model coefficients:")

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_[0]
})

print(feature_importance)
print()

print("Intercept:")
print(model.intercept_[0])
print()


# --------------------------------
# Save trained model
# --------------------------------

model_path = (
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "models/"
    "logistic_regression_model.pkl"
)

joblib.dump(model, model_path)


print("Model saved successfully!")
print(model_path)

print()
print("Machine Learning training completed!")