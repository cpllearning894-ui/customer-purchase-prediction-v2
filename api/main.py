from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# Create FastAPI application
app = FastAPI()


# Find the main project folder automatically
BASE_DIR = Path(__file__).resolve().parent.parent

# Build the model path automatically
model_path = BASE_DIR / "models" / "logistic_regression_model.pkl"

# Load trained machine learning model
model = joblib.load(model_path)


# Define customer data structure
class Customer(BaseModel):
    age: int
    income: int
    website_visits: int
    previous_purchases: int
    discount: int


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Purchase Prediction API is working!"
    }


# Prediction endpoint
@app.post("/predict")
def predict(customer: Customer):

    # Convert customer data into DataFrame
    customer_data = pd.DataFrame([{
        "age": customer.age,
        "income": customer.income,
        "website_visits": customer.website_visits,
        "previous_purchases": customer.previous_purchases,
        "discount": customer.discount
    }])

    # Make prediction
    prediction = int(model.predict(customer_data)[0])

    # Get probability of purchase
    probability = float(
        model.predict_proba(customer_data)[0][1]
    )

    # Create readable prediction label
    if prediction == 1:
        prediction_label = "Purchased"
    else:
        prediction_label = "Did Not Purchase"

    # Return prediction result
    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "purchase_probability": round(probability, 4)
    }