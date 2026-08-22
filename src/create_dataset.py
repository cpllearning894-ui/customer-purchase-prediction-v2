import pandas as pd
import numpy as np

# Make the random data reproducible
np.random.seed(42)

# Number of customers
n_customers = 200

# Generate customer information
age = np.random.randint(18, 65, n_customers)

income = np.random.randint(20000, 100000, n_customers)

website_visits = np.random.randint(1, 21, n_customers)

previous_purchases = np.random.randint(0, 11, n_customers)

discount = np.random.randint(0, 31, n_customers)

# Calculate a purchase score
score = (
    0.03 * age
    + 0.00001 * income
    + 0.15 * website_visits
    + 0.4 * previous_purchases
    + 0.08 * discount
    + np.random.normal(0, 1, n_customers)
)

# Convert score into purchase decision
target = (score > 4.5).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "age": age,
    "income": income,
    "website_visits": website_visits,
    "previous_purchases": previous_purchases,
    "discount": discount,
    "target": target
})

# Save original dataset
df.to_csv(
    "D:/DataScience_Portfolio/customer_purchase_prediction/data/raw/customers.csv",
    index=False
)

print("Dataset created successfully!")
print()
print(df.head())
print()
print("Dataset shape:", df.shape)
print()
print("Target distribution:")
print(df["target"].value_counts())