import pandas as pd


# --------------------------------
# Load raw dataset
# --------------------------------

df = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "data/raw/customers.csv"
)

print("Raw dataset loaded successfully!")
print()


# --------------------------------
# Check missing values
# --------------------------------

print("Missing values:")
print(df.isnull().sum())
print()


# --------------------------------
# Remove duplicate rows
# --------------------------------

df = df.drop_duplicates()

print("Duplicates removed.")
print("Dataset shape after removing duplicates:")
print(df.shape)
print()


# --------------------------------
# Separate features and target
# --------------------------------

X = df.drop("target", axis=1)

y = df["target"]

print("Features:")
print(X.head())
print()

print("Target:")
print(y.head())
print()


# --------------------------------
# Save processed dataset
# --------------------------------

processed_path = (
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "data/processed/"
    "customers_processed.csv"
)

df.to_csv(
    processed_path,
    index=False
)

print("Processed dataset saved successfully!")
print(processed_path)


# --------------------------------
# Final dataset information
# --------------------------------

print()
print("Final dataset shape:")
print(df.shape)

print()
print("Preprocessing completed!")