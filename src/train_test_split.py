import pandas as pd

from sklearn.model_selection import train_test_split


# --------------------------------
# Load processed dataset
# --------------------------------

df = pd.read_csv(
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "data/processed/"
    "customers_processed.csv"
)

print("Processed dataset loaded successfully!")
print()


# --------------------------------
# Separate features and target
# --------------------------------

X = df.drop("target", axis=1)

y = df["target"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print()


# --------------------------------
# Split data
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------
# Show split results
# --------------------------------

print("Training data:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print()

print("Testing data:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print()


# --------------------------------
# Check target distribution
# --------------------------------

print("Training target distribution:")
print(y_train.value_counts())
print()

print("Testing target distribution:")
print(y_test.value_counts())
print()


# --------------------------------
# Save split datasets
# --------------------------------

X_train.to_csv(
    "D:/DataScience_Portfolio/customer_purchase_prediction/data/processed/X_train.csv",
    index=False
)

X_test.to_csv(
    "D:/DataScience_Portfolio/customer_purchase_prediction/data/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "D:/DataScience_Portfolio/customer_purchase_prediction/data/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "D:/DataScience_Portfolio/customer_purchase_prediction/data/processed/y_test.csv",
    index=False
)

print("Train/test datasets saved successfully!")
print()
print("Train/test split completed!")