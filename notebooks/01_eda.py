import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


# Load the raw dataset
df = pd.read_csv(
    "D:/DataScience_Portfolio/customer_purchase_prediction/data/raw/customers.csv"
)

print("Dataset loaded successfully!")
print()

# Show the first 5 rows
print("First 5 rows:")
print(df.head())
print()

# Show dataset information
print("Dataset information:")
print(df.info())
print()

# Show dataset shape
print("Dataset shape:")
print(df.shape)
print()

# Check missing values
print("Missing values:")
print(df.isnull().sum())
print()

# Check duplicate rows
print("Duplicate rows:")
print(df.duplicated().sum())
print()

# Show statistical summary
print("Statistical summary:")
print(df.describe())
print()

# Show target distribution
print("Target distribution:")
print(df["target"].value_counts())

# -----------------------------
# Visualization
# -----------------------------



print()
print("Creating target distribution chart...")

# Count customers in each class
target_counts = df["target"].value_counts().sort_index()

# Create bar chart
plt.figure(figsize=(8, 5))

target_counts.plot(kind="bar")

plt.title("Customer Purchase Distribution")
plt.xlabel("Purchase Decision")
plt.ylabel("Number of Customers")

plt.xticks(
    [0, 1],
    ["Did Not Purchase", "Purchased"],
    rotation=0
)

plt.tight_layout()

# Save chart
chart_path = (
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "reports/"
    "target_distribution.png"
)

plt.savefig(chart_path, dpi=150, bbox_inches="tight")

# Close the figure
plt.close()

print("Chart saved successfully!")
print("Chart location:")
print(chart_path)

print("Chart exists:", __import__("os").path.exists(chart_path))