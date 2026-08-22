import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# --------------------------------
# Load dataset
# --------------------------------

df = pd.read_csv(
    "D:/DataScience_Portfolio/customer_purchase_prediction/data/raw/customers.csv"
)

print("Dataset loaded successfully!")
print()


# --------------------------------
# Compare features by purchase
# --------------------------------

print("Average feature values by purchase decision:")
print()

feature_means = df.groupby("target")[
    [
        "age",
        "income",
        "website_visits",
        "previous_purchases",
        "discount"
    ]
].mean()

print(feature_means)
print()


# --------------------------------
# Create readable target labels
# --------------------------------

df["purchase_status"] = df["target"].map({
    0: "Did Not Purchase",
    1: "Purchased"
})


# --------------------------------
# Website visits vs purchase
# --------------------------------

website_means = df.groupby("purchase_status")["website_visits"].mean()

plt.figure(figsize=(8, 5))

website_means.plot(kind="bar")

plt.title("Average Website Visits by Purchase Decision")
plt.xlabel("Purchase Decision")
plt.ylabel("Average Website Visits")
plt.xticks(rotation=0)

plt.tight_layout()

website_chart = (
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "reports/"
    "website_visits_vs_purchase.png"
)

plt.savefig(
    website_chart,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Saved:", website_chart)


# --------------------------------
# Previous purchases vs purchase
# --------------------------------

previous_means = df.groupby("purchase_status")[
    "previous_purchases"
].mean()

plt.figure(figsize=(8, 5))

previous_means.plot(kind="bar")

plt.title("Average Previous Purchases by Purchase Decision")
plt.xlabel("Purchase Decision")
plt.ylabel("Average Previous Purchases")
plt.xticks(rotation=0)

plt.tight_layout()

previous_chart = (
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "reports/"
    "previous_purchases_vs_purchase.png"
)

plt.savefig(
    previous_chart,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Saved:", previous_chart)


# --------------------------------
# Income vs purchase
# --------------------------------

income_means = df.groupby("purchase_status")["income"].mean()

plt.figure(figsize=(8, 5))

income_means.plot(kind="bar")

plt.title("Average Income by Purchase Decision")
plt.xlabel("Purchase Decision")
plt.ylabel("Average Income")
plt.xticks(rotation=0)

plt.tight_layout()

income_chart = (
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "reports/"
    "income_vs_purchase.png"
)

plt.savefig(
    income_chart,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Saved:", income_chart)


# --------------------------------
# Discount vs purchase
# --------------------------------

discount_means = df.groupby("purchase_status")["discount"].mean()

plt.figure(figsize=(8, 5))

discount_means.plot(kind="bar")

plt.title("Average Discount by Purchase Decision")
plt.xlabel("Purchase Decision")
plt.ylabel("Average Discount")
plt.xticks(rotation=0)

plt.tight_layout()

discount_chart = (
    "D:/DataScience_Portfolio/"
    "customer_purchase_prediction/"
    "reports/"
    "discount_vs_purchase.png"
)

plt.savefig(
    discount_chart,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Saved:", discount_chart)


# --------------------------------
# Check files
# --------------------------------

print()
print("Checking generated reports:")

for file_path in [
    website_chart,
    previous_chart,
    income_chart,
    discount_chart
]:
    print(
        os.path.basename(file_path),
        "exists:",
        os.path.exists(file_path)
    )

print()
print("Feature analysis completed!")