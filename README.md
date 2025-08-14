# İndoLike - Task 1: Customer Segmentation

This repository contains **Task 1: Customer Segmentation** for **İndoLike**. The goal is to group customers based on their purchasing behavior (`Quantity` and `TotalAmount`) using K-Means clustering, to better understand customer patterns and inform marketing strategies.

## Dataset

The dataset is from an online retail store and contains:

- `Invoice`: Invoice number
- `StockCode`: Product code
- `Description`: Product description
- `Quantity`: Quantity of the product purchased
- `InvoiceDate`: Date and time of the invoice
- `Price`: Price of the product
- `CustomerID`: Unique identifier for each customer
- `Country`: Country of the customer

> Note: The dataset contains missing values and some formatting issues, which are handled during preprocessing.

## Steps

1. **Data Loading and Cleaning**
   - Load CSV and handle problematic rows.
   - Rename columns for easier access.
   - Convert `Price` and `Quantity` to numeric and remove NaNs.

2. **Feature Engineering**
   - Calculate `TotalAmount = Quantity * Price` for each transaction.

3. **Customer-level Aggregation**
   - Aggregate data by `CustomerID` to get total quantity and total amount spent.

4. **Standardization**
   - Scale `Quantity` and `TotalAmount` using `StandardScaler`.

5. **K-Means Clustering**
   - Apply `KMeans` with 4 clusters.
   - Assign cluster labels to each customer.

6. **Visualization**
   - Scatter plot of `Quantity` vs `TotalAmount` colored by cluster.

## Example Code

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv("online_retail.csv", sep=';', encoding='ISO-8859-1', on_bad_lines='skip')
df.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)
df['TotalAmount'] = pd.to_numeric(df['Quantity'], errors='coerce') * pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Quantity', 'Price', 'TotalAmount'])

# Customer-level aggregation
customer_df = df.groupby('CustomerID').agg({
    'Quantity': 'sum',
    'TotalAmount': 'sum'
}).reset_index()

# Standardize features
X = customer_df[['Quantity', 'TotalAmount']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
customer_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualization
plt.figure(figsize=(8,6))
plt.scatter(customer_df['Quantity'], customer_df['TotalAmount'], c=customer_df['Cluster'], cmap='viridis')
plt.xlabel('Quantity')
plt.ylabel('TotalAmount')
plt.title('Customer Clusters - İndoLike')
plt.show()
