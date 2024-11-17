import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/creditcard.csv')

# Display the first few rows
print(df.head())

# Basic information about the dataset
df.info()

# Summary statistics
print(df.describe())

# Check for class imbalance
print(df['Class'].value_counts())

# Plot the distribution of 'Amount'
plt.figure(figsize=(10, 5))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Transaction Amount Distribution')
plt.show()

# Plot the distribution of 'Time'
plt.figure(figsize=(10, 5))
sns.histplot(df['Time'], bins=50, kde=True)
plt.title('Transaction Time Distribution')
plt.show()

# Step 6: Plot the heatmap
plt.figure(figsize=(20, 15))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Check for missing values
print(df.isnull().sum())

# Separate fraud and non-fraud data
fraud_data = df[df['Class'] == 1]
non_fraud_data = df[df['Class'] == 0]

# Plot the amount distribution for fraud vs non-fraud
plt.figure(figsize=(10, 5))
sns.histplot(non_fraud_data['Amount'], bins=50, color='blue', kde=True, label='Non-Fraud')
sns.histplot(fraud_data['Amount'], bins=50, color='red', kde=True, label='Fraud')
plt.legend()
plt.title('Transaction Amount Distribution by Class')
plt.show()

# Plot the time distribution for fraud vs non-fraud
plt.figure(figsize=(10, 5))
sns.histplot(non_fraud_data['Time'], bins=50, color='blue', kde=True, label='Non-Fraud')
sns.histplot(fraud_data['Time'], bins=50, color='red', kde=True, label='Fraud')
plt.legend()
plt.title('Transaction Time Distribution by Class')
plt.show()
