# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
try:
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Explore the dataset
print("\nFirst five rows of the dataset:")
print(df.head())

print("\nData types and missing values:")
print(df.info())

print("\nChecking for missing values:")
print(df.isnull().sum())

# Clean dataset (No missing values in this case)
# df.dropna(inplace=True) # Uncomment if needed

# Basic Statistics
print("\nBasic Statistics:")
print(df.describe())

# Grouping: Mean of numerical values per species
grouped = df.groupby("target").mean()
print("\nMean values grouped by species:")
print(grouped)

# Replace numerical target with species names for visualization
df['species'] = df['target'].apply(lambda x: iris_data.target_names[x])

# === Visualizations ===
sns.set(style="whitegrid")

# Line Chart (for illustrative purpose: trend of first 10 samples' sepal length)
plt.figure(figsize=(8, 4))
plt.plot(df['sepal length (cm)'][:10], marker='o', linestyle='-')
plt.title("Sepal Length of First 10 Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar Chart: Average petal length by species
plt.figure(figsize=(8, 4))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram: Distribution of Sepal Width
plt.figure(figsize=(8, 4))
plt.hist(df['sepal width (cm)'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 4))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.show()

# === Observations ===
print("\nObservations:")
print("- Sepal width is normally distributed.")
print("- Petal length and sepal length show a positive correlation.")
print("- Average petal length varies distinctly between species.")
print("- Line chart shows variability even within same species in initial samples.")
