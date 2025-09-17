import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
# Simulate customer transaction data
n_customers = 500
data = {
    'CustomerID': range(1, n_customers + 1),
    'Recency': np.random.randint(1, 365, size=n_customers),  # Days since last purchase
    'Frequency': np.random.poisson(lam=5, size=n_customers),  # Number of purchases
    'MonetaryValue': np.random.gamma(shape=2, scale=100, size=n_customers),  # Total spending
    'Churn': np.random.binomial(1, 0.2, size=n_customers)  # 0 = Not churned, 1 = Churned
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preparation ---
# (In this synthetic data, no cleaning is needed.  Real-world data would require more extensive cleaning)
# --- 3. Latent Class Analysis (Customer Segmentation) ---
# Prepare data for the model
X = df[['Recency', 'Frequency', 'MonetaryValue']]
# Fit Gaussian Mixture Model (GMM) for customer segmentation.  Number of components is arbitrary here; in real life, use techniques like BIC or AIC to determine optimal number of segments.
gmm = GaussianMixture(n_components=3, random_state=42) # 3 segments for demonstration
gmm.fit(X)
# Assign customers to segments
df['Segment'] = gmm.predict(X)
# --- 4. Churn Analysis by Segment ---
# Calculate churn rate for each segment
churn_by_segment = df.groupby('Segment')['Churn'].mean()
print("Churn Rate by Segment:")
print(churn_by_segment)
# --- 5. Visualization ---
# Plot churn rate by segment
plt.figure(figsize=(8, 6))
sns.barplot(x=churn_by_segment.index, y=churn_by_segment.values)
plt.title('Churn Rate by Customer Segment')
plt.xlabel('Segment')
plt.ylabel('Churn Rate')
plt.savefig('churn_by_segment.png')
print("Plot saved to churn_by_segment.png")
# Visualize segments in 2D (using PCA for dimensionality reduction if needed for higher dimensions)
# (Simplified for brevity; PCA would be beneficial for higher-dimensional data)
plt.figure(figsize=(8,6))
sns.scatterplot(x='Recency', y='MonetaryValue', hue='Segment', data=df)
plt.title('Customer Segments Visualization')
plt.savefig('customer_segments.png')
print("Plot saved to customer_segments.png")
#Further analysis (not shown for brevity):  You could delve deeper into the characteristics of each segment (e.g., average frequency, monetary value, etc.) to inform targeted retention strategies.