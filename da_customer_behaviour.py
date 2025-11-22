import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("customer_analysis_project_data.csv")
df.head()

df['Recency'] = df['LastPurchaseDays']
df['Frequency'] = df['VisitFrequency']
df['Monetary'] = df['TotalSpend']

rfm = df[['CustomerID','Recency','Frequency','Monetary']]
rfm.head()
print(rfm.head())

scaler = StandardScaler()
scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])
print(scaled)

wcss = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,10), wcss)
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled)
df.head()

sns.scatterplot(x='Frequency', y='Monetary', hue='Cluster', data=df, palette='viridis')
plt.show()

cluster_profile = df.groupby('Cluster').mean(numeric_only=True)
def assign_offer(cluster):
    if cluster == 0:
        return "Premium Loyalty Rewards"
    elif cluster == 1:
        return "Discount Coupon (20%)"
    else:
        return "Win-back Offer (Free Delivery)"

df['RecommendedOffer'] = df['Cluster'].apply(assign_offer)
df[['CustomerID','Cluster','RecommendedOffer']]
df.to_csv("customer_offers_output.csv", index=False)

