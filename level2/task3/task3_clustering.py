# ============================================
# Level 2 - Task 3: Clustering (Unsupervised Learning)
# Codveda Data Science Internship
# ============================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 2. Load dataset
df = pd.read_csv("1) iris.csv")

print("Dataset loaded successfully")
print(df.head())

# 3. Remove target column if present
if "species" in df.columns:
    df = df.drop(columns=["species"])

# 4. Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 5. Elbow Method
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(K, inertia, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# 6. Silhouette Score
sil_scores = {}

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels)

print("\nSilhouette Scores:")
for k, score in sil_scores.items():
    print(f"k = {k}: {score:.3f}")

# 7. Apply K-Means with optimal K (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# 8. Dimensionality Reduction (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# 9. Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="viridis",
    data=df
)
plt.title("K-Means Clustering Visualization (PCA)")
plt.show()
