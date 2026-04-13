import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import time

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def kcenter(P, k):
    print("\nRunning k-center (Farthest First)...")

    # Random first center
    centers = [P[np.random.randint(len(P))]]

    for _ in range(1, k):
        distances = []

        for point in P:
            # distance to nearest center
            min_dist = min(euclidean_distance(point, c) for c in centers)
            distances.append(min_dist)

        # pick farthest point
        next_center = P[np.argmax(distances)]
        centers.append(next_center)

    print("k-center completed!")
    return np.array(centers)


def kmeans_plus_plus_init(P, k):
    print("\nRunning K-Means++ initialization...")

    n = len(P)
    centers = []

    # Step 1: pick first center randomly
    centers.append(P[np.random.randint(n)])

    for _ in range(1, k):
        distances = []

        for point in P:
            min_dist = min(np.linalg.norm(point - c)**2 for c in centers)
            distances.append(min_dist)

        distances = np.array(distances)
        probabilities = distances / distances.sum()

        # choose next center based on probability
        next_center = P[np.random.choice(n, p=probabilities)]
        centers.append(next_center)

    print("K-Means++ initialization complete!")
    return np.array(centers)

def kmeansObj(P, C):
    print("\nComputing k-means objective...")

    total_distance = 0

    for point in P:
        min_dist = min(np.linalg.norm(point - center)**2 for center in C)
        total_distance += min_dist

    avg_distance = total_distance / len(P)

    print("k-means objective value:", avg_distance)
    return avg_distance

# Load dataset
path = "Datasets/Assignment 4- datasets/Assignment 4- datasets/Q1- UCI Spam clustering/spambase.data"

print("Loading dataset...")
df = pd.read_csv(path, header=None)

# Check shape
print("Dataset shape:", df.shape)

# Separate features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

# Normalize data
print("Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 2
centers_kcenter = kcenter(X_scaled, k)
print("k-center centers shape:", centers_kcenter.shape)

centers_kpp = kmeans_plus_plus_init(X_scaled, k)
print("kmeans++ centers shape:", centers_kpp.shape)

obj_value = kmeansObj(X_scaled, centers_kpp)

print("\nApplying PCA (Dimensionality Reduction)...")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("PCA complete! New shape:", X_pca.shape)

print("Scaling complete!")

print("Step 3 successful ✅")


print("Step 3 successful ✅")

# ================= ELBOW METHOD START =================
print("\nFinding optimal K using Elbow Method...")

inertia = []
K_range = range(1, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Plot graph
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
plt.savefig("elbow.png")
print("Elbow graph saved as elbow.png")
# ================= ELBOW METHOD END =================

print("\nApplying K-Means clustering...")

# Choose number of clusters (2: spam / not spam)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)

# Fit model
kmeans.fit(X_pca)

# Get cluster labels
clusters = kmeans.labels_

print("Clustering complete!")
print("Cluster labels sample:", clusters[:10])


print("\nEvaluating clustering...")

# Option 1: direct mapping
acc1 = accuracy_score(y, clusters)

# Option 2: reverse mapping
acc2 = accuracy_score(y, 1 - clusters)

# Choose best
best_acc = max(acc1, acc2)

print("Accuracy (direct):", acc1)
print("Accuracy (reversed):", acc2)
print("Best Accuracy:", best_acc)

print("\nConfusion Matrix (Best Mapping):")

# Choose correct mapping
if acc1 > acc2:
    final_clusters = clusters
else:
    final_clusters = 1 - clusters

cm = confusion_matrix(y, final_clusters)
print(cm)


print("\nPlotting clusters...")

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_clusters, cmap='viridis', s=10)

plt.title("K-Means Clustering (PCA Reduced Data)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.savefig("clusters.png")
print("Cluster plot saved as clusters.png")


print("\n========== FINAL ASSIGNMENT PIPELINE ==========")

k = 2
k1 = 5

# -------- Case 1 --------
start = time.time()
centers_kcenter = kcenter(X_scaled, k)
end = time.time()

print("\nCase 1: k-center(P, k)")
print("Time taken:", end - start)


# -------- Case 2 --------
centers_kpp = kmeans_plus_plus_init(X_scaled, k)
obj1 = kmeansObj(X_scaled, centers_kpp)

print("\nCase 2: kmeans++(P, k)")
print("Objective value:", obj1)


# -------- Case 3 --------
centers_kcenter_k1 = kcenter(X_scaled, k1)
centers_kpp_from_k1 = kmeans_plus_plus_init(centers_kcenter_k1, k)
obj2 = kmeansObj(X_scaled, centers_kpp_from_k1)

print("\nCase 3: kcenter(P, k1) -> kmeans++(X, k)")
print("Objective value:", obj2)

print("\nExecution completed successfully ✅")