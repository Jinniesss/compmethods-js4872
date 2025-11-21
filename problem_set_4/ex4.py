import pandas as pd
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("eeg-eye-state.csv")
data = data.drop(columns=["eyeDetection"])

for col in data.columns:
    mean = data[col].mean()
    std = data[col].std()
    data[col] = (data[col] - mean) / std

# PCA
pca = decomposition.PCA(n_components=2)
data_reduced = pca.fit_transform(data)
pc0 = data_reduced[:, 0]
pc1 = data_reduced[:, 1]
# Plot 1 
plt.scatter(pc0, pc1, c='b', alpha=0.5, s=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.show()
plt.savefig('ex4_pca1.png')

# remove outliers
mask = (pc0 > -2) & (pc0 < 4) & (pc1 > -1.5) & (pc1 < 1.5)
pc0_filtered = pc0[mask]
pc1_filtered = pc1[mask]
# Plot 2
plt.figure()
plt.scatter(pc0_filtered, pc1_filtered, c='b', alpha=0.5, s=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.show()
plt.savefig('ex4_pca2.png')

# k-means clustering (masked original data)
kmeans = KMeans(n_clusters=7, init='random')
kmeans.fit(data[mask])
labels = kmeans.labels_
centers = kmeans.cluster_centers_
centers_reduced = pca.transform(centers)

# Plot 3 with labeled clusters
plt.figure()
# Plot each cluster separately so we can add a legend entry per cluster
cmap = plt.get_cmap('tab10')
unique_labels = np.unique(labels)
for lab in unique_labels:
    lab_mask = labels == lab
    plt.scatter(pc0_filtered[lab_mask], pc1_filtered[lab_mask],
                color=cmap(int(lab) % 10), label=f'Cluster {int(lab)}', alpha=0.6, s=8)

# Plot centers
plt.title('K-means Clustering with 7 Clusters')
plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c='black', s=80, marker='x', label='Centers')
plt.legend(loc='best', fontsize='small')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.show()
plt.savefig('ex4_pca3.png')

# smaller points
plt.figure()
cmap = plt.get_cmap('tab10')
unique_labels = np.unique(labels)
for lab in unique_labels:
    lab_mask = labels == lab
    plt.scatter(pc0_filtered[lab_mask], pc1_filtered[lab_mask],
                color=cmap(int(lab) % 10), label=f'Cluster {int(lab)}', alpha=0.6, s=0.5)
plt.title('K-means Clustering with 7 Clusters')
plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c='black', s=80, marker='x', label='Centers')
plt.legend(loc='best', fontsize='small')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# plt.show()
plt.savefig('ex4_pca4.png')