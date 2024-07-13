import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA


data, _ = make_blobs(n_samples=300, n_features=5, centers=5, random_state=42)

print("Shape of data:", data.shape)
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

plt.scatter(data_reduced[:, 0], data_reduced[:, 1])
plt.title("Visualisation des données par PCA")
plt.show()

kmeans_random = KMeans(n_clusters=5, init='random', random_state=42)
kmeans_random.fit(data)
labels_random = kmeans_random.labels_

kmeans_plus = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans_plus.fit(data)
labels_plus = kmeans_plus.labels_

score_random = calinski_harabasz_score(data, labels_random)
score_plus = calinski_harabasz_score(data, labels_plus)
print("Calinski-Harabasz Score (random):", score_random)
print("Calinski-Harabasz Score (k-means++):", score_plus)

inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Méthode de l\'ébow')
plt.show()

best_init = 'k-means++' if score_plus > score_random else 'random'
print("Meilleure model de clustering:", best_init)

centers = kmeans_plus.cluster_centers_ if best_init == 'k-means++' else kmeans_random.cluster_centers_
centers_reduced = pca.transform(centers)

plt.scatter(data_reduced[:, 0], data_reduced[:, 1], label='Data')
plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c='red', label='Centers')
plt.title("Données et centres des clusters")
plt.legend()
plt.show()

pca_full = PCA()
data_pca = pca_full.fit_transform(data)
print("Nouvelle matrice des observation après PCA:", data_pca)

print("valeur propres", pca_full.explained_variance_)
print("Vecteurs propres", pca_full.components_)

inertie_axes = pca_full.explained_variance_ratio_
print("Inertie expliquée par chaque axe:", inertie_axes)

print("Somme des inerties", np.sum(inertie_axes))

plt.scatter(data_pca[:, 0], data_pca[:, 1], label='Data')
plt.scatter(centers_reduced[:, 0], centers_reduced[:, 1], c='red', marker='X' ,label='Centers')
plt.title("Données et centres des clusters après PCA")
plt.legend()
plt.show()


