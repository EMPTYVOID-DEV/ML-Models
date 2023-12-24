from pandas import read_csv
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np


data = read_csv("../datasets/Country-data.csv")

countries = data["country"]

data = data.drop("country", axis=1)

scaler = StandardScaler()

standardizedData = scaler.fit_transform(data)

hierarchical_cluster_data_portion = standardizedData[0:50, :]

max_cluster_size = 8

silhouette_scores = []

for i in range(2, max_cluster_size + 1):
    hierarchical_cluster = AgglomerativeClustering(n_clusters=i)
    currentLabels = hierarchical_cluster.fit_predict(hierarchical_cluster_data_portion)
    currentScore = silhouette_score(hierarchical_cluster_data_portion, currentLabels)
    silhouette_scores.append(currentScore)

# get the number of clusters with the best silhouette_score
optimal_cluster_size = int(np.argmax(silhouette_scores) + 2)

kmeans = KMeans(n_clusters=optimal_cluster_size, random_state=42)
kmeans.fit(standardizedData)

# clustering results
for i in range(0, len(countries)):
    print(
        "The cluster of the countery "
        + countries[i]
        + " is : "
        + str(kmeans.labels_[i])
    )
