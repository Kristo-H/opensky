import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ============================
# Section 1: Feature selection
# ============================

# plot using unfiltered data to detect anomalies
raw_data = pd.read_csv("data/opensky_states_snapshot.csv")
raw_features = raw_data[["geo_altitude", "baro_altitude", "velocity"]].dropna() # drop NaNs for KMeans
raw_data = raw_data.loc[raw_features.index] # keep only rows that are valid
raw_kmeans = KMeans(n_clusters=3, random_state=10)
raw_data["raw_cluster"] = raw_kmeans.fit_predict(raw_features)
plt.figure(figsize=(10,6))
plt.scatter(raw_data["geo_altitude"], raw_data["velocity"], c=raw_data["raw_cluster"], cmap="viridis", s=10)
plt.xlabel("Geo altitude (m)")
plt.xticks(range(0, 24000, 2000))
plt.yticks(range(0, 1400, 100))
plt.ylabel("Velocity (m/s)")
plt.title("Raw data, Clusters - geo_altitude vs velocity")
plt.colorbar(label="Cluster")
plt.show()


plt.figure(figsize=(10,6))
plt.scatter(raw_data["geo_altitude"], raw_data["velocity"], c=raw_data["raw_cluster"], cmap="viridis", s=10)
plt.xlabel("Baro altitude (m)")
plt.xticks(range(0, 24000, 2000))
plt.yticks(range(0, 1400, 100))
plt.ylabel("Velocity (m/s)")
plt.title("Raw data, Clusters - baro_altitude vs velocity")
plt.colorbar(label="Cluster")
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(raw_data["geo_altitude"], raw_data["baro_altitude"], c=raw_data["raw_cluster"], cmap="viridis", s=10)
plt.xlabel("Geo altitude (m)")
plt.xticks(range(0, 24000, 2000))
plt.yticks(range(0, 22000, 2000))
plt.ylabel("Baro altitude (m)")
plt.title("Raw data, Clusters - geo_altitude vs baro_altitude")
plt.colorbar(label="Cluster")
plt.show()

# Plot1. geo_altitude vs velocity shows that there are anomalies(unrealistic velocity/altitude combinations) (1200m/s at 11000m)
# Plot2. baro_altitude vs velocity shows also unrealistic combinations (20-25 m/s while at 4000m-7300m)
# Plot3. geo_altitude vs baro_altitude shows anomalies in differences between baro_altitude and geo_altitude -
# the points should be very close to the diagonal line, but some points have over 1000 meters of difference between baro_altitude and geo_altitude.
# filters need to be made -> removing anomalies in G1 at Section 4: Sanity checks


# After removing identified anomalies and filtered dataset in G1
# loading filtered dataset
filtered_data = pd.read_csv("data/filtered_final_data.csv")

print(filtered_data.head())
print(filtered_data.info())



# =============================================
# Section 2: Data normalization & preprocessing
# =============================================

#Clusters after normalizaton:
# features combinations for plots
feature_combinations = [
    ("geo_altitude", "velocity"),
    ("baro_altitude", "velocity"),
    ("geo_altitude", "baro_altitude")
]

#normalizing the features
minmaxscaler = MinMaxScaler()
features_scaled = minmaxscaler.fit_transform(filtered_data[["geo_altitude", "baro_altitude", "velocity"]])
filtered_data_scaled = pd.DataFrame(features_scaled, columns=["geo_altitude", "baro_altitude", "velocity"])

# features for clustering
features_clustering = filtered_data[["geo_altitude", "baro_altitude", "velocity"]]



# =========================================
# Section 3: K-means clustering exploration
# =========================================

#K-means
k_means = KMeans(n_clusters=3, random_state=10)
clusters = k_means.fit_predict(features_clustering)

# add cluster labels to data
filtered_data["cluster"] = clusters

#plot clusters before normalization
plt.figure(figsize=(10,6))
plt.scatter(filtered_data["geo_altitude"], filtered_data["velocity"], c=filtered_data["cluster"], cmap="viridis", s=10)
plt.xlabel("Geo altitude (m)")
plt.xticks(range(0, 24000, 2000))
plt.yticks(range(0, 350, 25))
plt.ylabel("Velocity (m/s)")
plt.title("K-means Clusters - geo_altitude vs velocity")
plt.colorbar(label="Cluster")
plt.show()


plt.figure(figsize=(10,6))
plt.scatter(filtered_data["baro_altitude"], filtered_data["velocity"], c=filtered_data["cluster"], cmap="viridis", s=10)
plt.xlabel("Baro altitude (m)")
plt.xticks(range(0, 24000, 2000))
plt.yticks(range(0, 350, 25))
plt.ylabel("Velocity (m/s)")
plt.title("Clusters - baro_altitude vs velocity")
plt.colorbar(label="Cluster")
plt.show()



plt.figure(figsize=(10,6))
plt.scatter(filtered_data["geo_altitude"], filtered_data["baro_altitude"], c=filtered_data["cluster"], cmap="viridis", s=10)
plt.xlabel("Geo altitude (m)")
plt.xticks(range(0, 24000, 2000))
plt.yticks(range(0, 24000, 2000))
plt.ylabel("Baro altitude (m)")
plt.title("Clusters - geo_altitude vs baro_altitude")
plt.colorbar(label="Cluster")
plt.show()



# explanation for each graph before normalization: 
# Graph 1. Geo_altitude vs velocity - dense clusters show normal flight behavior, there are some anomalies(points that are far from clusters).
# there are many flights with altitude 9500 where only the velocity grows (vertical line). 
# there are five points which are not close to dense clusters: P1(11000,320), P2(11500, 309), P3(16300, 3), P4(19700, 8), P5(21000, 3).
# meaning of these points: P3, P4, P5 are balloons. P1 and P2 are fast aircrafts. 
# those are not anomalies, they are real aircrafts, so no need to remove.

# Graph 2. Baro_altitude vs velocity
# there are no anomalies, vertical lines between baro_altitude 9700- 13800 and velocity 190-285
# are caused by different aircraft flying at different velocities but at the same pressure altitude

# Graph 3. Geo_altitude vs baro_altitude
# no anomalies, diagonal line shows geo_altitude and baro_altitude are correlated.
# no overlapping between features.

#flag points for geo_altitude vs velocity:
plt.figure(figsize=(10,6))
plt.scatter(filtered_data["geo_altitude"], filtered_data["velocity"], c=filtered_data["cluster"], cmap="viridis", s=10)
anomalies = {
    "P1": (11000,320),
    "P2": (11500,309),
    "P3": (16300,3),
    "P4": (19700,8),
    "P5": (21000,3)
}

# visualize the coordinates
for anomaly, (x,y) in anomalies.items():
    plt.text(x+50, y+5, anomaly)
plt.xlabel("Geo altitude (m)")
plt.xticks(range(0, 24000, 2000))
plt.yticks(range(0, 350, 25))
plt.ylabel("Velocity (m/s)")
plt.title("Clusters - geo_altitude vs velocity with flagged points")
plt.colorbar(label="Cluster")
plt.show()



# plot clusters on normalized data
k_means = KMeans(n_clusters=3, random_state=10)
filtered_data_scaled["cluster"] = k_means.fit_predict(filtered_data_scaled[["geo_altitude","baro_altitude","velocity"]])


# plot all K-means feature pairs
for x, y in feature_combinations:
    plt.figure(figsize=(10,6))
    plt.scatter(filtered_data_scaled[x], filtered_data_scaled[y], c=filtered_data_scaled["cluster"], cmap="viridis", s=10)
    plt.xlabel(f'{x} (normalized)')
    plt.ylabel(f'{y} (normalized)')
    plt.title(f'Clusters: {x} vs {y}')
    plt.colorbar(label="Cluster")
    plt.show()

# explanation for each graph after normalization:
# Graph 1. Normalized geo_altitude vs normalized velocity - 45 degree separation between clusters is good, 
# it shows combination of altitude and velocity (a balanced distance) instead of just altitude(a vertical line),
# values moved to another clusters due to now being grouped based on balanced features.

# Graph 2. Normalized baro_altitude vs normalized velocity - balanced distance between clusters, values moved to another clusters.

# Graph 3. Normalized geo_altitude vs normalized baro_altitude - there is now overlapping between the clusters, 
# because the two altitudes are highly correlated and always cluster along a diagonal line.


# ========================================
# Section 4: DBSCAN clustering exploration
# ========================================

# Clustering using DBSCAN
# standardize features
standardscaler = StandardScaler()
features_scaled_2 = standardscaler.fit_transform(filtered_data[["geo_altitude", "baro_altitude", "velocity"]])
filtered_data_scaled2 = pd.DataFrame(features_scaled_2, columns=["geo_altitude", "baro_altitude", "velocity"])

#finding epsilon for dbscan using nearest neighbors
X = features_scaled_2
min_samples = 5
#nearest neighbors
nearestneighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = nearestneighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

#distances to nearest k-th neighbor
k_distances = np.sort(distances[:, min_samples-1])

plt.figure(figsize=(8,5))
plt.plot(k_distances)
plt.xlabel("Points sorted by distance")
plt.ylabel(f'{min_samples}-NN distance')
plt.title("k-distance graph to choose eps for DBSCAN")
plt.show()
#conclusion: by using the elbow method on the graph above, it shows epsilon to be 0.3


# plotting all clusters with dbscan
#feature combinations:
feature_combinations = [
    ("baro_altitude", "velocity"),
    ("geo_altitude", "baro_altitude")
]

eps = 0.3
min_samples = 5

# plot ("geo_altitude", "velocity") pair in original units (m and m/s)
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(filtered_data_scaled2[["geo_altitude", "velocity"]])
plt.figure(figsize=(10,6))
plt.scatter(filtered_data["geo_altitude"], filtered_data["velocity"], c=clusters, cmap="viridis", s=10)
plt.xlabel("Geo altitude (m)")
plt.ylabel("Velocity (m/s")
plt.title("DBSCAN Clusters - geo_altitude vs velocity")
plt.colorbar(label="Cluster (-1 = noise)")
plt.show()



#plot other 2 cluster pairs with standardized units
for x, y in feature_combinations:
    X_combination = filtered_data_scaled2[[x, y]].values
    #DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_combination)

    #store cluster labels
    filtered_data_scaled2[f'dbscan_{x}_{y}'] = clusters

    #plot cluster pairs in standardized units
    plt.figure(figsize=(10,6))
    plt.scatter(filtered_data_scaled2[x], 
                filtered_data_scaled2[y], 
                c=clusters, cmap="viridis", s=10)
    plt.xlabel(f'{x} (standardized)')
    plt.ylabel(f'{y} (standardized)')
    plt.title(f'DBSCAN Clusters - {x} vs {y}')
    plt.colorbar(label="Cluster (-1 = noise)")
    plt.show()


filtered_data["dbscan_cluster"] = dbscan.fit_predict(features_scaled_2)
print(filtered_data["dbscan_cluster"].tail(5))


# =====================================
# Section 5: Cluster analysis & reports
# =====================================

# Conclusion of K-Means vs DBSCAN
# Graph 1. geo_altitude vs velocity cluster on standardized data:
# dbscan found 3 more points as potential outliers, in total 8. 
# with k-means found 5 in total.

# Graph 2. baro_altitude vs velocity cluster on standardized data:
# dbscan found 4 more points as potential outliers, in total 9.
# with k-means found 5 in total.

# Graph 3. geo_altitude vs baro_altitude cluster on standardized data:
# dbscan found 3 points in total as potential outliers.
# with k-means found 3 in total.

# Explored multiple clustering methods (K-means, DBSCAN).
# Found out that K-means always forms clusters and it won't flag some outliers, 
# therefore unusual points can still get grouped in a cluster.
# Found out that DBSCAN identifies dense clusters and labels points that do not fit as noise. So it shows more outliers.
# DBSCAN detected more unusual flights, K-Means is good for grouping flights.
