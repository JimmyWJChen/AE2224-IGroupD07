import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
# import vallenae as ae

# standard threshold for AE 34dB, we have at 45dB -> thus missing IDs

def standarize(datapoints):
    X_norm = datapoints - datapoints.mean()
    X_norm = X_norm/np.std(datapoints)
    X_norm.dropna(axis=1, inplace=True)
    return X_norm

def PCA(datapoints, n):
    datapoints = standarize(datapoints[["amplitude", "duration", "energy", "signal_strength", "counts"]])
    print(datapoints)
    C = np.cov(datapoints.astype(float), rowvar=False)
    evalues, evectors = np.linalg.eigh(C)
    sortedValues = evalues[:]
    sortedVectors = evectors[:]
    usedValues = [sortedValues[-i-1] for i in range(n)]
    usedVectors = [sortedVectors[-i-1] for i in range(n)]
    usedVectors = np.transpose(np.array(usedVectors))
    X_pca = np.matmul(datapoints, usedVectors)
    totalVariance = np.sum(evalues)
    var = usedValues/totalVariance
    return X_pca, var

def train_kmeans(x_train, n_clusters):
    x_train = np.sort(x_train.reshape(-1))
    cluster_model = KMeans(
        n_clusters=n_clusters, n_init=10).fit(
        x_train.reshape(-1, 1))
    _, indx = np.unique(cluster_model.labels_, return_index=True)
    lookup_labels = [cluster_model.labels_[i] for i in sorted(indx)]
    return cluster_model, lookup_labels, n_clusters

def train_hierarchical(x_train, n_clusters, linkage='ward'):
    x_train = np.sort(x_train.reshape(-1))
    cluster_model = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage).fit(
        x_train.reshape(-1, 1))
    _, indx = np.unique(cluster_model.labels_, return_index=True)
    lookup_labels = [cluster_model.labels_[i] for i in sorted(indx)]
    return cluster_model, lookup_labels, n_clusters

def predict(cluster_model, lookup_labels, data_dict, cluster_method):
    clusters_dict = {f'a{i + 1}': [] for i in range(6)}
    for i in range(len(data_dict)):
        if cluster_method == 'kmeans':
            pred = cluster_model.predict(
                np.array(data_dict[f'a{i + 1}']['strains']).squeeze().reshape(-1, 1))
            labels = []
            for j in range(len(pred)):
                labels.append(np.where(lookup_labels == pred[j])[0])
            clusters_dict[f'a{i + 1}'].extend(sorted(labels))
        else:
            labels = cluster_model.fit_predict(
                np.array(data_dict[f'a{i + 1}']['strains']).squeeze().reshape(-1, 1))
            clusters_dict[f'a{i + 1}'].extend(sorted(labels))

    return clusters_dict

if __name__=="__main__":
    datapoints = di.filterPrimaryDatabase(di.getPrimaryDatabase("PCLO", 3), "PCLO", 3)
    datapoints = datapoints[datapoints['channel'] == 2]
    X_cluster = datapoints[['amplitude', 'counts']].to_numpy()
    print(X_cluster)
    cluster_model, lookup_labels, n_clusters = train_kmeans(X_cluster, 4)
    clusters = predict(cluster_model, lookup_labels, 0, 'kmeans')
    # data_compressed, var = PCA(datapoints, 2)
    # plt.scatter(data_compressed[0], data_compressed[1])
    # plt.show()
    # print(data_compressed)
    # print(var)
