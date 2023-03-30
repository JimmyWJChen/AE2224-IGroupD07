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

def clustering_kmeans(x_train, n_clusters):
    cluster_model = KMeans(n_clusters=n_clusters, n_init=10).fit(x_train)
    x_train['cluster'] = cluster_model.predict(x_train)
    return x_train

def clustering_hierarchical(x_train, n_clusters, linkage='ward'):
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(x_train)
    x_train['cluster'] = cluster_model.labels_
    return x_train

if __name__=="__main__":
    datapoints = di.filterPrimaryDatabase(di.getPrimaryDatabase("PCLO", 3), "PCLO", 3)
    datapoints = di.addPeakFreq(datapoints, "PCLO", 3)
    datapoints = datapoints[datapoints['channel'] == 2]
    X_cluster = datapoints[['amplitude', 'frequency']]
    print(X_cluster)
    print(clustering_hierarchical(X_cluster, 4))
    # clusters = predict(cluster_model, lookup_labels, 0, 'kmeans')
    # data_compressed, var = PCA(datapoints, 2)
    # plt.scatter(data_compressed[0], data_compressed[1])
    # plt.show()
    # print(data_compressed)
    # print(var)
