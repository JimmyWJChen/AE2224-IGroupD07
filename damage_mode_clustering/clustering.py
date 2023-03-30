import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering


def standarize(datapoints):
    X_norm = datapoints - datapoints.mean()
    X_norm = X_norm/np.std(datapoints)
    X_norm.dropna(axis=1, inplace=True)
    return X_norm

def PCA(datapoints, n):
    datapoints = standarize(datapoints[["amplitude", "duration", "energy", "signal_strength", "counts", "frequency"]])
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

def clustering_kmeans(datapoints, n_clusters, params=['amplitude', 'frequency']):
    cluster_model = KMeans(n_clusters=n_clusters, n_init=10).fit(datapoints[params])
    datapoints['cluster'] = cluster_model.predict(datapoints[params])
    return datapoints

def clustering_hierarchical(datapoints, n_clusters, linkage='ward', params=['amplitude', 'frequency']):
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(datapoints[params])
    datapoints['cluster'] = cluster_model.labels_
    return datapoints

if __name__=="__main__":
    datapoints = di.filterPrimaryDatabase(di.getPrimaryDatabase("PCLO", 3), "PCLO", 3)
    datapoints = di.addPeakFreq(datapoints, "PCLO", 3)
    datapoints = datapoints[datapoints['channel'] == 2]
    n_clusters = 3
    datapoints = clustering_kmeans(datapoints, n_clusters)
    for i in range(n_clusters):
        pridb_cluster = datapoints.loc[datapoints['cluster'] == i].copy()
        plt.scatter(pridb_cluster['frequency'], pridb_cluster['amplitude'])
    plt.show()
