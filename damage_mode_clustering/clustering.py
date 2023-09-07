import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di
import pandas as pd
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

def clustering_kmeans(datapoints, n_clusters, params=['frequency', 'wpfrequency', 'amplitude_db', 'RA', 'rise_time']):
    cluster_model = KMeans(n_clusters=n_clusters, n_init=10).fit(datapoints[params])
    datapoints['cluster'] = cluster_model.predict(datapoints[params])
    # Calculate the center value sum for each cluster
    center_sums = {}
    for cluster_label in set(datapoints['cluster']):
        cluster_data = datapoints[datapoints['cluster'] == cluster_label]
        center_sums[cluster_label] = cluster_data[params[0]].sum()

    # Sort the cluster labels based on the center value sum in ascending order
    sorted_labels = sorted(center_sums, key=center_sums.get)

    # Assign new labels based on the sorted order
    new_labels = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    datapoints['cluster'] = datapoints['cluster'].map(new_labels)
    return datapoints

def clustering_hierarchical(datapoints, n_clusters, linkage='ward', params=['frequency', 'wpfrequency', 'amplitude_db', 'RA', 'rise_time']):
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(datapoints[params])
    datapoints['cluster'] = cluster_model.labels_
    # Calculate the center value sum for each cluster
    center_sums = {}
    for cluster_label in set(datapoints['cluster']):
        cluster_data = datapoints[datapoints['cluster'] == cluster_label]
        center_sums[cluster_label] = cluster_data[params[0]].sum()

    # Sort the cluster labels based on the center value sum in ascending order
    sorted_labels = sorted(center_sums, key=center_sums.get)

    # Assign new labels based on the sorted order
    new_labels = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    datapoints['cluster'] = datapoints['cluster'].map(new_labels)
    return datapoints

if __name__=="__main__":
    pd.set_option('display.max_rows', 20)
    n_clusters = 3
    params = ['frequency', 'wpfrequency', 'amplitude_db', 'RA', 'rise_time']
    l = 0
    labels = ['PD_PCLO_QI090', 'PD_PCLSR_QI00', 'PD_PCLSR_QI090', 'PD_PCLO_QI00']
    orders = np.array([[2, -1, -1], [0, 1, -1], [2, -1, -1], [2, -1, -1]])
    orders = orders
    for label in labels:
        # pridb = di.getPrimaryDatabase(label)
        # pridb = di.filterPrimaryDatabase(pridb, label)
        datapoints = di.getHitDatabase(label, created=True)
        datapoints['cluster'] = datapoints['cluster'] + 3
        for i in range(n_clusters):
            datapoints.loc[datapoints['cluster'] == i+3, ['cluster']] = datapoints.loc[datapoints['cluster'] == i+3, ['cluster']] - 3 + orders[l, i]

        # hits_total = int(datapoints.max()['hit_id'])
        # datapoints['cluster'] = np.zeros(len(datapoints.index))
        # datapoints = clustering_kmeans(datapoints, n_clusters, params=params)
        print(datapoints)
        # pridb.to_csv("testing_data/4-channels/" + label + ".csv", index=False)
        datapoints.to_csv("testing_data/4-channels/HITS_" + label + ".csv", index=False)
        # # datapoints = clustering_kmeans(datapoints, n_clusters, params=["wpfrequency", "frequency", "freqcentroid", "amplitude1", "amplitude2", "amplitude3", "amplitude4", "RA"])
        # delam_cluster = 3
        # print(f'Delamination percentage: {datapoints.loc[datapoints["cluster"] == delam_cluster - 1].shape[0] / datapoints.shape[0] * 100}%')

        # for i in range(n_clusters):
        #     pridb_cluster = datapoints.loc[datapoints['cluster'] == i]
        #     plt.scatter(pridb_cluster['amplitude_db'], pridb_cluster['frequency']/1000, label='Cluster '+str(i+1))
        # plt.xlabel('Amplitude [dB]')
        # plt.ylabel('Peak Frequency [kHz]')
        # plt.legend()
        # plt.savefig(f'figures/{label}.png', format='png')
        # plt.show()
        l+=1