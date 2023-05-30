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

def clustering_kmeans(datapoints, n_clusters, params=['amplitude', 'wpfrequency', 'rise_time']):
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

def clustering_hierarchical(datapoints, n_clusters, linkage='ward', params=['amplitude', 'frequency', 'energy', 'rms', 'counts', 'rise_time', 'signal_strength']):
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
    label = "PD_PCLSR_QI00"
    filtered = True
    n_clusters = 3
    params = ['frequency', 'wpfrequency', 'amplitude_db', 'RA', 'rise_time']
    multichannel = False

    sum_31 = 0
    sum_22 = 0
    sum_211 = 0
    sum_4 = 0
    
    datapoints = di.getHitDatabase(label, created=True)
    if not filtered:
        datapoints = di.addPeakFreq(datapoints, label)
        # datapoints = datapoints.read_hits()
    # datapoints = di.addRA(datapoints)
    # print(datapoints.head(20))
    hits_total = int(datapoints.max()['hit_id'])
    datapoints['cluster'] = np.zeros(len(datapoints.index))
    # channels = int(datapoints.max()['channel'])

    # if multichannel:
    #     for ch in range(1, channels+1):
    #         print(ch)
    #         datapoints_ch = datapoints.loc[datapoints['channel'] == ch].copy()
    #         datapoints_ch = clustering_hierarchical(datapoints_ch, n_clusters, params=params)
    #         # print(datapoints_ch)
    #         for index, row in datapoints_ch.iterrows():
    #             datapoints.loc[index, 'cluster'] = row['cluster']
    #         # for i in range(n_clusters):
    #         #     pridb_cluster = datapoints_ch.loc[datapoints_ch['cluster'] == i]
    #         #     plt.scatter(pridb_cluster['amplitude'], pridb_cluster['wpfrequency']/1000, label='Cluster '+str(i+1))
    #         # plt.xlabel('Amplitude [dB]')
    #         # plt.ylabel('Weighted Peak Frequency [kHz]')
    #         # plt.legend()
    #         # plt.show()

    datapoints = clustering_kmeans(datapoints, n_clusters, params=params)

    # print(datapoints.head(20))

    # for hit in range(hits_total):
    #     hit_points = datapoints.loc[datapoints['hit_id'] == hit].copy()
    #     clustercounts = hit_points['cluster'].value_counts().to_frame()
    #     if 3 in clustercounts.values:
    #         # print(clustercounts)
    #         correct_cluster = clustercounts.index[clustercounts['count'] == 3].to_list()[0]
    #         incorrect_cluster = clustercounts.index[clustercounts['count'] == 1].to_list()[0]
    #         trai_to_correct = hit_points.loc[hit_points['cluster'] == incorrect_cluster]['trai'].to_list()[0]
    #         datapoints.loc[datapoints['trai'] == trai_to_correct, 'cluster'] = correct_cluster
    #         # print(f'{trai_to_correct} moved from Cluster {incorrect_cluster} to {correct_cluster}')
    #         sum_31+=1
            
    #     elif 2 in clustercounts.values:
    #         if 1 in clustercounts.values:
    #             sum_211+=1
    #         else:
    #             sum_22+=1
    #     else:
    #         sum_4+=1
    # print(f'3 vs 1 clusters: {sum_31}, {sum_31/hits_total * 100}%')
    # print(f'2 vs 2 clusters: {sum_22}, {sum_22/hits_total * 100}%')
    # print(f'2 vs 1 vs 1 clusters: {sum_211}, {sum_211/hits_total * 100}%')
    # print(f'4 vs 0 clusters: {sum_4}, {sum_4/hits_total * 100}%')
    # datapoints = di.createHitDataframe(di.getPrimaryDatabase(label, filtered=filtered))
    print(datapoints)
    datapoints.to_csv("testing_data/4-channels/HITS_" + label + ".csv", index=False)
    # datapoints = clustering_kmeans(datapoints, n_clusters, params=["wpfrequency", "frequency", "freqcentroid", "amplitude1", "amplitude2", "amplitude3", "amplitude4", "RA"])
    for i in range(n_clusters):
        pridb_cluster = datapoints.loc[datapoints['cluster'] == i]
        plt.scatter(pridb_cluster['amplitude_db'], pridb_cluster['frequency']/1000, label='Cluster '+str(i+1))
    plt.xlabel('Amplitude [dB]')
    plt.ylabel('Peak Frequency [kHz]')
    plt.legend()
    plt.show()