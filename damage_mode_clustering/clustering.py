import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di
import pandas
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# import vallenae as ae

# standard threshold for AE 34dB, we have at 45dB -> thus missing IDs

def standarize(datapoints):
    X_norm = datapoints - datapoints.mean()
    X_norm = X_norm/np.std(datapoints)
    X_norm.dropna(axis=1, inplace=True)
    return X_norm

def PCA(datapoints, n):
    datapoints = standarize(datapoints[["amplitude", "duration", "energy", "rms", "rise_time", "signal_strength", "counts"]])
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

if __name__=="__main__":
    datapoints = di.filterPrimaryDatabase(di.getPrimaryDatabase("PCLO", 3), "PCLO", 3)
    datapoints = datapoints[datapoints['channel'] == 2]
    print(datapoints)
    data_compressed, var = PCA(datapoints, 4)
    plt.scatter(data_compressed[0], data_compressed[1])
    plt.show()
    print(data_compressed)
    print(var)
    # normalize data
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(datapoints)

    # perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(features_norm)

    # add cluster labels to original dataframe
    datapoints['cluster'] = kmeans.labels_

    # save results to new file
    data.to_csv('your_results.csv', index=False)
