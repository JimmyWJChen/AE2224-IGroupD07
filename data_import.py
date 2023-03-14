import os
import matplotlib.pyplot as plt
import vallenae as vae


def getPrimaryDatabase(label, testno=1):
    if label == "PCLO" or label == "PCLS":
        path = "Testing data/PLB-4-channels/PLBS4_CP090_" + label + str(testno) + ".pridb"
    elif label == "TEST":
        path = "Testing data/PLB-8-channels/PLBS8_QI090_" + label + ".pridb"
    else:
        path = "Testing data/PLB-8-channels/PLBS8_QI090_" + label + str(testno) + ".pridb"
    HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    PRIDB = os.path.join(HERE, path)
    # print(PRIDB)
    pridb = vae.io.PriDatabase(PRIDB)
    return pridb

def getWaveform(label, testno=1, trai=1):
    if label == "PCLO" or label == "PCLS":
        path = "Testing data/PLB-4-channels/PLBS4_CP090_" + label + str(testno) + ".tradb"
    elif label == "TEST":
        path = "Testing data/PLB-8-channels/PLBS8_QI090_" + label + ".tradb"
    else:
        path = "Testing data/PLB-8-channels/PLBS8_QI090_" + label + str(testno) + ".tradb"
    HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    TRADB = os.path.join(HERE, path)
    with vae.io.TraDatabase(TRADB) as tradb:
        y, t = tradb.read_wave(trai)
    return y, t

if __name__ == "__main__":
    pridb = getPrimaryDatabase("TEST")
    print(pridb.read_hits())
    y, t = getWaveform("TEST", 1, 5)
    plt.plot(t, y)
    plt.show()