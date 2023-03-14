import os
# import matplotlib.pyplot as plt
import vallenae as vae

def getPrimaryDatabase(label, testno):
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

if __name__ == "__main__":
    pridb = getPrimaryDatabase("TEST", 1)
    print(pridb.read_hits())