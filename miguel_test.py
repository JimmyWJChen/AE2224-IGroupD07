import os
import vallenae as vae

HERE = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
PRIDB = os.path.join(HERE, "Testing_Data/PLB-4-channels/PLBS4_CP090_PCLO1.pridb")

pridb = vae.io.PriDatabase(PRIDB)

print("Tables in database: ", pridb.tables())
print("Number of rows in data table (ae_data): ", pridb.rows())
print("Set of all channels: ", pridb.channel())