import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_import as di
# import vallenae as ae

test_pridb = di.getPrimaryDatabase("TEST")
print(test_pridb.read_hits())