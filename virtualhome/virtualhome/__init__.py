import glob
import sys
from sys import platform

# Needs to be fixed!
original_path = sys.path[5]
new_path = original_path + '/virtualhome/simulation'
new_path = '/Users/lzw365/Documents/MyData/CSE_repo/progprompt-vh/virtualhome/virtualhome/simulation'
sys.path.append(new_path)

from unity_simulator.comm_unity import UnityCommunication
from unity_simulator import utils_viz