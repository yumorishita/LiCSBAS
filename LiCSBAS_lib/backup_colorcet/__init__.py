import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

__all__ = {
 'CET_C1', 'CET_C1s', 'CET_C2', 'CET_C2s', 'CET_C4', 'CET_C4s', 'CET_C5', 'CET_C5s', 'CET_CBC1', 'CET_CBC2', 'CET_CBD1', 'CET_CBL1', 'CET_CBL2', 'CET_CBTC1', 'CET_CBTC2', 'CET_CBTD1', 'CET_CBTL1', 'CET_CBTL2', 'CET_D1', 'CET_D10', 'CET_D11', 'CET_D12', 'CET_D13', 'CET_D1A', 'CET_D2', 'CET_D3', 'CET_D4', 'CET_D6', 'CET_D7', 'CET_D8', 'CET_D9', 'CET_I1', 'CET_I2', 'CET_I3', 'CET_L1', 'CET_L10', 'CET_L11', 'CET_L12', 'CET_L13', 'CET_L14', 'CET_L15', 'CET_L16', 'CET_L17', 'CET_L18', 'CET_L19', 'CET_L2', 'CET_L3', 'CET_L4', 'CET_L5', 'CET_L6', 'CET_L7', 'CET_L8', 'CET_L9', 'CET_R1', 'CET_R2', 'CET_R3', 'colorwheel', 'bkr', 'bky', 'bwy', 'cwr', 'coolwarm', 'gwv', 'bjy', 'isolum', 'bgy', 'bgyw', 'kbc', 'blues', 'bmw', 'bmy', 'kgy', 'gray', 'dimgray', 'fire', 'kb', 'kg', 'kr', 'rainbow'}

for name in __all__:
    file = os.path.join(folder, name + '.csv')
    cm_data = np.loadtxt(file, delimiter=',')
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[name + '_r'] = LinearSegmentedColormap.from_list(name + '_r', np.flip(cm_data, axis=0))
