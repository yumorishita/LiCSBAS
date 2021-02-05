import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

__all__ = {
 'algae', 'amp', 'balance', 'curl', 'deep', 'delta', 'dense', 'diff', 'gray', 'haline', 'ice', 'matter', 'oxy', 'phase', 'rain', 'solar', 'speed', 'tarn', 'tempo', 'thermal', 'topo', 'turbid'}

for name in __all__:
    file = os.path.join(folder, name, name + '.txt')
    cm_data = np.loadtxt(file)
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[name + '_r'] = LinearSegmentedColormap.from_list(name + '_r', np.flip(cm_data, axis=0))
