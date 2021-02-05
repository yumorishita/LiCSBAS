import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

__all__ = {
 'acton', 'bam', 'bamO', 'bamako', 'batlow', 'batlowK', 'batlowW', 'berlin', 'bilbao', 'broc', 'brocO', 'buda', 'bukavu', 'cork', 'corkO', 'davos', 'devon', 'fes', 'grayC', 'hawaii', 'imola', 'lajolla', 'lapaz', 'lisbon', 'nuuk', 'oleron', 'oslo', 'roma', 'romaO', 'tofino', 'tokyo', 'turku', 'vanimo', 'vik', 'vikO'}

for name in __all__:
    file = os.path.join(folder, name, name + '.txt')
    cm_data = np.loadtxt(file)
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[name + '_r'] = LinearSegmentedColormap.from_list(name + '_r', np.flip(cm_data, axis=0))
