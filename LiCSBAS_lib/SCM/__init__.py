import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

__all__ = {
 'acton', 'bamako', 'batlow', 'berlin', 'bilbao', 'broc', 'brocO', 'buda', 'cork', 'corkO', 'davos', 'devon', 'grayC', 'hawaii', 'imola', 'lajolla', 'lapaz', 'lisbon', 'nuuk', 'oleron', 'oslo', 'roma', 'romaO', 'tofino', 'tokyo', 'turku', 'vik', 'vikO'}

for name in __all__:
    file = os.path.join(folder, name, name + '.txt')
    cm_data = np.loadtxt(file)
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
