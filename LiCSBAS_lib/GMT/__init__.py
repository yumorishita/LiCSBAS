import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

__all__ = {
 'abyss', 'bathy', 'cool', 'copper', 'cubhelix', 'cyclic', 'dem1', 'dem2', 'dem3', 'dem4', 'drywet', 'earth', 'elevation', 'etopo1', 'geo', 'globe', 'gray', 'haxby', 'hot', 'inferno', 'jet', 'magma', 'nighttime', 'no_green', 'ocean', 'plasma', 'polar', 'rainbow', 'red2green', 'relief', 'seafloor', 'sealand', 'seis', 'split', 'srtm', 'terra', 'topo', 'turbo', 'viridis', 'world', 'wysiwyg'}

for name in __all__:
    file = os.path.join(folder, name, name + '.txt')
    cm_data = np.loadtxt(file)
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[name + '_r'] = LinearSegmentedColormap.from_list(name + '_r', np.flip(cm_data, axis=0))
