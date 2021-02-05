#!/bin/bash -eu

indir="ScientificColourMaps7"
outdir="SCM"
init="$outdir/__init__.py"


### header of init file
mkdir -p $outdir
cat << EOF > $init
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

folder = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

__all__ = {
EOF


### Copy cmap.txt for all colors
for cmap in $(ls -d $indir/[a-z]* | awk -F/ '{print $NF}')
do
  echo $cmap
  mkdir -p $outdir/$cmap

  echo -n " '${cmap}'," >> $init

  \cp $indir/$cmap/${cmap}.txt $outdir/$cmap/
done


### footer of init file
sed -i '$s/.$//' $init  ## remove last ","
cat << EOF >> $init
}

for name in __all__:
    file = os.path.join(folder, name, name + '.txt')
    cm_data = np.loadtxt(file)
    vars()[name] = LinearSegmentedColormap.from_list(name, cm_data)
    vars()[name + '_r'] = LinearSegmentedColormap.from_list(name + '_r', np.flip(cm_data, axis=0))
EOF


### other files pdf and png
\cp $indir/*pdf $outdir/
\cp $indir/*png $outdir/
