#!/bin/bash -eu

cptdir="[Insert path]/gmt/cpt"
outdir="GMT"
init="$outdir/__init__.py"

discrete=("categorical" "gebco" "ibcso" "mag" "paired" "panoply")
SCM=("acton" "bam" "bamO" "bamako" "batlow" "batlowK" "batlowW" "berlin" "bilbao" "broc" "brocO" "buda" "bukavu" "cork" "corkO" "davos" "devon" "fes" "grayC" "hawaii" "imola" "lajolla" "lapaz" "lisbon" "nuuk" "oleron" "oslo" "roma" "romaO" "tofino" "tokyo" "turku" "vanimo" "vik" "vikO")

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


### Make cmap.txt for all colors
for cpt in $(ls $cptdir/*.cpt | awk -F/ '{print $NF}')
do
  cmap="${cpt%.cpt}"
  echo "$cmap"

  for a in ${discrete[@]}
  do
    if [ "$cmap" == "$a" ]; then
      echo "Skip because discrete."
      continue 2
    fi
  done

  for a in ${SCM[@]}
  do
    if [ "$cmap" == "$a" ]; then
      echo "Skip because SCM."
      continue 2
    fi
  done

  mkdir -p $outdir/$cmap

  echo -n " '${cmap}'," >> $init

  gmt makecpt -C$cpt -T0/256/1 -Fr -Z | head -256 | awk '{print $2}' | \
  awk -F/ '{printf "%8.6f %8.6f %8.6f\n", $1/255, $2/255, $3/255}' > $outdir/$cmap/${cmap}.txt
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

