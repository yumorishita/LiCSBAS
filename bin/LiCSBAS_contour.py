#!/usr/bin/env python3
"""
v1.0 20200408 Yu Morishita, GSI

========
Overview
========
This script draws contours from a GeoTIFF file and output a GeoJSON file (with GSImaps style).

=====
Usage
=====
LiCSBAS_contour.py -i geotiff -c cont_int [-q cut_nodes] [-o contfile] [-a attrib] [--nodata float] [--no_zero] [--color_n colorcode] [--color_p colorcode] [--color_0 colorcode] [--width float] [--opacity float]

 -i  Input GeoTIFF file
 -c  Contour interval
 -q  Do not draw contours with less nodes than this number (Default: 10)
 -o  Output contour GeoJSON file (Default: [geotiff%.tif].cont.geojson)
 -a  Name for the attribute (good to include unit) (Default: geotiff file name)
 --nodata  Nodata value (Default: nan)
 --no_zero  Do not draw contours with 0
 --color_[n|p|0]  Color code of contours with negative, positive, 0 values.
    (e.g., --color_n "#0000ff" --color_p "#ff0000", blue for negative and red for positive)
    (Default: "#000000" (black) for all) 
 --width  Width of contour lines (Default: 2)
 --opacity Opacity of contour lines (Default: 0.5)

"""
#%% Change log
'''
v1.0 20200408 Yu Morishita, GSI
 - Original implementationf
'''

#%% Import
import getopt
import os
import sys
import time
import json
import numpy as np
import subprocess as subp

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


#%% Main
def main(argv=None):
   
    #%% Check argv
    if argv == None:
        argv = sys.argv
        
    start = time.time()
    ver=1.0; date=20200408; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    #%% Set default
    infile = []
    cint = []
    node_thre = 10
    outfile = []
    attrib = []
    nodata = np.nan
    no_zero_flag = False

    color_n = '#000000'
    color_p = '#000000'
    color_0 = '#000000'
    opacity = 0.5
    width = 2

    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:c:q:a:", ["help", "nodata=", "no_zero", "color_n=", "color_p=", "color_0=", "opacity=", "width="])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                infile = a
            elif o == '-c':
                cint = float(a)
            elif o == '-o':
                outfile = a
            elif o == '-q':
                node_thre = int(a)
            elif o == '-a':
                attrib = a
            elif o == '--nodata':
                nodata = float(a)
            elif o == '--no_zero':
                no_zero_flag = True
            elif o == '--color_n':
                color_n = a
            elif o == '--color_p':
                color_p = a
            elif o == '--color_0':
                color_0 = a
            elif o == '--opacity':
                opacity = float(a)
            elif o == '--width':
                width = float(a)

        if not infile:
            raise Usage('No input file given, -i is not optional!')
        elif not os.path.exists(infile):
            raise Usage('No {} exists!'.format(infile))
        if not cint:
            raise Usage('No cont_int given, -c is not optional!')

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Read info
    if not outfile:
        outfile = infile.replace('.tif', '.cont.geojson')
    if not attrib:
        attrib = infile
        

    #%% Make contour geojson
    call = ["gdal_contour", "-snodata", str(nodata), "-a", attrib, "-i", str(cint), "-f", "GeoJSON", infile, outfile ]
        
    p = subp.Popen(call, stdout = subp.PIPE, stderr = subp.STDOUT)
    for line in iter(p.stdout.readline, b''):
        print(line.rstrip().decode("utf8"))


    #%% Read json
    with open(outfile, 'r') as f:
        json_dict = json.load(f)
    features_list = json_dict['features'] ## list
    n_feature_in = len(features_list)

    #%% Prepare output
    features_out_list = []
    for feature in features_list:
        ## Remove zero line if no_zero
        if no_zero_flag and feature['properties'][attrib] == 0:
            continue

        ## Remove lines with small n_node
        n_node = len(feature['geometry']['coordinates'])
        if n_node <= node_thre:
            continue

        ## Remove ID
        del feature['properties']['ID']

        ## Add color
        if feature['properties'][attrib] == 0:
            feature['properties']['_color'] = color_0
        elif feature['properties'][attrib] > 0:
            feature['properties']['_color'] = color_p
        else:
            feature['properties']['_color'] = color_n

        ## Set opacity and weight            
        feature['properties']['_opacity'] = opacity
        feature['properties']['_weight'] = width

        features_out_list.append(feature)


    #%% Output json
    n_feature_out = len(features_out_list)
    print('\nNumber of features: {} -> {}'.format(n_feature_in, n_feature_out))

    if n_feature_out == 0:
        print('No features remain, not output {}\n'.format(outfile))
        os.remove(outfile)
    else:
        jsonout_dict = {'type':json_dict['type'], 'features':features_out_list}
        with open(outfile, 'w') as f:
            json.dump(jsonout_dict, f, indent=None)


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output: {}\n'.format(outfile), flush=True)


#%% main
if __name__ == "__main__":
    sys.exit(main())
