#!/usr/bin/env python3
"""
========
Overview
========
This script gets values from a float file at specified points in geographical coordinates. Average values in a boxcar window are also output.

=========
Changelog
=========
v1.1 20191218 Yu Morishita, Uni of Leeds and GSI
 - Add win_size option
v1.0 20190801 Yu Morishita, Uni of Leeds and GSI
 - Original implementationf

=====
Usage
=====
LiCSBAS_get_value_geo.py -i infile -p dempar -l locfile [-o outfile] [--win_size 3] [--bigendian]

 -i  Input file (float, little endian, geocoded)
 -p  Dem parameter file (EQA.dem_par)
 -l  Text file of lists of point locations (lat lon)
 -o  Output text file (Default: [locfile]v.txt)
     Format: lat lon x y value value_avg (x/y start from 0)
 --win_size   Window size of boxcar averaging, must be odd integer (Default: 3)
 --bigendian  If input file is in big endian

"""


#%% Import
import getopt
import os
import sys
import time
import numpy as np
import LiCSBAS_io_lib as io_lib

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

    #%% Set default
    infile = []
    dempar = []
    locfile = []
    outfile = []
    endian = 'little'
    win_size = 3

    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:p:l:o:", ["help", "win_size=", "bigendian"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                infile = a
            elif o == '-p':
                dempar = a
            elif o == '-l':
                locfile = a
            elif o == '-o':
                outfile = a
            elif o == '--win_size':
                win_size = int(a)
            elif o == '--bigendian':
                endian = 'big'

        if not infile:
            raise Usage('No input file given, -i is not optional!')
        elif not os.path.exists(infile):
            raise Usage('No {} exists!'.format(infile))
        elif not dempar:
            raise Usage('No dempar file given, -p is not optional!')
        elif not os.path.exists(dempar):
            raise Usage('No {} exists!'.format(dempar))
        elif not locfile:
            raise Usage('No location file given, -l is not optional!')
        elif not os.path.exists(locfile):
            raise Usage('No {} exists!'.format(locfile))
        elif win_size % 2 == 0:
            raise Usage('win_size must be odd integer!')

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Read info
    if not outfile:
        outfile = locfile.replace('.txt', '')+'v.txt'

    width = int(io_lib.get_param_par(dempar, 'width'))
    length = int(io_lib.get_param_par(dempar, 'nlines'))
    win_half = int((win_size-1)/2)

    ### Geo info
    dlat = float(io_lib.get_param_par(dempar, 'post_lat'))
    dlon = float(io_lib.get_param_par(dempar, 'post_lon'))
    lat_n = float(io_lib.get_param_par(dempar, 'corner_lat')) #grid reg
    lon_w = float(io_lib.get_param_par(dempar, 'corner_lon')) #grid reg

    ### Location list
    with open(locfile) as f:
        latlon_list = f.readlines()
    latlon_list = [ [float(j) for j in i.strip().split()] for i in latlon_list ]

    ### float file
    data = io_lib.read_img(infile, length, width, endian=endian)
    
    
    #%% Make txt file
    f = open(outfile, 'w')
    print('# lat lon x y value value_avg(win_size:{})'.format(win_size), file=f)
    for lat, lon in latlon_list:
        ### Identify x/y from lat/lon
        x = int(np.round((lon-lon_w)/dlon))
        y = int(np.round((lat-lat_n)/dlat))
        
        if x >= width or x < 0 or y >= length or y < 0: ### If outside of area
            x = y = value = value_avg = np.nan
        else: ### Inside
            value = data[y, x]
            
            ### Average
            x1 = x-win_half if x >= win_half else 0 ## to avoid negative value of x1
            y1 = y-win_half if y >= win_half else 0 ## to avoid negative value of y1
            x2 = x+win_half+1
            y2 = y+win_half+1
            if not np.all(np.isnan(data[y1:y2, x1:x2])):
                value_avg = np.nanmean(data[y1:y2, x1:x2])
            else:
                value_avg = np.nan

        print('{} {} {} {} {} {}'.format(lat, lon, x, y, value, value_avg), file=f)

    f.close()


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
