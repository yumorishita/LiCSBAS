#!/usr/bin/env python3
"""
v1.0 20190916 Yu Morishita, Uni of Leeds and GSI

========
Overview
========
This script gets a profile data between two points specified in geographical coordinates or xy coordinates from a float file. A quick look image is displayed and a text file and kml file are output. 

=====
Usage
=====
LiCSBAS_profile.py -i infile -p dempar [-r x1,y1/x2,y2] [-g lon1,lat1/lon2,lat2] [-o outfile] [--bigendian] [--nodisplay]

 -i  Input file (float, little endian)
 -p  Dem parameter file (EQA.dem_par)
 -r  Point locations in xy coordinates
 -g  Point locations in geographical coordinates
 -o  Output text file (Default: profile.txt)
     Format: lat lon x y distance value (x/y start from 0)
 --bigendian  If input file is in big endian
 --nodisplay  Not display quick look images

 Note: either -r or -g must be specified.

"""
#%% Change log
'''
v1.0 20190916 Yu Morishita, Uni of Leeds and GSI
 - Original implementationf
'''

#%% Import
import getopt
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg

#%%
def make_line_kml(lon1, lat1, lon2, lat2, kmlfile):
    with open(kmlfile, "w") as f:
        print('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n<Placemark>\n<LineString>\n<coordinates>{},{} {},{}</coordinates>\n</LineString>\n</Placemark>\n</Document>\n</kml>'.format(lon1, lat1, lon2, lat2), file=f)


#%% Main
def main(argv=None):
   
    #%% Check argv
    if argv == None:
        argv = sys.argv
        
    start = time.time()
    ver=1.0; date=20190916; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    #%% Set default
    infile = []
    dempar = []
    outfile = []
    range_str = []
    range_geo_str = []
    endian = 'little'
    display_flag = True

    margin = 10 # pixel

    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:p:r:g:o:", ["help", "bigendian", "nodisplay"])
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
            elif o == '-r':
                range_str = a
            elif o == '-g':
                range_geo_str = a
            elif o == '-o':
                outfile = a
            elif o == '--bigendian':
                endian = 'big'
            elif o == '--nodisplay':
                display_flag = False

        if not infile:
            raise Usage('No input file given, -i is not optional!')
        elif not os.path.exists(infile):
            raise Usage('No {} exists!'.format(infile))
        elif not dempar:
            raise Usage('No dempar file given, -p is not optional!')
        elif not os.path.exists(dempar):
            raise Usage('No {} exists!'.format(dempar))
        elif not range_str and not range_geo_str:
            raise Usage('No point locations given, use either -r or -g!')
        if range_str and range_geo_str:
            raise Usage('Both -r and -g given, use either -r or -g not both!')

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Read info
    if not outfile:
        outfile = 'profile.txt'

    width = int(io_lib.get_param_par(dempar, 'width'))
    length = int(io_lib.get_param_par(dempar, 'nlines'))

    ### Geo info
    dlat = float(io_lib.get_param_par(dempar, 'post_lat'))
    dlon = float(io_lib.get_param_par(dempar, 'post_lon'))
    lat_n = float(io_lib.get_param_par(dempar, 'corner_lat')) #grid reg
    lon_w = float(io_lib.get_param_par(dempar, 'corner_lon')) #grid reg

    ### float file
    data = io_lib.read_img(infile, length, width, endian=endian)
    
    
    #%% Check and set range to be clipped
    ### Read -r or -g option
    if range_str: ## -r
        if not tools_lib.read_range_line(range_str, width, length):
            print('\nERROR in {}\n'.format(range_str), file=sys.stderr)
            return 1
        else:
            x1, x2, y1, y2 = tools_lib.read_range_line(range_str, width, length)
    else: ## -g
        if not tools_lib.read_range_line_geo(range_geo_str, width, length, lat_n, dlat, lon_w, dlon):
            print('\nERROR in {}\n'.format(range_geo_str), file=sys.stderr)
            return 1
        else:
            x1, x2, y1, y2 = tools_lib.read_range_line_geo(range_geo_str, width, length, lat_n, dlat, lon_w, dlon)
            range_str = '{},{}/{},{}'.format(x1, x2, y1, y2)


    ### Calc latlon info
    lat1, lon1 = tools_lib.xy2bl(x1, y1, lat_n, dlat, lon_w, dlon)
    lat2, lon2 = tools_lib.xy2bl(x2, y2, lat_n, dlat, lon_w, dlon)
    
    centerlat = (lat1+lat2)/2
    ra = float(io_lib.get_param_par(dempar, 'ellipsoid_ra'))
    recip_f = float(io_lib.get_param_par(dempar, 'ellipsoid_reciprocal_flattening'))
    rb = ra*(1-1/recip_f) ## polar radius
    dlat_m = 2*np.pi*rb/360*abs(dlat)
    dlon_m = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))


    #%% Get profile
    dist_xy = int(np.hypot(x2-x1, y2-y1))
    dist_m = int(np.hypot((x2-x1)*dlon_m, (y2-y1)*dlat_m))

    xs = np.int32(np.round(np.linspace(x1, x2, dist_xy)))
    ys = np.int32(np.round(np.linspace(y1, y2, dist_xy)))
    lats, lons = tools_lib.xy2bl(xs, ys, lat_n, dlat, lon_w, dlon)
    profile = data[ys, xs]
    dists_m = np.linspace(0, dist_m, dist_xy)


    #%% Make txt file
    f = open(outfile, 'w')
    print('#lat      lon           x    y distance(m)  value', file=f)
    for i in range(len(xs)):
        print('{:.6f} {:.6f} {:4d} {:4d} {:7.1f}  {}'.format(lats[i], lons[i], xs[i], ys[i], dists_m[i], profile[i]), file=f)

    f.close()


    #%% Make kml file
    make_line_kml(lon1, lat1, lon2, lat2, outfile.replace('.txt', '.kml'))


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output: {}\n'.format(outfile), flush=True)


    #%% Display
    if display_flag:
        fig, axes = plt.subplots(2, 1, num='Profile')
        fig.suptitle('Profile from A({},{}) to B({},{})\n({:.6f},{:.6f}) -> ({:.6f},{:.6f})'.format(x1, y1, x2, y2, lon1, lat1, lon2, lat2))
        xmin = x1-margin if x1<x2 else x2-margin
        xmax = x1+margin if x1>x2 else x2+margin
        ymin = y1-margin if y1<y2 else y2-margin
        ymax = y1+margin if y1>y2 else y2+margin
        if xmin < 0: xmin = 0
        if xmax > width: xmax = width
        if ymin < 0: ymin = 0
        if ymax > length: ymax = length

        im = axes[0].imshow(data)
        axes[0].plot([x1, x2], [y1, y2], 'ro-')
        axes[0].text(x1, y1, 'A')
        axes[0].text(x2, y2, 'B')
        fig.colorbar(im, ax=axes[0])
        axes[0].set_xlim([xmin, xmax])
        axes[0].set_ylim([ymax, ymin])
        
        dists_km = dists_m/1000
        axes[1].plot(dists_km, profile, 'bo-')
        vshift = (np.nanmax(profile)-np.nanmin(profile))/20
        axes[1].text(dists_km[0], profile[0]+vshift, 'A', horizontalalignment='center')
        axes[1].text(dists_km[-1], profile[-1]+vshift, 'B', horizontalalignment='center')
        axes[1].set_xlabel('Distance (km)')
        axes[1].grid()
        
        plt.show()
    

#%% main
if __name__ == "__main__":
    sys.exit(main())
