#!/usr/bin/env python3
"""
v1.1 20200227 Yu Morishita, Uni of Leeds and GSI

========
Overview
========
This script outputs a txt file of time series of displacement at a specified point from cum*.h5.

=====
Usage
=====
LiCSBAS_cum2tstxt.py [-p x/y] [-g lon/lat] [-i cumfile] [-o tsfile] [-r x1:x2/y1:y2] [--mask maskfile]

 -p  x/y coordinate of a point to be output (index range 0 to width-1)
 -g  Lon/Lat of a point to be output
 -i  Input cum*.h5 file (Default: cum_filt.h5)
 -o  Output txt file of time series (Default: ts_[x]_[y].txt)
 -r  Reference area (Default: same as info/*ref.txt)
     Note: x1/y1 range 0 to width-1, while x2/y2 range 1 to width
     0 for x2/y2 means all. (i.e., 0:0/0:0 means whole area).
 --mask  Path to mask file for ref calculation (Default: No mask)

 Note: either -p or -g must be specified.

"""
#%% Change log
'''
v1.1 20200227 Yu Morishita, Uni of Leeds and GSI
 - Add hgt_linear_flag
v1.0 20190730 Yu Morishita, Uni of Leeds and GSI
 - Original implementationf
'''

#%% Import
import getopt
import os
import sys
import time
import re
import numpy as np
import h5py as h5
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib

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
    ver=1.1; date=20200227; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    xy_str = []
    lonlat_str = []
    cumfile = 'cum_filt.h5'
    tsfile = []
    refarea = []
    maskfile = []


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hp:g:i:o:r:", ["help", "mask="])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-p':
                xy_str = a
            elif o == '-g':
                lonlat_str = a
            elif o == '-i':
                cumfile = a
            elif o == '-o':
                tsfile = a
            elif o == '-r':
                refarea = a
            elif o == '--mask':
                maskfile = a

        if not xy_str and not lonlat_str:
            raise Usage('No point location given, use either -p or -g!')
        elif not os.path.exists(cumfile):
            raise Usage('No {} exists! Use -i option.'.format(cumfile))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Read info
    ### Read cumfile
    cumh5 = h5.File(cumfile,'r')
    cum = cumh5['cum']
    gap = cumh5['gap']
    imdates = cumh5['imdates'][()].astype(str).tolist()
    n_im, length, width = cum.shape

    if 'corner_lat' in list(cumh5.keys()):
        geocod_flag = True
        lat1 = float(cumh5['corner_lat'][()])
        lon1 = float(cumh5['corner_lon'][()])
        dlat = float(cumh5['post_lat'][()])
        dlon = float(cumh5['post_lon'][()])
    else:
        geocod_flag = False
    
    if 'deramp_flag' in list(cumh5.keys()):
        deramp_flag = cumh5['deramp_flag'][()]
    else:
        deramp_flag = None

    if 'hgt_linear_flag' in list(cumh5.keys()):
        hgt_linear_flag = cumh5['hgt_linear_flag'][()]
    else:
        hgt_linear_flag = None

    if 'filtwidth_km' in list(cumh5.keys()):
        filtwidth_km = float(cumh5['filtwidth_km'][()])
        filtwidth_yr = float(cumh5['filtwidth_yr'][()])
    else:
        filtwidth_km = filtwidth_yr = None


    #%% Set info
    ###Set ref area
    if not refarea:
        refarea = cumh5['refarea'][()]
        refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]
    else:
        if not tools_lib.read_range(refarea, width, length):
            print('\nERROR in {}\n'.format(refarea), file=sys.stderr)
            return 2
        else:
            refx1, refx2, refy1, refy2 = tools_lib.read_range(refarea, width, length)

    if geocod_flag:
        reflat1, reflon1 = tools_lib.xy2bl(refx1, refy1, lat1, dlat, lon1, dlon)
        reflat2, reflon2 = tools_lib.xy2bl(refx2-1, refy2-1, lat1, dlat, lon1, dlon)
    else:
        reflat1 = reflon1 = reflat2 = reflon2 = None

    ### Set point
    if xy_str: ## -p option
        x, y = [ int(s) for s in xy_str.split('/')]
        if not 1 <= x <= width:
            print("\nERROR: {} is out of range ({}-{})".format(x, 0, width-1), file=sys.stderr)
            return 2
        elif not 1 <= y <= length:
            print("\nERROR: {} is out of range ({}-{})".format(y, 0, length-1), file=sys.stderr)
            return 2

        if geocod_flag:
            lat, lon = tools_lib.xy2bl(x, y, lat1, dlat, lon1, dlon)
        else:
            lat = lon = None
        
    else: ## -g option
        if not geocod_flag:
            print('\nERROR: not geocoded, -g option unavailable\n', file=sys.stderr)
            return 2

        lat2 = lat1+dlat*(length-1)
        lon2 = lon1+dlon*(width-1)
        lon, lat = [ float(s) for s in lonlat_str.split('/')]
        if not lon1 <= lon <= lon2:
            print("\nERROR: {} is out of range ({}-{})".format(lon, lon1, lon2), file=sys.stderr)
            return 2
        elif not lat2 <= lat <= lat1:
            print("\nERROR: {} is out of range ({}-{})".format(lat, lat2, lat1), file=sys.stderr)
            return 2
        
        x, y = tools_lib.bl2xy(lon, lat, width, length, lat1, dlat, lon1, dlon)
        ## update latlon
        lat, lon = tools_lib.xy2bl(x, y, lat1, dlat, lon1, dlon)

    if geocod_flag:
        print('Location: {:.5f}/{:.5f}'.format(lon, lat))

    if not tsfile:
        tsfile = 'ts_{}_{}.txt'.format(x, y)

    ### Gaps
    gap1 = gap[:, y, x]

    ### mask
    if maskfile:
        mask = io_lib.read_img(maskfile, length, width)
        mask[mask==0] = np.nan
    else:
        mask = np.ones((length, width), dtype=np.float32)

    #%% Read cum data
    ts = cum[:, y, x]*mask[y, x]
    if np.all(np.isnan(ts)):
        print('\nERROR: All cum data are Nan at {}/{}!\n'.format(x, y), file=sys.stderr)
        return 2
        
    ts_ref = np.nanmean(cum[:, refy1:refy2, refx1:refx2]*mask[refy1:refy2, refx1:refx2], axis=(1, 2))
    if np.all(np.isnan(ts_ref)):
        print('\nERROR: Ref area has only NaN value!\n', file=sys.stderr)
        return 2
    
    ts_dif = ts-ts_ref
    ts_dif = ts_dif-ts_dif[0] ## Make first date zero

    ### Make txt
    io_lib.make_tstxt(x, y, imdates, ts_dif, tsfile, refx1, refx2, refy1, refy2, gap1, lat=lat, lon=lon, reflat1=reflat1, reflat2=reflat2, reflon1=reflon1, reflon2=reflon2, deramp_flag=deramp_flag, hgt_linear_flag=hgt_linear_flag, filtwidth_km=filtwidth_km, filtwidth_yr=filtwidth_yr)


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output: {}\n'.format(tsfile), flush=True)


#%% main
if __name__ == "__main__":
    sys.exit(main())
