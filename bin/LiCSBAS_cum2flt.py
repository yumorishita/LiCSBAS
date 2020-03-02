#!/usr/bin/env python3
"""
v1.1 20190813 Yu Morishita, Uni of Leeds and GSI

========
Overview
========
This script outputs a float32 file of cumulative displacement from cum*.h5.

=====
Usage
=====
LiCSBAS_cum2flt.py -d yyyymmdd [-i infile] [-o outfile] [-m yyyymmdd] [-r x1:x2/y1:y2] [--mask maskfile] [--png] 

 -d  Date to be output
 -i  Path to input cum file (Default: cum_filt.h5)
 -o  Output float32 file (Default: yyyymmdd_yyyymmdd.cum)
 -m  Master (reference) date (Default: first date)
 -r  Reference area (Default: same as info/*ref.txt)
     Note: x1/y1 range 0 to width-1, while x2/y2 range 1 to width
     0 for x2/y2 means all. (i.e., 0:0/0:0 means whole area).
 --mask  Path to mask file for ref phase calculation (Default: No mask)
 --png   Make png file (Default: Not make png)

"""
#%% Change log
'''
v1.1 20190813 Yu Morishita, Uni of Leeds and GSI
 - Bug fix about masking
v1.0 20190730 Yu Morishita, Uni of Leeds and GSI
 - Original implementationf
'''

#%% Import
import getopt
import os
import sys
import re
import time
import numpy as np
import h5py as h5
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib

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
    ver=1.1; date=20190813; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    imd_s = []
    cumfile = 'cum_filt.h5'
    outfile = []
    imd_m = []
    refarea = []
    maskfile = []
    pngflag = False


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hd:i:o:m:r:", ["help", "png", "mask="])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-d':
                imd_s = a
            elif o == '-i':
                cumfile = a
            elif o == '-o':
                outfile = a
            elif o == '-m':
                imd_m = a
            elif o == '-r':
                refarea = a
            elif o == '--mask':
                maskfile = a
            elif o == '--png':
                pngflag = True

        if not imd_s:
            raise Usage('No date given, -d is not optional!')
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
    imdates = cumh5['imdates'][()].astype(str).tolist()
    cum = cumh5['cum']
    n_im, length, width = cum.shape

    if not refarea:
        refarea = cumh5['refarea'][()]
        refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]
    else:
        if not tools_lib.read_range(refarea, width, length):
            print('\nERROR in {}\n'.format(refarea), file=sys.stderr)
            return 2
        else:
            refx1, refx2, refy1, refy2 = tools_lib.read_range(refarea, width, length)
    
    ### Master (reference) date
    if not imd_m:
        imd_m = imdates[0]
        
    ### mask
    if maskfile:
        mask = io_lib.read_img(maskfile, length, width)
        mask[mask==0] = np.nan
    else:
        mask = np.ones((length, width), dtype=np.float32)
        
    ### Check date
    if not imd_s in imdates:
        print('\nERROR: No date of {} exist in {}!'.format(imd_s, cumfile), file=sys.stderr)
        return 2
    if not imd_m in imdates:
        print('\nERROR: No date of {} exist in {}!'.format(imd_m, cumfile), file=sys.stderr)
        return 2

    ix_s = imdates.index(imd_s)
    ix_m = imdates.index(imd_m)
    
    ### Outfile
    if not outfile:
        outfile = '{}_{}.cum'.format(imd_m, imd_s)


    #%% Make flt
    cum_s = cum[ix_s, :, :]
    cum_m = cum[ix_m, :, :]

    cum_dif = cum_s-cum_m
    cum_dif = cum_dif-np.nanmean(cum_dif[refy1:refy2, refx1:refx2])
    cum_dif = cum_dif*mask
        
    cum_dif.tofile(outfile)

       
    #%% Make png if specified
    if pngflag:
        pngfile = outfile+'.png'
        title = '{} (Ref X/Y {}:{}/{}:{})'.format(outfile, refx1, refx2, refy1, refy2)
        plot_lib.make_im_png(cum_dif, pngfile, 'jet', title)
    

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
