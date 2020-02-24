#!/usr/bin/env python3
"""
v1.0 20190731 Yu Morishita, Uni of Leeds and GSI

========
Overview
========
This script masks a float file using a mask file.

=====
Usage
=====
LiCSBAS_mask_flt.py -i infile -m maskfile [-o outfile]

 -i  Path to input float file
 -m  Path to maskfile
 -o  Output masked float file (Default: infile.mskd)

"""
#%% Change log
'''
v1.0 20190731 Yu Morishita, Uni of Leeds and GSI
 - Original implementationf
'''

# [--png]
# --png   Make png file (Default: No)
# no png option because it needs size of float...

#%% Import
import getopt
import os
import sys
import time
import numpy as np
#import LiCSBAS_plot_lib as plot_lib

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
    ver=1.0; date=20190731; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    infile = []
    maskfile = []
    outfile = []
    pngflag = False

    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:m:", ["help", "png"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                infile = a
            elif o == '-o':
                outfile = a
            elif o == '-m':
                maskfile = a
            elif o == '--png':
                pngflag = True

        if not infile:
            raise Usage('No infile given, -i is not optional!')
        if not maskfile:
            raise Usage('No maskfile given, -m is not optional!')
        elif not os.path.exists(infile):
            raise Usage('No {} exists!'.format(infile))
        elif not os.path.exists(maskfile):
            raise Usage('No {} exists!'.format(maskfile))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2

    
    #%% Read data
    data = np.fromfile(infile, dtype=np.float32)
    mask = np.fromfile(maskfile, dtype=np.float32)
    mask[mask==0] = np.nan

    ### Outfile
    if not outfile:
        outfile = infile+'.mskd'


    #%% Mask flt and output
    data_mskd = data*mask
    data_mskd.tofile(outfile)

       
#    #%% Make png if specified
#    if pngflag:
#        pngfile = outfile+'.png'
#        plot_lib.make_im_png(data_mskd, pngfile, 'jet', outfile)
    

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
