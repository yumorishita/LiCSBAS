#!/usr/bin/env python3
"""
v1.0 20200225 Yu Morishita, Uni of Leeds and GSI

========
Overview
========
This script creates a png file (or in other formats) of SB network. A Gap of the network are denoted by a black vertical line if a gap exist. Bad ifgs can be denoted by red lines.

=====
Usage
=====
LiCSBAS_plot_network.py -i ifg_list -b bperp_list [-o outpngfile] [-r bad_ifg_list] [--not_plot_bad]

 -i  Text file of ifg list (format: yyymmdd_yyyymmdd)
 -b  Text file of bperp list (format: yyyymmdd yyyymmdd bperp dt)
 -o  Output image file (Default: netowrk.png)
     Available file formats: png, ps, pdf, or svg
     (see manual for matplotlib.pyplot.savefig)
 -r  Text file of bad ifg list to be plotted with red lines (format: yyymmdd_yyyymmdd)
 --not_plot_bad  Not plot bad ifgs with red lines

"""

#%% Change log
'''
v1.0 20200225 Yu Morishita, Uni of Leeds and GSI
 - Original implementationf
'''

#%% Import
import getopt
import os
import sys
import time
import numpy as np
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
    ver=1.0; date=20200225; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    #%% Set default
    ifgfile = []
    bperpfile = []
    pngfile = 'network.png'
    bad_ifgfile = []
    plot_bad_flag = True


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:b:o:r:", ["help", "not_plot_bad"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                ifgfile = a
            elif o == '-b':
                bperpfile = a
            elif o == '-o':
                pngfile = a
            elif o == '-r':
                bad_ifgfile = a
            elif o == '--not_plot_bad':
                plot_bad_flag = False

        if not ifgfile:
            raise Usage('No ifg list given, -i is not optional!')
        elif not os.path.exists(ifgfile):
            raise Usage('No {} exists!'.format(ifgfile))
        elif not bperpfile:
            raise Usage('No bperp list given, -b is not optional!')
        elif not os.path.exists(bperpfile):
            raise Usage('No {} exists!'.format(bperpfile))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Read info
    ifgdates = io_lib.read_ifg_list(ifgfile)
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    bperp = io_lib.read_bperp_file(bperpfile, imdates)

    if bad_ifgfile:
        bad_ifgdates = io_lib.read_ifg_list(bad_ifgfile)
    else:
        bad_ifgdates = []

    #%% Plot image
    plot_lib.plot_network(ifgdates, bperp, bad_ifgdates, pngfile, plot_bad_flag)


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output: {}\n'.format(pngfile), flush=True)


#%% main
if __name__ == "__main__":
    sys.exit(main())
