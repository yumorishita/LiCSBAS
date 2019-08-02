#!/usr/bin/env python3
"""
========
Overview
========
This script masks some parts of unw data. The masking is effective when the unw data include areas which have many unwrapping errors and are not of interest, and can improve the result of step 12 (loop closure).

=========
Changelog
=========
v1.0 20190730 Yu Morishita, Uni of Leeds and GSI
 - Original implementation

===============
Input & output files
===============
Inputs in GEOCml* directory:
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.unw
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.cc
 - slc.mli.par
 
Outputs in output directory
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.unw[.png]
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.cc (symbolic link)
 - Other files in input directory

=====
Usage
=====
LiCSBAS04op_mask_unw.py -i in_dir -o out_dir [-r x1:x2/y1:y2] [-f txtfile]

 -i  Path to the GEOCml* dir containing stack of unw data.
 -o  Path to the output dir.
 -r  Range to be masked. Index starts from 0.
     0 for x2/y2 means all. (i.e., 0:0/0:0 means whole area).
 -f  Text file of a list of ranges to be masked (format is x1:x2/y1:y2)

"""


#%% Import
import getopt
import os
import sys
import glob
import shutil
import time
import numpy as np
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg

#%%
def main(argv=None):
   
    #%% Check argv
    if argv == None:
        argv = sys.argv
        
    start = time.time()
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    in_dir = []
    out_dir = []
    ex_range_str = []
    ex_range_file = []


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:r:f:", ["help"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                in_dir = a
            elif o == '-o':
                out_dir = a
            elif o == '-r':
                ex_range_str = a
            elif o == '-f':
                ex_range_file = a

        if not in_dir:
            raise Usage('No input directory given, -i is not optional!')
        if not out_dir:
            raise Usage('No output directory given, -o is not optional!')
        if not ex_range_str and not ex_range_file:
            raise Usage('Neither -r nor -f option is given!')
        elif not os.path.isdir(in_dir):
            raise Usage('No {} dir exists!'.format(in_dir))
        elif not os.path.exists(os.path.join(in_dir, 'slc.mli.par')):
            raise Usage('No slc.mli.par file exists in {}!'.format(in_dir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2

    
#%% Read info and make dir
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    ifgdates = tools_lib.get_ifgdates(in_dir)
    n_ifg = len(ifgdates)

    mlipar = os.path.join(in_dir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

    speed_of_light = 299792458 #m/s
    radar_frequency = float(io_lib.get_param_par(mlipar, 'radar_frequency')) #Hz
    wavelength = speed_of_light/radar_frequency #meter
    if wavelength > 0.2: ## L-band
        cycle = 1.5  # 2pi/cycle for png
    else: ## C-band
        cycle = 3  # 2pi*3/cycle for png


    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    #%% Check and set range to be masked
    ### Read -r option
    ex_range_list_list = []
    if ex_range_str:
        if not tools_lib.read_range(ex_range_str, width, length):
            print('ERROR in {}\n'.format(ex_range_str))
            return 1
        else:
            ex_range_list_list.append(tools_lib.read_range(ex_range_str, width, length))
    
    ### Read -f option
    if ex_range_file:
        with open(ex_range_file) as f:
            ex_range_str_all = f.readlines()
        
        for ex_range_str1 in ex_range_str_all:
            if not tools_lib.read_range(ex_range_str1, width, length):
                print('ERROR in {}\n'.format(ex_range_str1))
                return 1
            else:
                ex_range_list_list.append(tools_lib.read_range(ex_range_str1, width, length))

    print("\nArea to be masked:", flush=True)
    print(ex_range_list_list, flush=True)
    

    #%% Mask unw
    print('\nMask unw and link cc', flush=True)
    ### First, check if already exist
    ifgdates2 = []
    for ifgix, ifgd in enumerate(ifgdates): 
        out_dir1 = os.path.join(out_dir, ifgd)
        unwfile_m = os.path.join(out_dir1, ifgd+'.unw')
        ccfile_m = os.path.join(out_dir1, ifgd+'.cc')
        if not (os.path.exists(unwfile_m) and os.path.exists(ccfile_m)):
            ifgdates2.append(ifgd)

    n_ifg2 = len(ifgdates2)
    if n_ifg-n_ifg2 > 0:
        print("  {0:3}/{1:3} masked unw and cc already exist. Skip".format(n_ifg-n_ifg2, n_ifg), flush=True)
   
    ### Mask
    for ifgix, ifgd in enumerate(ifgdates2): 
        if np.mod(ifgix,100) == 0:
            print("  {0:3}/{1:3}th unw...".format(ifgix, n_ifg2), flush=True)
    
        unwfile = os.path.join(in_dir, ifgd, ifgd+'.unw')
        unw = io_lib.read_img(unwfile, length, width)

        ### Mask
        for ex_range_list in ex_range_list_list:
            x1, x2, y1, y2 = ex_range_list
            unw[y1:y2, x1:x2] = np.nan

        ### Output
        out_dir1 = os.path.join(out_dir, ifgd)
        if not os.path.exists(out_dir1): os.mkdir(out_dir1)
        
        unw.tofile(os.path.join(out_dir1, ifgd+'.unw'))
        
        if not os.path.exists(os.path.join(out_dir1, ifgd+'.cc')):
            ccfile = os.path.join(in_dir, ifgd, ifgd+'.cc')
            os.symlink(os.path.relpath(ccfile, out_dir1), os.path.join(out_dir1, ifgd+'.cc'))

        ## Output png for masked unw
        pngfile = os.path.join(out_dir1, ifgd+'.unw.png')
        title = '{} ({}pi/cycle)'.format(ifgd, cycle*2)
        plot_lib.make_im_png(np.angle(np.exp(1j*unw/cycle)*cycle), pngfile, 'insar', title, -np.pi, np.pi, cbar=False)
            
    print("", flush=True)


    #%% Copy other files
    files = glob.glob(os.path.join(in_dir, '*'))
    for file in files:
        if not os.path.isdir(file): #not copy directory, only file
            print('Copy {}'.format(os.path.basename(file)), flush=True)
            shutil.copy(file, out_dir)


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(out_dir)))


#%% main
if __name__ == "__main__":
    sys.exit(main())
