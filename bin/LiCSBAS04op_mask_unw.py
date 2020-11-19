#!/usr/bin/env python3
"""
v1.3.4 20201119 Yu Morishita, GSI

This script masks specified areas or low coherence areas in the unw data. The masking is effective when the unw data include areas which have many unwrapping errors and are not of interest, and can improve the result of Step 1-2 (loop closure). Existing files are not re-created to save time, i.e., only the newly available data will be processed. This step is optional.

===============
Input & output files
===============
Inputs in GEOCml*/:
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw
   - yyyymmdd_yyyymmdd.cc
 - slc.mli.par
 
Outputs in GEOCml*mask/
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw[.png]
   - yyyymmdd_yyyymmdd.cc (symbolic link)
 - mask[.png]
[- coh_avg[.png]] (if -c is used)
 - Other files in input directory

=====
Usage
=====
LiCSBAS04op_mask_unw.py -i in_dir -o out_dir [-c coh_thre] [-r x1:x2/y1:y2] [-f txtfile] [--n_para int]

 -i  Path to the GEOCml* dir containing stack of unw data.
 -o  Path to the output dir.
 -c  Threshold for average coherence (e.g., 0.2)
 -r  Range to be masked. Index starts from 0.
     0 for x2/y2 means all. (i.e., 0:0/0:0 means whole area).
 -f  Text file of a list of ranges to be masked (format is x1:x2/y1:y2)
 --n_para  Number of parallel processing (Default: # of usable CPU)

 Note: either -c, -r or -f must be specified.

"""
#%% Change log
'''
v1.3.4 20201119 Yu Morishita, GSI
 - Change default cmap for wrapped phase from insar to SCM.romaO
v1.3.3 20201118 Yu Morishita, GSI
 - Again Bug fix of multiprocessing
v1.3.2 20201116 Yu Morishita, GSI
 - Bug fix of multiprocessing in Mac python>=3.8
v1.3.1 20201028 Yu Morishita, GSI
 - Update how to get n_para
v1.3 20200909 Yu Morishita, GSI
 - Parallel processing
v1.2 20200614 Yu Morishita, GSI
 - Wrong png name printing fixed
v1.1 20200409 Yu Morishita, GSI
 - Add -c (coherence based mask) option
v1.0 20190730 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''

#%% Import
import getopt
import os
import sys
import glob
import shutil
import time
import numpy as np
import multiprocessing as multi
import SCM
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
    ver="1.3.4"; date=20201119; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    ### For paralell processing
    global ifgdates2, in_dir, out_dir, length, width, bool_mask, cycle, cmap_wrap


    #%% Set default
    in_dir = []
    out_dir = []
    coh_thre = []
    ex_range_str = []
    ex_range_file = []
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()

    cmap_noise = 'viridis'
    cmap_wrap = SCM.romaO
    q = multi.get_context('fork')


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:c:r:f:", ["help", "n_para="])
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
            elif o == '-c':
                coh_thre = float(a)
            elif o == '-r':
                ex_range_str = a
            elif o == '-f':
                ex_range_file = a
            elif o == '--n_para':
                n_para = int(a)

        if not in_dir:
            raise Usage('No input directory given, -i is not optional!')
        if not out_dir:
            raise Usage('No output directory given, -o is not optional!')
        if not coh_thre and not ex_range_str and not ex_range_file:
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

    bool_mask = np.zeros((length, width), dtype=np.bool)


    #%% Check and set pixels to be masked based on coherence
    if coh_thre:
        ### Calc coh_avg
        print("\nCalculate coh_avg and define mask (<={})".format(coh_thre), flush=True)
        coh_avg = np.zeros((length, width), dtype=np.float32)
        n_coh = np.zeros((length, width), dtype=np.int16)
        for ifgix, ifgd in enumerate(ifgdates): 
            ccfile = os.path.join(in_dir, ifgd, ifgd+'.cc')
            if os.path.getsize(ccfile) == length*width:
                coh = io_lib.read_img(ccfile, length, width, np.uint8)
                coh = coh.astype(np.float32)/255
            else:
                coh = io_lib.read_img(ccfile, length, width)
                coh[np.isnan(coh)] = 0 # Fill nan with 0

            coh_avg += coh
            n_coh += (coh!=0)

        n_coh[n_coh==0] = 99999 #to avoid zero division
        coh_avg = coh_avg/n_coh

        ### Set mask
        bool_mask[coh_avg <= coh_thre] = True

        ### Save image
        coh_avgfile = os.path.join(out_dir, 'coh_avg')
        coh_avg.tofile(coh_avgfile)
        title = 'Average coherence'
        plot_lib.make_im_png(coh_avg, coh_avgfile+'.png', cmap_noise, title)


    #%% Check and set range to be masked based on specified area
    ### Read -r option
    if ex_range_str:
        if not tools_lib.read_range(ex_range_str, width, length):
            print('ERROR in {}\n'.format(ex_range_str))
            return 1
        else:
            x1, x2, y1, y2 = tools_lib.read_range(ex_range_str, width, length)
            bool_mask[y1:y2, x1:x2] = True
    
    ### Read -f option
    if ex_range_file:
        with open(ex_range_file) as f:
            ex_range_str_all = f.readlines()
        
        for ex_range_str1 in ex_range_str_all:
            if not tools_lib.read_range(ex_range_str1, width, length):
                print('ERROR in {}\n'.format(ex_range_str1))
                return 1
            else:
                x1, x2, y1, y2 = tools_lib.read_range(ex_range_str1, width, length)
                bool_mask[y1:y2, x1:x2] = True
    
    ### Save image of mask
    mask = np.float32(~bool_mask)
    maskfile = os.path.join(out_dir, 'mask')
    mask.tofile(maskfile)

    pngfile = maskfile+'.png'
    title = 'Mask'
    plot_lib.make_im_png(mask, pngfile, cmap_noise, title, 0, 1)
    
    print('\nMask defined.')


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
   
    if n_ifg2 > 0:
        ### Mask with parallel processing
        if n_para > n_ifg2:
            n_para = n_ifg2
            
        print('  {} parallel processing...'.format(n_para), flush=True)
        p = q.Pool(n_para)
        p.map(mask_wrapper, range(n_ifg2))
        p.close()

    print("", flush=True)


    #%% Copy other files
    files = glob.glob(os.path.join(in_dir, '*'))
    for file in files:
        if not os.path.isdir(file): #not copy directory, only file
            print('Copy {}'.format(os.path.basename(file)), flush=True)
            shutil.copy(file, out_dir)

    print('\nMasked area can be check in:')
    print('{}'.format(os.path.relpath(maskfile+'.png')), flush=True)


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(out_dir)))


#%%
def mask_wrapper(ifgix):
    ifgd = ifgdates2[ifgix]
    if np.mod(ifgix,100) == 0:
        print("  {0:3}/{1:3}th unw...".format(ifgix, len(ifgdates2)), flush=True)

    unwfile = os.path.join(in_dir, ifgd, ifgd+'.unw')
    unw = io_lib.read_img(unwfile, length, width)

    ### Mask
    unw[bool_mask] = np.nan

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
    plot_lib.make_im_png(np.angle(np.exp(1j*unw/cycle)*cycle), pngfile, cmap_wrap, title, -np.pi, np.pi, cbar=False)


#%% main
if __name__ == "__main__":
    sys.exit(main())
