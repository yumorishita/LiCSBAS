#!/usr/bin/env python3
"""

========
Overview
========
This script calculates the standard deviation of the velocity by the bootstrap method and STC (spatio-temporal consistency; Hanssen et al., 2008).
Optionally, it can improve velocity estimates using RANSAC algorithm
===============
Input & output files
===============
Inputs in TS_GEOCml*/ :
 - cum.h5 : Cumulative displacement (time-series) in mm

Outputs in TS_GEOCml*/results/ :
 - vstd[.png] : Std of velocity in mm/yr
 - stc[.png]  : Spatio-temporal consistency in mm
 [- vel2[.png], intercept2: Velocity and intercept after RANSAC outlier-free regression]

=====
Usage
=====
LiCSBAS14_vel_std.py -t tsadir [-i cumfile] [--mem_size float] [--gpu] [--ransac]

 -t  Path to the TS_GEOCml* dir.
 -i  Path to cum file (Default: cum.h5)
 --mem_size   Max memory size for each patch in MB. (Default: 4000)
 --gpu        Use GPU (Need cupy module)
 --ransac     Recalculate velocity free from outliers (use RANSAC algorithm)

"""
#%% Change log
'''
v1.3 20221115 Milan Lazecky, Uni of Leeds
 - Add RANSAC option
v1.2 20210309 Yu Morishita, GSI
 - Add GPU option
v1.1 20190805 Yu Morishita, Uni of Leeds and GSI
 - Bag fix of stc calculation with overlapping
v1.0 20190725 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''

#%% Import
import getopt
import os
import sys
import time
import h5py as h5
import numpy as np
import datetime as dt
import SCM
import LiCSBAS_io_lib as io_lib
import LiCSBAS_inv_lib as inv_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
from LiCSBAS_version import *

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
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    tsadir = []
    memory_size = 4000
    gpu = False
    ransac = False
    cmap_noise_r = 'viridis_r'
    cumfile = False
    
    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "ht:i:",
                                       ["help", "mem_size=", "gpu", "ransac"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-t':
                tsadir = a
            elif o == '-i':
                cumfile = a
            elif o == '--mem_size':
                memory_size = float(a)
            elif o == '--gpu':
                gpu = True
            elif o == '--ransac':
                ransac = True


        if not tsadir:
            raise Usage('No tsa directory given, -d is not optional!')
        elif not os.path.isdir(tsadir):
            raise Usage('No {} dir exists!'.format(tsadir))
        if gpu:
            print("\nGPU option is activated. Need cupy module.\n")
            import cupy as cp

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Directory settings
    tsadir = os.path.abspath(tsadir)
    resultsdir = os.path.join(tsadir,'results')


    #%% Read data information
    if not cumfile:
        cumfile=os.path.join(tsadir,'cum.h5')
        if ransac:
            print('WARNING, using unmasked result (cum.h5) with RANSAC iterations - might take long (not parallel yet)')
    else:
        if not os.path.exists(cumfile):
            print('Error reading specified input file, please fix')
            return 2
    
    cumh5 = h5.File(cumfile, 'r')

    imdates = cumh5['imdates'][()].astype(str).tolist()
    cum = cumh5['cum']
    n_im, length, width = cum.shape

    imdates_dt = [dt.datetime.strptime(imd, '%Y%m%d').toordinal() for imd in imdates]
    dt_cum = np.float32((np.array(imdates_dt)-imdates_dt[0])/365.25)


    #%% Get patch row number
    n_store_data = n_im*2.25+100 #3:cum,data,M(bool); 100:bootnum

    n_patch, patchrow = tools_lib.get_patchrow(width, length, n_store_data, memory_size)


    #%% For each patch
    for i, rows in enumerate(patchrow):
        print('\nProcess {0}/{1}th line ({2}/{3}th patch)...'.format(rows[1], patchrow[-1][-1], i+1, n_patch), flush=True)
        start2 = time.time()

        lengththis = rows[1] - rows[0]

        #%% Calc STC
        print('  Calculating STC...', flush=True)
        ### Read data with extra 1 line for overlapping
        row_ex1 = 0 if i == 0 else 1 ## first patch
        row_ex2 = 0 if i == len(patchrow)-1 else 1 ## last patch

        _cum = cum[:, rows[0]-row_ex1:rows[1]+row_ex2, :].reshape(n_im, lengththis+row_ex1+row_ex2, width)

        ### Calc STC
        stc = inv_lib.calc_stc(_cum, gpu=gpu)[row_ex1:lengththis+row_ex1, :] ## original length
        del _cum

        ### Output data and image
        stcfile = os.path.join(resultsdir, 'stc')

        openmode = 'w' if i == 0 else 'a' #w only 1st patch
        with open(stcfile, openmode) as f:
            stc.tofile(f)


        #%% Calc vstd
        ### Read data for vstd
        n_pt_all = lengththis*width
        cum_patch = cum[:, rows[0]:rows[1], :].reshape((n_im, n_pt_all)).transpose() #(n_pt_all, n_im)

        ### Remove invalid points
        bool_unnan_pt = ~np.isnan(cum_patch[:, 0])

        cum_patch = cum_patch[bool_unnan_pt, :] ## remain only unnan data
        n_pt_unnan = bool_unnan_pt.sum()
        print('  {}/{} points removed due to no data...'.format(n_pt_all-n_pt_unnan, n_pt_all), flush=True)

        ### Calc vstd by bootstrap
        vstd = np.zeros((n_pt_all), dtype=np.float32)*np.nan

        print('  Calculating std of velocity by bootstrap...', flush=True)
        vstd[bool_unnan_pt] = inv_lib.calc_velstd_withnan(cum_patch, dt_cum,
                                                          gpu=gpu)

        ### Output data and image
        vstdfile = os.path.join(resultsdir, 'vstd')

        openmode = 'w' if i == 0 else 'a' #w only 1st patch
        with open(vstdfile, openmode) as f:
                vstd.tofile(f)

        if ransac:
            vel2 = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            intercept2 = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            print('  Recalculating velocity using RANSAC algorithm...', flush=True)
            '''
            for the next release:
            import dask
            n_para = ...
            get_vel_ransac_dask = dask.delayed(inv_lib.get_vel_ransac)
            winsize=(100,dt_cum.shape[0])
            cumda=da.from_array(cum_patch, chunks=winsize)
            vel2int = get_vel_ransac2(dt_cum, cumda, True)
            vel2[bool_unnan_pt], intercept2[bool_unnan_pt] = vel2int.compute(num_workers=n_para)
            '''
            vel2[bool_unnan_pt], intercept2[bool_unnan_pt] = inv_lib.get_vel_ransac(dt_cum, cum_patch, return_intercept=True)
            
            ### Output data and image
            vel2file = os.path.join(resultsdir, 'vel2')
            inter2file = os.path.join(resultsdir, 'intercept2')
            with open(vel2file, openmode) as f:
                vel2.tofile(f)
            with open(inter2file, openmode) as f:
                intercept2.tofile(f)

        #%% Finish patch
        elapsed_time2 = int(time.time()-start2)
        print('  Elapsed time for {0}th patch: {1} sec'.format(i+1, elapsed_time2))


    #%% Close h5 file
    cumh5.close()


    #%% Output png
    print('\nOutput png images...')

    stc = io_lib.read_img(stcfile, length, width)
    pngfile = stcfile+'.png'
    title = 'Spatio-temporal consistency (mm)'
    cmin = np.nanpercentile(stc, 1)
    cmax = np.nanpercentile(stc, 99)
    plot_lib.make_im_png(stc, pngfile, cmap_noise_r, title, cmin, cmax)

    vstd = io_lib.read_img(vstdfile, length, width)
    pngfile = vstdfile+'.png'
    title = 'STD of velocity (mm/yr)'
    cmin = np.nanpercentile(vstd, 1)
    cmax = np.nanpercentile(vstd, 99)
    plot_lib.make_im_png(vstd, pngfile, cmap_noise_r, title, cmin, cmax)
    
    if ransac:
        vel2 = io_lib.read_img(vel2file, length, width)
        pngfile = vel2file+'.png'
        title = 'Outlier-free velocity (mm/yr)'
        cmin = np.nanpercentile(vel2, 1)
        cmax = np.nanpercentile(vel2, 99)
        cmap_vel = SCM.roma.reversed()
        plot_lib.make_im_png(vel2, pngfile, cmap_vel, title, cmin, cmax)


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(tsadir)))


#%% main
if __name__ == "__main__":
    sys.exit(main())
