#!/usr/bin/env python3
"""
========
Overview
========
This script:
 - copies file 130cum*.h5 to cum.h5
 - copies folder 130results* to results
 - calcs n_unw, n_loop_err, coh_avg
 - assumbles all results into cum.h5

===============
Input & output files
===============

Inputs in GEOCml*/ (--comp_cc_dir):
 - slc.mli.par
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.cc

Inputs in GEOCml*/ (--unw_dir):
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw

Inputs in TS_GEOCml*/ :
 - 130cum*.h5            : Cumulative displacement (time-seires) in mm from final iteration, to copy to cum.h5

 - 130results*           : Results from final iteration of automatic correction, to copy to results
    -vel[.png]           : Velocity in mm/yr (positive means LOS decrease; uplift)
   - vintercept[.png]    : Constant part of linear velocity (c for vt+c) in mm
   - resid_rms[.png]     : RMS of residual in mm
   - n_gap[.png]         : Number of gaps in SB network
   - n_ifg_noloop[.png]  : Number of ifgs with no loop
   - maxTlen[.png]       : Max length of continous SB network in year

 - info
   - 120ref.txt          : Reference window

Outputs in TS_GEOCml*/ :
 - cum.h5                : Cumulative displacement (time-seires) in mm

 - results/
   - n_unw[.png]         : Number of available unwrapped data to be used
   - coh_avg[.png]       : Average coherence
   - n_loop_err[.png]    : Number of remaining loop errors (>pi) in data to be used

=====
Usage
=====
LiCSBAS133_write_h5.py [-h] [-f FRAME_DIR] [-c COMP_CC_DIR] [-t TS_DIR] [--suffix SUFFIX]
"""

#%% Change log
'''
v1.0 20220928 Qi Ou, Leeds Uni
'''

#%% Import
import os
import time
import shutil
import numpy as np
import h5py as h5
from pathlib import Path
import argparse
import sys
import re
import xarray as xr
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_loop_lib as loop_lib
import LiCSBAS_plot_lib as plot_lib


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    '''
    Use a multiple inheritance approach to use features of both classes.
    The ArgumentDefaultsHelpFormatter class adds argument default values to the usage help message
    The RawDescriptionHelpFormatter class keeps the indentation and line breaks in the ___doc___
    '''
    pass


def init_args():
    global args
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument('-f', dest='frame_dir', default="./", help="directory of LiCSBAS output")
    # parser.add_argument('-d', dest='unw_dir', help="folder containing unw input to be corrected, if not given, use $comp_cc_dir_$suffix")
    parser.add_argument('-c', dest='comp_cc_dir', default="GEOCml10GACOS", help="folder containing connected components and coherence files")
    parser.add_argument('-t', dest='ts_dir', default="TS_GEOCml10GACOS", help="folder containing time series")
    parser.add_argument('-l', dest='ifg_list', default=None, type=str, help="text file containing a list of ifgs, if not given, all ifgs in -c are read")
    parser.add_argument('--suffix', default="", type=str, help="suffix of the final iteration")
    parser.add_argument('--stay', default=False, action='store_true', help="don't copy to results, save everything in results$suffix")

    args = parser.parse_args()



def start():
    global start_time
    # intialise and print info on screen
    start_time = time.time()
    ver="1.0"; date=20221020; author="Qi Ou"
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)


def finish():
    #%% Finish
    elapsed_time = time.time() - start_time
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))
    print("\n{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)
    print('Output directory: {}\n'.format(os.path.relpath(tsadir)))


def set_input_output():
    global ccdir, ifgdir, tsadir, infodir, last_result_dir, resultsdir, last_cumh5file, cumh5file, ifgdates

    # define input directories and file
    ccdir = os.path.abspath(os.path.join(args.frame_dir, args.comp_cc_dir))
    if args.suffix == "1":
        ifgdir = os.path.abspath(os.path.join(args.frame_dir, args.comp_cc_dir))
    else:
        ifgdir = os.path.abspath(os.path.join(args.frame_dir, args.comp_cc_dir+args.suffix))

    if args.ifg_list:
        ifgdates = io_lib.read_ifg_list(args.ifg_list)
    else:
        ifgdates = tools_lib.get_ifgdates(ifgdir)

    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    infodir = os.path.join(tsadir, 'info')
    last_result_dir = os.path.join(tsadir, '130results{}'.format(args.suffix))
    last_cumh5file = os.path.join(tsadir, '130cum{}.h5'.format(args.suffix))

    # define output directory and file
    if args.stay:
        resultsdir = last_result_dir
        cumh5file = last_cumh5file
    else:
        resultsdir = os.path.join(tsadir, 'results')
        cumh5file = os.path.join(tsadir, 'cum.h5')
        # copy everything from last iter to final
        shutil.copyfile(last_cumh5file, cumh5file)
        shutil.copytree(last_result_dir, resultsdir, dirs_exist_ok=True)


def read_length_width():
    global length, width
    # read ifg size
    mlipar = os.path.join(ccdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))


def calc_n_unw():
    print("Computing number of unw per pixel...")
    n_unw = np.zeros((length, width), dtype=np.float32)
    for ifgd in ifgdates:
        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        unw = io_lib.read_img(unwfile, length, width)
        unw[unw == 0] = np.nan # Fill 0 with nan
        n_unw += ~np.isnan(unw) # Summing number of unnan unw


    ### Write to file
    n_unw[n_unw == 0] = np.nan
    n_unwfile = os.path.join(resultsdir, 'n_unw')
    n_unw.tofile(n_unwfile)

    ### Save png
    title = 'Number of used unw data'
    cmap_noise = 'viridis'
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    n_im = len(imdates)
    plot_lib.make_im_png(n_unw, n_unwfile+'.png', cmap_noise, title, n_im)

    return n_unw


def calc_coh_avg():
    print("Computing average coherence...")
    # calc n_unw and avg_coh of final data set
    coh_avg = np.zeros((length, width), dtype=np.float32)
    n_coh = np.zeros((length, width), dtype=np.int16)
    for ifgd in ifgdates:
        ccfile = os.path.join(ccdir, ifgd, ifgd+'.cc')
        if os.path.getsize(ccfile) == length*width:
            coh = io_lib.read_img(ccfile, length, width, np.uint8)
            coh = coh.astype(np.float32)/255
        else:
            coh = io_lib.read_img(ccfile, length, width)
            coh[np.isnan(coh)] = 0  # Fill nan with 0
        coh_avg += coh
        n_coh += (coh!=0)
    coh_avg[n_coh==0] = np.nan
    n_coh[n_coh==0] = 1 #to avoid zero division
    coh_avg = coh_avg/n_coh
    coh_avg[coh_avg==0] = np.nan

    ### Write to file
    coh_avgfile = os.path.join(resultsdir, 'coh_avg')
    coh_avg.tofile(coh_avgfile)

    ### Save png
    title = 'Average coherence'
    cmap_noise = 'viridis'
    plot_lib.make_im_png(coh_avg, coh_avgfile+'.png', cmap_noise, title)


def calc_n_loop_error(n_unw):
    ''' same as loop_closure_4th in LiCSBAS12_loop_closure.py '''
    print('Compute n_loop_error...', flush=True)

    # read reference
    reffile = os.path.join(infodir, '120ref.txt')
    with open(reffile, "r") as f:
        refarea = f.read().split()[0]  # str, x1/x2/y1/y2
    refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]

    # create 3D cube - False means presumed error in the loop
    a = np.full((length, width,len(ifgdates)), False)
    da = xr.DataArray(
        data=a,
        dims=[ "y", "x", "ifgd"],
        coords=dict(y=np.arange(length), x=np.arange(width), ifgd=ifgdates))

    ### Get loop matrix
    Aloop = loop_lib.make_loop_matrix(ifgdates)
    n_loop = Aloop.shape[0]

    ### Count loop error by pixel
    n_loop_err = np.zeros((length, width), dtype=np.float32)
    for i in range(0, len(Aloop)):
        if np.mod(i, 100) == 0:
            print("  {0:3}/{1:3}th loop...".format(i, n_loop), flush=True)

        ### Read unw
        unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = loop_lib.read_unw_loop_ph(Aloop[i, :], ifgdates, ifgdir, length, width)

        ## Compute ref
        ref_unw12 = np.nanmean(unw12[refy1:refy2, refx1:refx2])
        ref_unw23 = np.nanmean(unw23[refy1:refy2, refx1:refx2])
        ref_unw13 = np.nanmean(unw13[refy1:refy2, refx1:refx2])

        ## Calculate loop phase taking into account ref phase
        loop_ph = unw12 + unw23 - unw13 - (ref_unw12 + ref_unw23 - ref_unw13)

        ## Count number of loops with suspected unwrap error (>pi)
        loop_ph[np.isnan(loop_ph)] = 0  # to avoid warning
        is_ok = np.abs(loop_ph) < np.pi
        da.loc[:, :, ifgd12] = np.logical_or(da.loc[:, :, ifgd12], is_ok)
        da.loc[:, :, ifgd23] = np.logical_or(da.loc[:, :, ifgd23], is_ok)
        da.loc[:, :, ifgd13] = np.logical_or(da.loc[:, :, ifgd13], is_ok)
        n_loop_err = n_loop_err + ~is_ok  # suspected unw error

    # write to file
    n_loop_err[np.isnan(n_unw)] = np.nan
    # n_loop_err = np.array(n_loop_err)
    n_loop_err_file = os.path.join(resultsdir, 'n_loop_err')
    n_loop_err.tofile(n_loop_err_file)

    # save png
    title = 'Number of unclosed loops'
    cmap_noise_r = 'viridis_r'
    plot_lib.make_im_png(n_loop_err, n_loop_err_file+'.png', cmap_noise_r, title)


def write_h5():
    # Write additional results to h5
    print('\nWriting to HDF5 file...')
    cumh5 = h5.File(cumh5file, 'a')
    compress = 'gzip'
    indices = ['coh_avg', 'hgt', 'n_loop_err', 'n_unw', 'slc.mli']

    for index in indices:
        print(index)
        file = os.path.join(resultsdir, index)
        if os.path.exists(file):
            data = io_lib.read_img(file, length, width)
            cumh5.create_dataset(index, data=data, compression=compress)
        else:
            print('  {} not exist in results dir. Skip'.format(index))

    LOSvecs = ['E.geo', 'N.geo', 'U.geo']
    for LOSvec in LOSvecs:
        file = os.path.join(ccdir, LOSvec)
        if os.path.exists(file):
            data = io_lib.read_img(file, length, width)
            cumh5.create_dataset(LOSvec, data=data, compression=compress)
        else:
            print('  {} not exist in GEOCml dir. Skip'.format(LOSvec))

    cumh5.close()


def main():
    global ifgdates

    # intialise
    start()
    init_args()

    # directory settings
    set_input_output()
    read_length_width()

    # calc quality stats based on the final corrected unw
    n_unw = calc_n_unw()
    calc_coh_avg()
    calc_n_loop_error(n_unw)

    # compile all results to h5
    write_h5()

    # report finish
    finish()


if __name__ == '__main__':
    main()





