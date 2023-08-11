#!/usr/bin/env python3
"""
v2.0.0 20220706 Jack McGrath, Leeds Uni
v1.5.4 20221020 Qi Ou, Leeds Uni
v1.5.3 20211122 Milan Lazecky, Leeds Uni
v1.5.2 20210311 Yu Morishita, GSI

This script will take the statistics options from LiCSBAS130_sb_inv.py, and identifies
and removes pixels that are not associated with a loop closure. Aggressive nullification
will produce more no_loop_pixels, so run this before that to remove/mask pixels that
natively have no loops.

===============
Input & output files
===============
Inputs in GEOCml*/ ():
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.cc
 - EQA.dem_par
 - slc.mli.par

Inputs in GEOCml*/ (r):
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw

Outputs in GEOCml*/:
 - yyyymmdd_yyyymmdd/ :
    - *.unw :   unw with noloop pixels nulled
    - *_orig.unw : original, unnulled unw

 - no_loop_ifg/ :
    - yyyymmdd_yyyymmdd/ : Isolated interferograms fully not in any loop   

Outputs in TS_GEOCml*/ :
 - 130results*/
   - n_gap*[.png]      : Number of gaps in SB network
   - n_ifg_noloop*[.png] :  Number of ifgs with no loop
   - maxTlen*[.png]    : Max length of continous SB network in year

=====
Usage
=====
LiCSBAS130_sb_inv.py -d ifgdir [--n_para int] [--n_unw_r_thre float] [--null_mask] [--backup]

"""
#%% Change log
'''
v2.0.0 20230706 Jack McGrath, Uni of Leeds
 - Re-edit for only nulling no-loop pixels
v1.0 20190730 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''
# - network/network13*.png : Figures of the network

#%% Import
import getopt
import os
import sys
import re
import time
import psutil
import h5py as h5
import numpy as np
import datetime as dt
import multiprocessing as multi
import SCM
import LiCSBAS_io_lib as io_lib
import LiCSBAS_inv_lib as inv_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_loop_lib as loop_lib
import LiCSBAS_plot_lib as plot_lib
import argparse
import shutil


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    '''
    Use a multiple inheritance approach to use features of both classes.
    The ArgumentDefaultsHelpFormatter class adds argument default values to the usage help message
    The RawDescriptionHelpFormatter class keeps the indentation and line breaks in the ___doc___
    '''
    pass


class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


def init_args():
    # read inputs
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument('-f', dest='frame_dir', default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-d', dest='ifg_dir', default="GEOCml10GACOS", help="folder containing unw files")
    parser.add_argument('-t', dest='ts_dir', default="TS_GEOCml10GACOS", help="folder containing time series")
    parser.add_argument('-m', dest='memory_size', default=2048, type=float, help="Max memory size for each patch in MB")
    parser.add_argument('-l', dest='ifg_list', default=None, type=str, help="text file containing a list of ifgs")
    parser.add_argument('--n_unw_r_thre', metavar="THRE", type=float, help="Threshold of n_unw (number of used unwrap data) \n (Note this value is ratio to the number of images; i.e., 1.5*n_im) \n Larger number (e.g. 2.5) makes processing faster but result sparser. \n (Default: 1 and 0.5 for C- and L-band, respectively)")
    parser.add_argument('--n_para', type=int, help="Number of parallel processing (Default: # of usable CPU)")
    parser.add_argument('--null_mask', dest='nullify', default=True, action='store_false', help="Create mask of no-loop pixels rather than null")
    parser.add_argument('--backup', dest='backup', default=False, action='store_true', help="Create a backup of the un-nulled data")
    args = parser.parse_args()

    return args


def main():

    start = time.time()
    ver="1.0"; date=20220929; author="Q. Ou and Jack McGrath"
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)

    ## For parallel processing
    global n_para_gap, G, Aloop, unwpatch, imdates, incdir, ifgdir, length, width,\
        coef_r2m, ifgdates, ref_unw, cycle, keep_incfile, resdir, restxtfile, \
        cmap_vel, cmap_wrap, wavelength, args

    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()

    os.environ["OMP_NUM_THREADS"] = "1"
    # Because np.linalg.lstsq use full CPU but not much faster than 1CPU.
    # Instead parallelize by multiprocessing

    cmap_vel = SCM.roma.reversed()
    cmap_noise = 'viridis'
    cmap_noise_r = 'viridis_r'
    cmap_wrap = SCM.romaO
    q = multi.get_context('fork')

    args = init_args()
    # Check if assigned number of cores is greater than those availiable
    if args.n_para:
        if args.n_para < n_para:
            n_para = args.n_para

    # define input directories
    ifgdir = os.path.abspath(os.path.join(args.frame_dir, args.ifg_dir))  # to read .cc
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))   # to read 120.ref, to write cum.h5

    # define output directories and files
    resultsdir = os.path.join(tsadir, 'results')  # to save vel, vintercept, rms etc
    if not os.path.exists(resultsdir):
        os.mkdir(resultsdir)

    if n_para > 40:
        n_para = 40 # Don't fully jam the server

    #%% Read data information
    ### Get size
    mlipar = os.path.join(ifgdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    speed_of_light = 299792458 #m/s
    radar_frequency = float(io_lib.get_param_par(mlipar, 'radar_frequency')) #Hz
    wavelength = speed_of_light/radar_frequency #meter
    coef_r2m = -wavelength/4/np.pi*1000 #rad -> mm, positive is -LOS

    ### Set n_unw_r_thre and cycle depending on L- or C-band
    if wavelength > 0.2: ## L-band
        if args.n_unw_r_thre is None:
            n_unw_r_thre = 0.5
        else:
            n_unw_r_thre = args.n_unw_r_thre
        cycle = 1.5 # 2pi/cycle for comparison png
    elif wavelength <= 0.2: ## C-band
        if args.n_unw_r_thre is None:
            n_unw_r_thre = 1.0
        else:
            n_unw_r_thre = args.n_unw_r_thre
        cycle = 3 # 3*2pi/cycle for comparison png
    else: # any other
        if args.n_unw_r_thre is None:
            n_unw_r_thre = 1.0
        else:
            n_unw_r_thre = args.n_unw_r_thre
        cycle = 3 # 3*2pi/cycle for comparison png
    #%% Read date and network information
    ### Get all ifgdates in ifgdir
    if args.ifg_list:
        ifgdates = io_lib.read_ifg_list(args.ifg_list)
    else:
        ifgdates = tools_lib.get_ifgdates(ifgdir)

    ### Construct G and Aloop matrix for increment and n_gap
    G = inv_lib.make_sb_matrix(ifgdates)
    Aloop = loop_lib.make_loop_matrix(ifgdates)

    ### Extract no loop ifgs
    ns_loop4ifg = np.abs(Aloop).sum(axis=0)
    ixs_ifg_no_loop = np.where(ns_loop4ifg==0)[0]
    no_loop_ifg = [ifgdates[ix] for ix in ixs_ifg_no_loop]

    no_loop_dir = os.path.join(ifgdir, 'no_loop_ifg')
    if not os.path.exists(no_loop_dir):
        os.mkdir(no_loop_dir)

    print('Moving {} no loop ifgs to no_loop_ifg'.format(len(no_loop_ifg)))
    for ifg in no_loop_ifg:
        shutil.move(os.path.join(ifgdir, ifg), os.path.join(no_loop_dir, ifg))

    # Reset IFG dates list
    ifgdates = list(set(ifgdates) - set(no_loop_ifg)).sort()

    imdates = tools_lib.ifgdates2imdates(ifgdates)
    n_ifg = len(ifgdates)
    n_im = len(imdates)
    n_unw_thre = int(n_unw_r_thre*n_im)

    ### Calc dt in year
    imdates_dt = ([dt.datetime.strptime(imd, '%Y%m%d').toordinal() for imd in imdates])
    dt_cum = np.float32((np.array(imdates_dt)-imdates_dt[0])/365.25)

    #%% Get patch row number
    ### Check RAM
    mem_avail = (psutil.virtual_memory().available)/2**20 #MB
    if args.memory_size > mem_avail/2:
        print('\nNot enough memory available compared to mem_size ({} MB).'.format(args.memory_size))
        print('Reduce mem_size automatically to {} MB.'.format(int(mem_avail/2)))
        memory_size = int(mem_avail/2)
    else:
        memory_size = args.memory_size

    ### Determine if read cum on memory (fast) or hdf5 (slow)
    cum_size = int(n_im*length*width*4/2**20) #MB
    if memory_size > cum_size*2:
        print('Read cum data on memory (fast but need memory).')
        save_mem = False # read on memory
        memory_size_patch = memory_size - cum_size
    else:
        print('Read cum data in HDF5 (save memory but slow).')
        save_mem = True # read on hdf5
        memory_size_patch = memory_size

    n_store_data = n_ifg*2+n_im*2+n_im*0.3 #not sure

    n_patch, patchrow = tools_lib.get_patchrow(width, length, n_store_data, memory_size_patch)

    #%% Display and output settings & parameters
    print('')
    print('Size of image (w,l)    : {}, {}'.format(width, length))
    print('# of images to be used : {}'.format(n_im))
    print('# of ifgs to be used   : {}'.format(n_ifg))
    print('Threshold of used unw  : {}'.format(n_unw_thre))
    print('')
    print('Allowed memory size    : {} MB'.format(memory_size))
    print('Number of patches      : {}'.format(n_patch))

    # names for layers to be stored/pngs generated
    names = ['n_gap', 'n_ifg_noloop', 'maxTlen']

    #%% For each patch
    for i_patch, rows in enumerate(patchrow):
        print('\nProcess {0}/{1}th line ({2}/{3}th patch)...'.format(rows[1], patchrow[-1][-1], i_patch+1, n_patch), flush=True)
        start2 = time.time()

        #%% Read data
        ### Allocate memory
        lengththis = rows[1] - rows[0]
        n_pt_all = lengththis*width
        unwpatch = np.zeros((n_ifg, lengththis, width), dtype=np.float32)

        ### For each ifg
        print("  Reading {0} ifg's unw data...".format(n_ifg), flush=True)
        countf = width*rows[0]
        countl = width*lengththis
        for i, ifgd in enumerate(ifgdates):
            suffix = '.unw'
            if i_patch == 0:
                unwfile = os.path.join(ifgdir, ifgd, ifgd + '.unw')
                if args.nullify and args.backup:
                    ## Check for backed up data. THIS ASSUMES ALL DATA HAS BEEN NULLED THE SAME WAY
                    trueorigfile = os.path.join(ifgdir, ifgd, ifgd + '_orig.unw')
                    # If no trueorigfile exists, data is truely untouched
                    if not os.path.exists(trueorigfile):
                        shutil.move(unwfile, trueorigfile)
                        suffix = '_orig13.unw'
                        origfile = os.path.join(ifgdir, ifgd, ifgd + suffix)
                        # Softlink (used for checking if null has already occurred for LiCSBAS12, softlink to save space)
                        os.symlink(trueorigfile, origfile)
                    else:
                        # True orig exists - check if it is because loop err nullification has occurred
                        suffix = '_orig12.unw'
                        origfile = os.path.join(ifgdir, ifgd, ifgd + suffix)
                        if os.path.exists(origfile):
                            # orig12 exists - backup unw as orig1213
                            suffix = '_orig1213.unw'
                            origfile = os.path.join(ifgdir, ifgd, ifgd + suffix)
                        else:
                            # orig12 doesn't exist. There should therefore be orig13 so no need to backup
                            suffix = '_orig13.unw'
                            origfile = os.path.join(ifgdir, ifgd, ifgd + suffix)
                        shutil.move(unwfile, origfile)
            
            datafile = os.path.join(ifgdir, ifgd, ifgd + suffix)
            f = open(datafile, 'rb')
            f.seek(countf*4, os.SEEK_SET) #Seek for >=2nd patch, 4 means byte

            ### Read unw data (mm) at patch area
            unw = np.fromfile(f, dtype=np.float32, count=countl).reshape((lengththis, width))*coef_r2m
            unw[unw == 0] = np.nan # Fill 0 with nan
            unwpatch[i] = unw
            f.close()

        unwpatch = unwpatch.reshape((n_ifg, n_pt_all)).transpose() #(n_pt_all, n_ifg)
        #%% Remove points with less valid data than n_unw_thre
        ix_unnan_pt = np.where(np.sum(~np.isnan(unwpatch), axis=1) > n_unw_thre)[0]
        n_pt_unnan = len(ix_unnan_pt)
        unwpatch = unwpatch[ix_unnan_pt,:] ## keep only unnan data

        print('  {}/{} points removed due to not enough ifg data (e.g sea, incoherent areas)...'.format(n_pt_all-n_pt_unnan, n_pt_all), flush=True)

        #%% Compute number of gaps, ifg_noloop, maxTlen point-by-point
        if n_pt_unnan != 0:
            ns_gap_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            gap_patch = np.zeros((n_im-1, n_pt_all), dtype=np.int8)
            ns_ifg_noloop_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            maxTlen_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan

            print('\n  Identifing gaps, and counting n_gap and n_ifg_noloop')

            ### Determine n_para
            n_pt_patch_min = 1000
            if n_pt_patch_min*n_para > n_pt_unnan:
                ## Too much n_para
                n_para_gap = int(np.floor(n_pt_unnan/n_pt_patch_min))
                if n_para_gap == 0:
                    n_para_gap = 1
                if n_para_gap > n_para:
                    n_para_gap = n_para
            else:
                n_para_gap = n_para


            print('  with {} parallel processing...'.format(n_para_gap),
                  flush=True)

            ### Devide unwpatch by n_para for parallel processing
            p = q.Pool(n_para_gap)
            _result = np.array(p.map(null_noloop_wrapper, range(n_para_gap)),
                               dtype=object)
            p.close()

            ns_gap_patch[ix_unnan_pt] = np.hstack(_result[:, 0]) #n_pt
            gap_patch[:, ix_unnan_pt] = np.hstack(_result[:, 1]) #n_im-1, n_pt
            ns_ifg_noloop_patch[ix_unnan_pt] = np.hstack(_result[:, 2])
            no_loop_log = np.hstack(_result[:, 3]).T

            if args.nullify:
                print('\n  Removing {} IFG pixels that have no loop'.format(np.sum((no_loop_log == 1).flatten() > 0)))
                unwpatch[np.where(no_loop_log == 1)] = np.nan
                null_patch = np.zeros((n_ifg, n_pt_all), dtype=np.float32)*np.nan
                null_patch[:, ix_unnan_pt] = unwpatch.T

            ### maxTlen
            _maxTlen = np.zeros((n_pt_unnan), dtype=np.float32) #temporaly
            _Tlen = np.zeros((n_pt_unnan), dtype=np.float32) #temporaly
            for imx in range(n_im-1):
                _Tlen = _Tlen + (dt_cum[imx+1]-dt_cum[imx]) ## Adding dt
                _Tlen[gap_patch[imx, ix_unnan_pt]==1] = 0 ## reset to 0 if gap
                _maxTlen[_maxTlen<_Tlen] = _Tlen[_maxTlen<_Tlen] ## Set Tlen to maxTlen
            maxTlen_patch[ix_unnan_pt] = _maxTlen

        #%% Fill by np.nan if n_pt_unnan == 0
        else:
            gap_patch = np.zeros((n_im-1, n_pt_all), dtype=np.int8)
            ns_gap_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            ns_ifg_noloop_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            maxTlen_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
            null_patch = np.zeros((n_ifg, n_pt_all), dtype=np.float32) * np.nan


        #%% Output data and image
        #unwnull[:, rows[0]:rows[1], :] = unwpatch.reshape((n_ifg, lengththis, width))

        ### Others
        openmode = 'w' if rows[0] == 0 else 'a' #w only 1st patch

        if args.nullify:
            ## For each ifg
            for i, ifgd in enumerate(ifgdates):
                ifgfile = os.path.join(ifgdir, ifgd, '{0}.unw'.format(ifgd))
                # Write patch to file
                with open(ifgfile, openmode) as f:
                    null_patch[i, :].astype(np.float32).tofile(f)

        ## velocity and noise indicies in results dir
        # names = ['n_gap', 'n_ifg_noloop', 'maxTlen'] # had to be defined earlier
        data = [ns_gap_patch, ns_ifg_noloop_patch, maxTlen_patch]
        for i in range(len(names)):
            file = os.path.join(resultsdir, '{}_preNullNoLoop'.format(names[i]))
            with open(file, openmode) as f:
                data[i].astype(np.float32).tofile(f)

        #%% Finish patch
        elapsed_time2 = int(time.time()-start2)
        hour2 = int(elapsed_time2/3600)
        minite2 = int(np.mod((elapsed_time2/60),60))
        sec2 = int(np.mod(elapsed_time2,60))
        print("  Elapsed time for {0}th patch: {1:02}h {2:02}m {3:02}s".format(i_patch+1, hour2, minite2, sec2), flush=True)

    if args.nullify:
        ## Write zeros file for n_no_loop for later masking
        zero_no_loop = np.zeros((length, width))
        file = os.path.join(resultsdir, 'n_ifg_noloop_preNullNoLoop')
        data = io_lib.read_img(file, length, width)
        zero_no_loop[np.where(np.isnan(data))] = np.nan
        file = os.path.join(resultsdir, 'n_ifg_no_loop')
        with open(file, 'w') as f:
            zero_no_loop.tofile(f)

    #%% Output png images
    ### Velocity and noise indices
    cmins = [None, None, None]
    cmaxs = [None, None, None]
    cmaps = [cmap_noise_r, cmap_noise_r, cmap_noise]
    titles = ['Number of gaps in SB network', 'Number of ifgs with no loops', 'Max length of connected SB network (yr)']

    if args.nullify:
        labels = ['_preNullNoLoop', ' before null_no_loop']
    else:
        labels = ['','']

    print('\nOutput noise png images...', flush=True)
    for i in range(len(names)):
        file = os.path.join(resultsdir, '{}{}}'.format(names[i], labels[0]))
        data = io_lib.read_img(file, length, width)
        pngfile = file + '.png'
        ## Get color range if None
        if cmins[i] is None:
            cmins[i] = np.nanpercentile(data, 1)
        if cmaxs[i] is None:
            cmaxs[i] = np.nanpercentile(data, 99)
        if cmins[i] == cmaxs[i]: cmins[i] = cmaxs[i]-1
        # print(pngfile)
        plot_lib.make_im_png(data, pngfile, cmaps[i], '{}'.format(titles[i], labels[1]), cmins[i], cmaxs[i])

    ### New unw pngs
    if n_para > 1 and n_ifg > 20:
        pool = q.Pool(processes=n_para)
        pool.map(null_png_wrapper, even_split(ifgdates, n_para))
    else:
        null_png_wrapper(range(len(n_ifg)))


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(sys.argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(tsadir)))

def null_noloop_wrapper(i):
    print("    Running {:2}/{:2}th patch...".format(i+1, n_para_gap), flush=True)
    n_pt_patch = int(np.ceil(unwpatch.shape[0]/n_para_gap))
    n_im = G.shape[1]+1
    n_loop, n_ifg = Aloop.shape

    if i*n_pt_patch >= unwpatch.shape[0]:
        # Nothing to do
        return

    ### n_gap and gap location
    ns_unw_unnan4inc = np.array([(G[:, j]*
                          (~np.isnan(unwpatch[i*n_pt_patch:(i+1)*n_pt_patch])))
                         .sum(axis=1, dtype=np.int16) for j in range(n_im-1)])
                    #n_ifg*(n_pt,n_ifg) -> (n_im-1,n_pt)
    _ns_gap_patch = (ns_unw_unnan4inc==0).sum(axis=0) #n_pt
    _gap_patch = ns_unw_unnan4inc==0

    del ns_unw_unnan4inc

    ### n_ifg_noloop
    # n_ifg*(n_pt,n_ifg)->(n_loop,n_pt)
    # Number of ifgs for each loop at eath point.
    # 3 means complete loop, 1 or 2 means broken loop.
    ns_ifg4loop = np.array([(np.abs(Aloop[j, :])*
                         (~np.isnan(unwpatch[i*n_pt_patch:(i+1)*n_pt_patch])))
                            .sum(axis=1) for j in range(n_loop)])
    bool_loop = (ns_ifg4loop==3) #(n_loop,n_pt) identify complete loop only
    #bad_bool_loop = (ns_ifg4loop != 3) # Identify incomplete loops (Loop x Pixel)
    del ns_ifg4loop

    # n_loop*(n_loop,n_pt)*n_pt->(n_ifg,n_pt)
    # Number of loops for each ifg at each point.
    ns_loop4ifg = np.array([(
            (np.abs(Aloop[:, j])*bool_loop.T).T*
            (~np.isnan(unwpatch[i*n_pt_patch:(i+1)*n_pt_patch, j]))
            ).sum(axis=0) for j in range(n_ifg)]) #    <= This is the variable that contains the number of loops for each IFG for each pixel
    del bool_loop
    ns_ifg_noloop_ix = (ns_loop4ifg == 0).astype('int') # Int array of n_ifg x n_px of px with no loops
    ns_ifg_noloop_tmp = ns_ifg_noloop_ix.sum(axis=0) # Number of incomplete loops per pixel
    #del ns_loop4ifg

    ns_nan_ifg = np.isnan(unwpatch[i*n_pt_patch:(i+1)*n_pt_patch, :]).sum(axis=1)
    #n_pt, nan ifg count
    _ns_ifg_noloop_patch = ns_ifg_noloop_tmp - ns_nan_ifg # IFGs with no loop = IFGs with no loop - IFGs that are NaNs anyway (number no_loop per pixel)

    return _ns_gap_patch, _gap_patch, _ns_ifg_noloop_patch, ns_ifg_noloop_ix

def null_png_wrapper(ifglist):
    for ifgd in ifglist:
        infile = os.path.join(ifgdir, ifgd, '{}.unw'.format(ifgd))
        unw = io_lib.read_img(infile, length, width)

        pngfile = infile+'_noLoop_null.png'
        title = '{} (NoLoop Pixels Nulled)'.format(ifgd)
        plot_lib.make_im_png(np.angle(np.exp(1j*(unw/coef_r2m/cycle))*cycle), pngfile, cmap_wrap, title, vmin=-np.pi, vmax=np.pi, cbar=False)

def even_split(a, n):
    """ Divide a list, a, in to n even parts"""
    n = min(n, len(a)) # to avoid empty lists
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


if __name__ == "__main__":
    sys.exit(main())
