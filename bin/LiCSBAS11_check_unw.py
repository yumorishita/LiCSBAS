#!/usr/bin/env python3
"""
========
Overview
========
This script checks quality of unw data and identifies bad interferograms based on average coherence and coverage of the unw data. This also prepares a time series working directory (overwrite if already exists).
This script also identifies coregistration error, which looks like a ramp in azimuth, based on the absolute slope and R-squares obtained from a linear fit to the pixels in the middle column o the ifg

===============
Input & output files
===============
Inputs in GEOCml*/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw[.png]
   - yyyymmdd_yyyymmdd.cc
 - slc.mli[.par|.png]
 - baselines (can be dummy)
 - EQA.dem_par

 Outputs in TS_GEOCml*/ :
 - info/
   - 11bad_ifg.txt    : List of bad ifgs discarded from further processing
   - 11ifg_stats.txt  : Statistics of interferograms
   - EQA.dem_par (copy)
   - slc.mli.par (copy)
 - results/
   - slc.mli[.png] (copy, if exist)
   - hgt[.png, U] (copy, if exist)
 - 11bad_ifg_ras/yyyymmdd_yyyymmdd.unw.png : png of bad ifgs
 - 11ifg_ras/yyyymmdd_yyyymmdd.unw.png     : png of good ifgs
 - network/network11*.png  : Figures of baseline configuration

=====
Usage
=====
LiCSBAS11_check_unw.py -d ifgdir [-t tsadir] [-c coh_thre] [-u unw_thre] [-s]

 -d  Path to the GEOCml* dir containing stack of unw data.
 -t  Path to the output TS_GEOCml* dir. (Default: TS_GEOCml*)
 -c  Threshold of average coherence (Default: 0.05)
 -u  Threshold of coverage of unw data (Default: 0.3)
 -s  Check for coregistration error in the form of a significant azimuthal ramp

"""
#%% Change log
'''
v1.4 20221011 Qi Ou, Uni of Leeds
 - Detect coregistration error as big azimuthal ramp in the middle (arbitrary) column
 - Shortlist by slope > 30, R2 > 0.95, then expand based on repeated epochs in the shortlist, threshold with slopd > 20
v1.3.4 20211129 Milan Lazecky, Uni of Leeds
 - Extra check on file dimensions - happens if LiCSAR data is inconsistent - should be moved to previous step
v1.3.3 20210402 Yu Morishita, GSI
 - Treat all nan as bad ifg
 - Raise error if all ifgs are bad
v1.3.2 20201116 Yu Morishita, GSI
 - Exit if suffix is not set
v1.3.1 20200911 Yu Morishita, GSI
 - Change default to -c 0.05 -u 0.3
v1.3 20200703 Yu Morishita, GSI
 - Replace problematic terms
v1.2 20200225 Yu Morishita, Uni of Leeds and GSI
 - Not output network pdf
 - Deal with cc file in uint8 format
v1.1 20191115 Yu Morishita, Uni of Leeds and GSI
 - Add hgt
v1.0 20190729 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''

#%% Import
import getopt
import os
import sys
import time
import shutil
import numpy as np
import datetime as dt
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
from scipy import stats
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
    ifgdir = []
    tsadir = []
    coh_thre = 0.05
    unw_cov_thre = 0.3
    check_coreg_slope = False

    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hd:t:c:u:s", ["help"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-d':
                ifgdir = a
            elif o == '-t':
                tsadir = a
            elif o == '-c':
                coh_thre = float(a)
            elif o == '-u':
                unw_cov_thre = float(a)
            elif o == '-s':
                check_coreg_slope = True

        if not ifgdir:
            raise Usage('No data directory given, -d is not optional!')
        elif not os.path.isdir(ifgdir):
            raise Usage('No {} dir exists!'.format(ifgdir))
        elif not os.path.exists(os.path.join(ifgdir, 'slc.mli.par')):
                raise Usage('No slc.mli.par file exists in {}!'.format(ifgdir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2

    print("\ncoh_thre     : {}".format(coh_thre), flush=True)
    print("unw_cov_thre : {}".format(unw_cov_thre), flush=True)


    #%% Directory setting
    ifgdir = os.path.abspath(ifgdir)

    if not tsadir:
        tsadir = os.path.join(os.path.dirname(ifgdir), 'TS_'+os.path.basename(ifgdir))

    if not os.path.exists(tsadir): os.mkdir(tsadir)

    ifg_rasdir = os.path.join(tsadir, '11ifg_ras')
    bad_ifg_rasdir = os.path.join(tsadir, '11bad_ifg_ras')

    if os.path.exists(ifg_rasdir): shutil.rmtree(ifg_rasdir)
    if os.path.exists(bad_ifg_rasdir): shutil.rmtree(bad_ifg_rasdir)
    os.mkdir(ifg_rasdir)
    os.mkdir(bad_ifg_rasdir)

    netdir = os.path.join(tsadir, 'network')
    if not os.path.exists(netdir): os.mkdir(netdir)

    infodir = os.path.join(tsadir, 'info')
    if not os.path.exists(infodir): os.mkdir(infodir)

    resultsdir = os.path.join(tsadir, 'results')
    if not os.path.exists(resultsdir): os.mkdir(resultsdir)


    ### Get size
    mlipar = os.path.join(ifgdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    print("\nSize         : {} x {}".format(width, length), flush=True)

    ### Get resolution
    dempar = os.path.join(ifgdir, 'EQA.dem_par')
    lattitude_resolution = float(io_lib.get_param_par(dempar, 'post_lat'))

    ### Check for corrupted or wrong size unws - remove from ifgdir
    ifgdates = tools_lib.get_ifgdates(ifgdir)
    n_ifg = len(ifgdates)
    for ifgix, ifgd in enumerate(ifgdates):
        if np.mod(ifgix,100) == 0:
            print("  {0:3}/{1:3}th unw checked for dimension and readability".format(ifgix, n_ifg), flush=True)
        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        try:
            unw = io_lib.read_img(unwfile, length, width)
        except:
            print('probably dimension ERROR in '+ifgd+' unw file - source file inconsistent in LiCSAR frame. Deleting multilooked')
            #we can directly delete it here, as we work with ml* data
            shutil.rmtree(os.path.dirname(unwfile))

    #%% Read date and network information
    ### Get dates
    ifgdates = tools_lib.get_ifgdates(ifgdir)
    imdates = tools_lib.ifgdates2imdates(ifgdates)

    n_ifg = len(ifgdates)
    n_im = len(imdates)

    ### Copy dempar and mli[png|par]
    for file in ['slc.mli.par', 'EQA.dem_par']:
        if os.path.exists(os.path.join(ifgdir, file)):
            shutil.copy(os.path.join(ifgdir, file), infodir)

    for file in ['slc.mli', 'slc.mli.png', 'hgt', 'hgt.png', 'U']:
        if os.path.exists(os.path.join(ifgdir, file)):
            shutil.copy(os.path.join(ifgdir, file), resultsdir)


    #%% Read data
    ### Allocate memory
    n_unw = np.zeros((length, width), dtype=np.float32)
    coh_avg_ifg = []
    n_unw_ifg = []
    slope_ifg = []
    r_square_ifg = []

    ### Read data and calculate
    print('\nReading unw and cc data...', flush=True)
    ## First, identify valid area (n_unw>im)
    for ifgix, ifgd in enumerate(ifgdates):
        if np.mod(ifgix,100) == 0:
            print("  {0:3}/{1:3}th unw to identify valid area...".format(ifgix, n_ifg), flush=True)
        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        unw = io_lib.read_img(unwfile, length, width)

        unw[unw == 0] = np.nan # Fill 0 with nan
        n_unw += ~np.isnan(unw) # Summing number of unnan unw

    ## Identify valid area and calc rate_cov
    bool_valid = (n_unw>=n_im)
    n_unw_valid = bool_valid.sum()

    ## coregistration error shortlist
    coreg_error_ifg = []

    ## Read cc and unw data
    for ifgix, ifgd in enumerate(ifgdates):
        if np.mod(ifgix,100) == 0:
            print("  {0:3}/{1:3}th cc and unw...".format(ifgix, n_ifg), flush=True)
        ## unw
        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        unw = io_lib.read_img(unwfile, length, width)

        unw[unw == 0] = np.nan # Fill 0 with nan
        unw[~bool_valid] = np.nan # Fill sea area with nan
        n_unw_ifg.append((~np.isnan(unw)).sum())

        ## cc
        ccfile = os.path.join(ifgdir, ifgd, ifgd+'.cc')
        if os.path.getsize(ccfile) == length*width:
            coh = io_lib.read_img(ccfile, length, width, np.uint8)
            coh = coh.astype(np.float32)/255
            coh[coh==0] = np.nan
        else:
            coh = io_lib.read_img(ccfile, length, width)

        coh_avg_ifg.append(np.nanmean(coh[bool_valid])) # Use valid area only

        if check_coreg_slope:
            ## middle column slope
            middle_column = unw[:, width // 2]
            middle_column_latitudes = np.arange(length) * lattitude_resolution
            non_nan_mask = ~np.isnan(middle_column)
            if np.sum(non_nan_mask) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(middle_column_latitudes[non_nan_mask], middle_column[non_nan_mask])
                slope_ifg.append(abs(slope))
                r_square_ifg.append(r_value**2)
                if abs(slope) > 30 and r_value**2 > 0.95:
                    coreg_error_ifg.append(ifgd)
            else:
                slope_ifg.append(0)
                r_square_ifg.append(0)

    if check_coreg_slope:
        ## identify epochs with more than 1 coreg_errors captured by threshold:
        primarylist = []
        secondarylist = []
        for pairs in coreg_error_ifg:
            primarylist.append(pairs[:8])
            secondarylist.append(pairs[-8:])
        all_epochs = primarylist + secondarylist
        all_epochs.sort()
        coreg_error_epochs, counts = np.unique(all_epochs, return_counts=True)
        coreg_error_epochs = coreg_error_epochs[counts>1]

        ## grow the shortlist with repeated epochs in the shortlist
        ifg_containing_coreg_error_epochs = np.zeros(len(ifgdates))
        all_ifg_epoch1 = []
        all_ifg_epoch2 = []
        for pairs in ifgdates:
            all_ifg_epoch1.append(pairs[:8])
            all_ifg_epoch2.append(pairs[-8:])
        for epoch in coreg_error_epochs:
            ifg_containing_coreg_error_epochs += np.array(all_ifg_epoch1) == epoch
            ifg_containing_coreg_error_epochs += np.array(all_ifg_epoch2) == epoch
        ## threshold the expanded list with criteria slope_ifg => 20
        ifg_containing_coreg_error_epochs[np.array(slope_ifg) < 20] = 0
    else:
        ifg_containing_coreg_error_epochs = np.zeros(len(ifgdates)) # dummy

    ## convert unw pixels into percentage unw coverage
    rate_cov = np.array(n_unw_ifg)/n_unw_valid

    ## Read bperp data or dummy
    bperp_file = os.path.join(ifgdir, 'baselines')
    if os.path.exists(bperp_file):
        bperp = io_lib.read_bperp_file(bperp_file, imdates)
    else: #dummy
        bperp = np.random.random(n_im).tolist()


    #%% Identify bad ifgs, link ras and output stats information
    bad_ifgdates = []
    ixs_bad_ifgdates = []

    ### Header of stats file
    ifg_statsfile = os.path.join(infodir, '11ifg_stats.txt')
    fstats = open(ifg_statsfile, 'w')
    print('# Size: {0}({1}x{2}), n_valid: {3}'.format(width*length, width, length, n_unw_valid), file=fstats)
    if check_coreg_slope:
        print('# unw_cov_thre: {0}, coh_thre: {1}, |slope|:30 & r^2: 0.95 => repeated epochs => |slope|:20'.format(unw_cov_thre, coh_thre), file=fstats)
        print('# ifg dates         bperp   dt unw_cov  coh_av   |slope|   r^2', file=fstats)
    else:
        print('# unw_cov_thre: {0}, coh_thre: {1}'.format(unw_cov_thre, coh_thre), file=fstats)
        print('# ifg dates         bperp   dt unw_cov  coh_av', file=fstats)

    ### Identify suffix of raster image (png, ras or bmp?)
    unwfile = os.path.join(ifgdir, ifgdates[0], ifgdates[0]+'.unw')
    if os.path.exists(unwfile+'.ras'):
        suffix = '.ras'
    elif os.path.exists(unwfile+'.bmp'):
        suffix = '.bmp'
    elif os.path.exists(unwfile+'.png'):
        suffix = '.png'
    else:
        print('\nERROR: No browse image available for {}!\n'
              .format(unwfile), file=sys.stderr)
        return 2

    for i, ifgd in enumerate(ifgdates):
        rasname = ifgdates[i]+'.unw'+suffix
        rasorg = os.path.join(ifgdir, ifgdates[i], rasname)

        if not os.path.exists(rasorg):
            print('\nERROR: No browse image {} available!\n'
                  .format(rasorg), file=sys.stderr)
            return 2

        ### Identify bad ifgs and link ras
        if rate_cov[i] < unw_cov_thre or coh_avg_ifg[i] < coh_thre or \
           np.isnan(rate_cov[i]) or np.isnan(coh_avg_ifg[i]) or ifg_containing_coreg_error_epochs[i] > 0:
            bad_ifgdates.append(ifgdates[i])
            ixs_bad_ifgdates.append(i)
            rm_flag = '*'
            os.symlink(os.path.relpath(rasorg, bad_ifg_rasdir), os.path.join(bad_ifg_rasdir, rasname))
        else:
            os.symlink(os.path.relpath(rasorg, ifg_rasdir), os.path.join(ifg_rasdir, rasname))
            rm_flag = ''

        ### For stats file
        ix_primary = imdates.index(ifgd[:8])
        ix_secondary = imdates.index(ifgd[-8:])
        bperp_ifg = bperp[ix_secondary]-bperp[ix_primary]
        mday = dt.datetime.strptime(ifgd[:8], '%Y%m%d').toordinal()
        sday = dt.datetime.strptime(ifgd[-8:], '%Y%m%d').toordinal()
        dt_ifg = sday-mday
        if check_coreg_slope:
            print('{0}  {1:6.1f}  {2:3}   {3:5.3f}   {4:5.3f}    {5:5.3f}    {6:5.3f}  {7}'.format(ifgd, bperp_ifg, dt_ifg, rate_cov[i],  coh_avg_ifg[i], slope_ifg[i], r_square_ifg[i], rm_flag), file=fstats)
        else:
            print('{0}  {1:6.1f}  {2:3}   {3:5.3f}   {4:5.3f}    {5}'.format(ifgd, bperp_ifg, dt_ifg, rate_cov[i], coh_avg_ifg[i], rm_flag), file=fstats)

    fstats.close()

    ### Output list of bad ifg
    bad_ifgfile = os.path.join(infodir, '11bad_ifg.txt')
    print('\n{0}/{1} ifgs are discarded from further processing.'.format(len(bad_ifgdates), n_ifg))
    with open(bad_ifgfile, 'w') as f:
        if check_coreg_slope:
            print('ifg dates        unw_cov coh_av  |slope|   R^2')
            for i, ifgd in enumerate(bad_ifgdates):
                print('{}'.format(ifgd), file=f)
                print('{}  {:5.3f}  {:5.3f}   {:5.3f}   {:5.3f}'.format(ifgd, rate_cov[ixs_bad_ifgdates[i]],  coh_avg_ifg[ixs_bad_ifgdates[i]], slope_ifg[ixs_bad_ifgdates[i]], r_square_ifg[ixs_bad_ifgdates[i]]), flush=True)
        else:
            print('ifg dates        unw_cov coh_av')
            for i, ifgd in enumerate(bad_ifgdates):
                print('{}'.format(ifgd), file=f)
                print('{}  {:5.3f}  {:5.3f}'.format(ifgd, rate_cov[ixs_bad_ifgdates[i]],  coh_avg_ifg[ixs_bad_ifgdates[i]]), flush=True)

    ### Raise error if all ifgs are bad
    if len(bad_ifgdates) == n_ifg:
        raise ValueError('All ifgs are regarded as bad!\nChange the parameters or check the input ifgs.\n')


    #%% Identify removed image and output file
    good_ifgdates = list(set(ifgdates)-set(bad_ifgdates))
    good_ifgdates.sort()
    good_imdates = tools_lib.ifgdates2imdates(good_ifgdates)
    bad_imdates = list(set(imdates)-set(good_imdates))
    bad_imdates.sort()


    ### Output list of removed image
    bad_imfile = os.path.join(infodir, '11removed_image.txt')
    with open(bad_imfile, 'w') as f:
        for i in bad_imdates:
            print('{}'.format(i), file=f)


    #%% Plot network
    pngfile = os.path.join(netdir, 'network11_all.png')
    plot_lib.plot_network(ifgdates, bperp, [], pngfile)

    pngfile = os.path.join(netdir, 'network11.png')
    plot_lib.plot_network(ifgdates, bperp, bad_ifgdates, pngfile)

    pngfile = os.path.join(netdir, 'network11_nobad.png')
    plot_lib.plot_network(ifgdates, bperp, bad_ifgdates, pngfile, plot_bad=False)


    #%% Finish
    print('\nCheck network/*, 11bad_ifg_ras/* and 11ifg_ras/* in TS dir.')
    print('If you want to change the bad ifgs to be discarded, re-run with different thresholds or make a ifg list and indicate it by --rm_ifg_list option in the next step.')

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

