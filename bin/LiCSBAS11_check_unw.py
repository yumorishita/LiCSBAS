#!/usr/bin/env python3
"""
========
Overview
========
This script checks quality of unw data and identifies bad interferograms based on average coherence and coverage of the unw data. This also prepares a time series working directory.

=========
Changelog
=========
v1.1 20191115 Yu Morishita, Uni of Leeds and GSI
 - Add hgt
v1.0 20190729 Yu Morishita, Uni of Leeds and GSI
 - Original implementation

===============
Input & output files
===============
Inputs in GEOCml* :
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.unw[.png]
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.cc
 - slc.mli[.par|.png] (single master only, to get parameters of width etc.)
 - baselines (can be dummy)
 - EQA.dem_par

 Outputs in TS_GEOCml* directory
 - info/
   - 11bad_ifg.txt : List of bad ifgs discarded from further processing
   - 11ifg_stats.txt  : Statistics of interferograms
   - EQA.dem_par (copy)
   - slc.mli.par (copy)
 - results/slc.mli[.png] (copy, if exist)
 - results/hgt[.png] (copy, if exist)
 - 11bad_ifg_ras/yyyymmdd_yyyymmdd.unw[.bmp|.png] : Ras images of bad ifgs
 - 11ifg_ras/yyyymmdd_yyyymmdd.unw[.bmp|.png] : Ras images other than bad ifgs
 - network/network11*.png    : Figures of baseline configuration

=====
Usage
=====
LiCSBAS11_check_unw.py -d ifgdir [-t tsadir] [-c coh_thre] [-u unw_thre]

 -d  Path to the GEOCml* dir containing stack of unw data.
 -t  Path to the output TS_[IFG|GEOC]ml?? dir. (Default: TS_GEOCml*)
 -c  IFGs with smaller average coherence than this value are listed as bad data.
     (Default: 0.1)
 -u  IFGs with smaller coverage than this value are listed as bad data.
     (Default: 0.5)

"""


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
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    ifgdir = []
    tsadir = []
    coh_thre = 0.1
    unw_cov_thre = 0.5


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hd:t:c:u:", ["help"])
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


    #%% Read date, network information and size
    ### Get dates
    ifgdates = tools_lib.get_ifgdates(ifgdir)
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    
    n_ifg = len(ifgdates)
    n_im = len(imdates)

    ### Get size
    mlipar = os.path.join(ifgdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    print("\nSize         : {} x {}".format(width, length), flush=True)

    ### Copy dempar and mli[png|par]
    for file in ['slc.mli.par', 'EQA.dem_par']:
        if os.path.exists(os.path.join(ifgdir, file)):
            shutil.copy(os.path.join(ifgdir, file), infodir)

    for file in ['slc.mli', 'slc.mli.png', 'hgt', 'hgt.png']:
        if os.path.exists(os.path.join(ifgdir, file)):
            shutil.copy(os.path.join(ifgdir, file), resultsdir)


    #%% Read data 
    ### Allocate memory
    n_unw = np.zeros((length, width), dtype=np.float32)
    coh_avg_ifg = []
    n_unw_ifg = []

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
        coh = io_lib.read_img(ccfile, length, width)

        coh_avg_ifg.append(np.nanmean(coh[bool_valid])) # Use valid area only

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
    print('# unw_cov_thre: {0}, coh_thre: {1}'.format(unw_cov_thre, coh_thre), file=fstats)
    print('# ifg dates         bperp   dt unw_cov  coh_av', file=fstats)

    ### Identify suffix of raster image (ras or bmp?)
    unwfile = os.path.join(ifgdir, ifgdates[0], ifgdates[0]+'.unw')
    if os.path.exists(unwfile+'.ras'):
        suffix = '.ras'
    elif os.path.exists(unwfile+'.bmp'):
        suffix = '.bmp'
    elif os.path.exists(unwfile+'.png'):
        suffix = '.png'

    for i, ifgd in enumerate(ifgdates):
        rasname = ifgdates[i]+'.unw'+suffix
        rasorg = os.path.join(ifgdir, ifgdates[i], rasname)

        ### Identify bad ifgs and link ras
        if rate_cov[i] < unw_cov_thre or coh_avg_ifg[i] < coh_thre:
            bad_ifgdates.append(ifgdates[i])
            ixs_bad_ifgdates.append(i)
            rm_flag = '*'
            os.symlink(os.path.relpath(rasorg, bad_ifg_rasdir), os.path.join(bad_ifg_rasdir, rasname))
        else:
            os.symlink(os.path.relpath(rasorg, ifg_rasdir), os.path.join(ifg_rasdir, rasname))
            rm_flag = ''

        ### For stats file
        ix_master = imdates.index(ifgd[:8])
        ix_slave = imdates.index(ifgd[-8:])
        bperp_ifg = bperp[ix_slave]-bperp[ix_master]
        mday = dt.datetime.strptime(ifgd[:8], '%Y%m%d').toordinal()
        sday = dt.datetime.strptime(ifgd[-8:], '%Y%m%d').toordinal()
        dt_ifg = sday-mday

        print('{0}  {1:6.1f}  {2:3}   {3:5.3f}   {4:5.3f} {5}'.format(ifgd, bperp_ifg, dt_ifg, rate_cov[i],  coh_avg_ifg[i], rm_flag), file=fstats)

    fstats.close()

    ### Output list of bad ifg            
    print('\n{0}/{1} ifgs are discarded from further processing.'.format(len(bad_ifgdates), n_ifg))
    print('ifg dates        unw_cov coh_av')
    bad_ifgfile = os.path.join(infodir, '11bad_ifg.txt')
    with open(bad_ifgfile, 'w') as f:
        for i, ifgd in enumerate(bad_ifgdates):
            print('{}'.format(ifgd), file=f)
            print('{}  {:5.3f}  {:5.3f}'.format(ifgd, rate_cov[ixs_bad_ifgdates[i]],  coh_avg_ifg[ixs_bad_ifgdates[i]]), flush=True)


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
    plot_lib.plot_network(ifgdates, bperp, [], pngfile, pdf=True)

    pngfile = os.path.join(netdir, 'network11.png')
    plot_lib.plot_network(ifgdates, bperp, bad_ifgdates, pngfile, pdf=True)

    pngfile = os.path.join(netdir, 'network11_nobad.png')
    plot_lib.plot_network(ifgdates, bperp, bad_ifgdates, pngfile, plot_bad=False, pdf=True)


    #%% Finish
    print('\nCheck network/*, 11bad_ifg_ras/* and 11ifg_ras/* in TS dir.')
    print('If you want to change the bad ifgs to be discarded, re-run with different thresholds or edit bad_ifg11.txt before next step.')

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

