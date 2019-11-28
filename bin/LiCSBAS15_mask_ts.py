#!/usr/bin/env python3
"""
========
Overview
========
This script makes a mask for time series using several noise indices.

=========
Changelog
=========
v1.3 20191128 Yu Morishita, Uni of Leeds and GSI
 - Add noautoadjust option
v1.2 20190918 Yu Morishita, Uni of Leeds and GSI
 - Output mask_ts_mskd.png
v1.1 20190906 Yu Morishita, Uni of Leeds and GSI
 - tight_layout and auto ajust of size for png
v1.0 20190724 Yu Morishita, Uni of Leeds and GSI
 - Original implementation

===============
Input & output files
===============
Inputs in TS_GEOCml* :
 - results/
   - vel
   - coh_avg
   - n_unw
   - vstd
   - maxTlen
   - n_gap
   - stc
   - n_ifg_noloop
   - n_loop_err
   - resid_rms
 - info/parameters.txt
 
Outputs in TS_GEOCml* directory
 - mask_ts[_mskd].png
 - results/vel.mskd[.png]
 - results/mask[.png]

=====
Usage
=====
LiCSBAS15_mask_ts.py -t tsadir [-c coh_thre] [-u n_unw_r_thre] [-v vstd_thre] [-T maxTlen_thre] [-g n_gap_thre] [-s stc_thre] [-i n_ifg_noloop_thre] [-l n_loop_err_thre] [-r resid_rms_thre] [--vmin vmin] [--vmax vmin] [--keep_isolated] [--noautoadjust]

 -t  Path to the TS_GEOCml* dir.
 -c  Threshold of coh_avg (average coherence)
 -u  Threshold of n_unw (number of used unwrap data)
     (Note this value is ratio to the number of images; i.e., 1.5*n_im)
 -v  Threshold of vstd (std of the velocity (mm/yr))
 -T  Threshold of maxTlen (max time length of connected network (year))
 -g  Threshold of n_gap (number of gaps in network)
 -s  Threshold of stc (spatio-temporal consistency (mm))
 -i  Threshold of n_ifg_noloop (number of ifgs with no loop)
 -l  Threshold of n_loop_err (number of loop_err)
 -r  Threshold of resid_rms (RMS of residuals in inversion (mm))
 --v[min|max]  Min|Max value for output figure of velocity (Default: auto)
 --keep_isolated  Keep (not mask) isolated pixels
                  (Default: they are masked using STC)
 --noautoadjust  Do not auto adjust threshold when all pixels are masked
                 (Default: do auto adjust)
 
 Default thresholds for L-band:
   C-band : -c 0.05 -u 1.5 -v 10 -T 1 -g 10 -s 5  -i 10 -l 5 -r 2
   L-band : -c 0.01 -u 1   -v 20 -T 1 -g 10 -s 10 -i 10 -l 1 -r 10
 
"""


#%% Import
import getopt
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import sys
import time
import numpy as np
import LiCSBAS_io_lib as io_lib
import LiCSBAS_plot_lib as plot_lib

import warnings
import matplotlib
with warnings.catch_warnings(): ## To silence user warning
    warnings.simplefilter('ignore', UserWarning)
    matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


#%%
def add_subplot(fig, i, data, vmin, vmax, cmap, title):
    ax = fig.add_subplot(3, 4, i+1) #index start from 1
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(im)
    ax.set_title('{0}'.format(title))
    ax.set_xticklabels([])
    ax.set_yticklabels([])


#%% Main
def main(argv=None):
   
    #%% Check argv
    if argv == None:
        argv = sys.argv
        
    start = time.time()
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    tsadir = []
    thre_dict = {}
    vmin = []
    vmax = []
    keep_isolated = False
    auto_adjust = True


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "ht:c:u:v:g:i:l:r:T:s:", ["version", "help", "vmin=", "vmax=", "keep_isolated", "noautoadjust"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-t':
                tsadir = a
            elif o == '-c':
                thre_dict['coh_avg'] = float(a)
            elif o == '-u':
                thre_dict['n_unw_r'] = float(a)
            elif o == '-v':
                thre_dict['vstd'] = float(a)
            elif o == '-T':
                thre_dict['maxTlen'] = float(a)
            elif o == '-g':
                thre_dict['n_gap'] = int(a)
            elif o == '-s':
                thre_dict['stc'] = float(a)
            elif o == '-i':
                thre_dict['n_ifg_noloop'] = int(a)
            elif o == '-l':
                thre_dict['n_loop_err'] = int(a)
            elif o == '-r':
                thre_dict['resid_rms'] = float(a)
            elif o == '--vmin':
                vmin = float(a)
            elif o == '--vmax':
                vmax = float(a)
            elif o == '--keep_isolated':
                keep_isolated = True
            elif o == '--noautoadjust':
                auto_adjust = False

        if not tsadir:
            raise Usage('No tsa directory given, -t is not optional!')
        elif not os.path.isdir(tsadir):
            raise Usage('No {} dir exists!'.format(tsadir))
        elif not os.path.isdir(os.path.join(tsadir, 'results')):
            raise Usage('No results dir exists in {}!'.format(tsadir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2
 

    #%% Directory and file setting and get info
    tsadir = os.path.abspath(tsadir)
    resultsdir = os.path.join(tsadir,'results')

    parmfile = os.path.join(tsadir, 'info', 'parameters.txt')
    maskts_png = os.path.join(tsadir,'mask_ts.png')
    maskts2_png = os.path.join(tsadir,'mask_ts_masked.png')

    names = ['coh_avg', 'n_unw', 'vstd', 'maxTlen', 'n_gap', 'stc', 'n_ifg_noloop', 'n_loop_err', 'resid_rms'] ## noise indices
    gt_lt = ['lt', 'lt', 'gt', 'lt', 'gt', 'gt', 'gt', 'gt', 'gt'] ## > or <
    ## gt: greater values than thre are masked 
    ## lt: more little values than thre are masked (coh_avg, n_unw, maxTlen)

    units = ['', '', 'mm/yr', 'yr', '', 'mm', '', '', 'mm']


    ### Get size and ref
    width = int(io_lib.get_param_par(parmfile, 'range_samples'))
    length = int(io_lib.get_param_par(parmfile, 'azimuth_lines'))
    wavelength = float(io_lib.get_param_par(parmfile, 'wavelength'))

    n_im = int(io_lib.get_param_par(parmfile, 'n_im'))

    
    #%% Determine default thretholds depending on frequency band
    if not 'maxTlen' in thre_dict: thre_dict['maxTlen'] = 1
    if not 'n_gap' in thre_dict: thre_dict['n_gap'] = 10
    if not 'n_ifg_noloop' in thre_dict: thre_dict['n_ifg_noloop'] = 10

    if wavelength > 0.2: ## L-band
        if not 'coh_avg' in thre_dict: thre_dict['coh_avg'] = 0.01
        if not 'n_unw_r' in thre_dict: thre_dict['n_unw_r'] = 1.0
        if not 'vstd' in thre_dict: thre_dict['vstd'] = 20
        if not 'stc' in thre_dict: thre_dict['stc'] = 10
        if not 'n_loop_err' in thre_dict: thre_dict['n_loop_err'] = 1
        if not 'resid_rms' in thre_dict: thre_dict['resid_rms'] = 10
    if wavelength < 0.2: ## C-band
        if not 'coh_avg' in thre_dict: thre_dict['coh_avg'] = 0.05
        if not 'n_unw_r' in thre_dict: thre_dict['n_unw_r'] = 1.5
        if not 'vstd' in thre_dict: thre_dict['vstd'] = 10
        if not 'stc' in thre_dict: thre_dict['stc'] = 5
        if not 'n_loop_err' in thre_dict: thre_dict['n_loop_err'] = 5
        if not 'resid_rms' in thre_dict: thre_dict['resid_rms'] = 2
    
    thre_dict['n_unw'] = int(n_im*thre_dict['n_unw_r'])

    
    #%% Read data
    velfile = os.path.join(resultsdir,'vel')
    vel = io_lib.read_img(velfile, length, width)
    bool_nan = np.isnan(vel)
    bool_nan[vel==0] = True ## Ref point. Unmask later
    n_pt_all = (~bool_nan).sum() ## Number of unnan points

    data_dict = {}
    for name in names:
        file = os.path.join(resultsdir, name)
        data_dict[name] = io_lib.read_img(file, length, width)

    ## stc is always nan at isolted pixels.
    if keep_isolated:
        ## Give 0 to keep isolated pixels
        data_dict['stc'][np.isnan(data_dict['stc'])] = 0
    else:
        ## Give stc_thre to remove isolated pixels
        data_dict['stc'][np.isnan(data_dict['stc'])] = thre_dict['stc']+1
        

    #%% Make mask
    ### Evaluate only valid pixels in vel
    mask_pt = np.ones_like(vel)[~bool_nan]
    mskd_rate = []
    
    for i, name in enumerate(names):
        _data = data_dict[name][~bool_nan]
        _thre = thre_dict[name]

        if gt_lt[i] == 'lt': ## coh_avg, n_unw, maxTlen
            ## Multiply -1 to treat as if gt
            _data = -1*_data
            _thre = -1*_thre

        ### First check if the thre masks not all pixels
        ### If all pixels are masked, change thre to the max/min value
        if auto_adjust:
            minvalue = np.nanmin(_data)
            if minvalue > _thre:
                print('\nAll pixels would be masked with {} thre of {}'.format(name, thre_dict[name]), flush=True)
                thre_dict[name] = np.ceil(minvalue)
                _thre = thre_dict[name]
                if gt_lt[i] == 'lt':
                    thre_dict[name] = -1*thre_dict[name]
                print('Automatically change the thre to {} (ceil of min value)'.format(thre_dict[name]), flush=True)

        ### Make mask for this index
        with warnings.catch_warnings(): ## To silence RuntimeWarning of nan<thre
            warnings.simplefilter('ignore', RuntimeWarning)
            _mask_pt = (_data <= _thre) # nan returns false
        mskd_rate.append((1-_mask_pt.sum()/n_pt_all)*100)
        mask_pt = mask_pt*_mask_pt
    
    ### Make total mask
    mask = np.ones_like(vel)*np.nan
    mask[~bool_nan] = mask_pt  #1:valid, 0:masked, nan:originally nan
    mask[vel==0] = 1 ## Retrieve ref point

    ### Apply mask
    vel_mskd = vel*mask
    vel_mskd[mask==0] = np.nan
        
    ### Count total mask
    n_nomask = int(np.nansum(mask))
    rate_nomask = n_nomask/n_pt_all*100


    #%% Stdout info
    print('')
    print('Noise index    : Threshold  (rate to be masked)')
    for i, name in enumerate(names):
        print('- {:12s} : {:4} {:5} ({:4.1f}%)'.format(name, thre_dict[name], units[i], mskd_rate[i]))
    print('')
    print('Masked pixels  : {}/{} ({:.1f}%)'.format(n_pt_all-n_nomask, n_pt_all, 100-rate_nomask))
    print('Kept pixels    : {}/{} ({:.1f}%)\n'.format(n_nomask, n_pt_all, rate_nomask), flush=True)

    if n_nomask == 1:
        print('All pixels are masked!!', file=sys.stderr)
        print('Try again with different threshold.\n', file=sys.stderr)
        return 1


    #%% Prepare for png
    ## Set color range for vel
    if not vmin: ## auto
        vmin = np.nanpercentile(vel_mskd, 1)
        if np.isnan(vmin): ## In case no data in vel_mskd
            vmin = np.nanpercentile(vel, 1)
    if not vmax: ## auto
        vmax = np.nanpercentile(vel_mskd, 99)
        if np.isnan(vmax): ## In case no data in vel_mskd 
            vmax = np.nanpercentile(vel, 1)
        

    #%% Output thumbnail png
    if length > width:
        figsize_y = 9
        figsize_x = int(figsize_y*4/3*width/length+2)
    else:
        figsize_x = 12
        figsize_y = int((figsize_x)/4*3*length/width)
        if figsize_y < 4: figsize_y = 4
    
    fig = plt.figure(figsize = (figsize_x, figsize_y))
    fig2 = plt.figure(figsize = (figsize_x, figsize_y))

    ##First 3; vel.mskd, vel, mask
    data = [vel_mskd, vel, mask]
    titles = ['vel.mskd', 'vel', 'mask']
    vmins = [vmin, vmin, 0]
    vmaxs = [vmax, vmax, 1]
    cmaps = ['jet', 'jet', 'viridis']
    for i in range(3): 
        add_subplot(fig, i, data[i], vmins[i], vmaxs[i], cmaps[i], titles[i])
        i2 = 0 if i==1 else 1 if i==0 else 2 # inv vel and vel.mskd
        add_subplot(fig2, i2, data[i], vmins[i], vmaxs[i], cmaps[i], titles[i])


    ## Next 9 noise indices
    mask_nan = mask.copy()
    mask_nan[mask==0] = np.nan
    for i, name in enumerate(names):
        data = data_dict[name]
        ## Mask nan in vel for each indeces except coh_avg and n_unw
        if not name == 'coh_avg' and not name == 'n_unw':
            data[bool_nan] = np.nan
        
        if gt_lt[i] == 'lt': ## coh_avg, n_unw, maxTlen
            cmap = 'viridis'
            vmin_n = thre_dict[name]*0.8
            vmax_n = np.nanmax(data)
        else:
            cmap = 'viridis_r'
            vmin_n = 0
            vmax_n = thre_dict[name]*1.2

        title = '{} {}({})'.format(name, units[i], thre_dict[name])
        add_subplot(fig, i+3, data, vmin_n, vmax_n, cmap, title)
        add_subplot(fig2, i+3, data*mask_nan, vmin_n, vmax_n, cmap, title)
        #i+3 because 3 data already plotted
              

    fig.tight_layout()
    fig.savefig(maskts_png)
    fig2.tight_layout()
    fig2.savefig(maskts2_png)
    
    plt.close(fig=fig)

    
    #%% Output vel.mskd and mask
    velmskdfile = os.path.join(resultsdir,'vel.mskd')
    vel_mskd.tofile(velmskdfile)

    pngfile = velmskdfile+'.png'
    cmap = 'jet'
    title = 'Masked velocity (mm/yr)'
    plot_lib.make_im_png(vel_mskd, pngfile, cmap, title, vmin, vmax)


    maskfile = os.path.join(resultsdir,'mask')
    mask.tofile(maskfile)

    pngfile = maskfile+'.png'
    cmap = 'viridis'
    title = 'Mask'
    plot_lib.make_im_png(mask, pngfile, cmap, title, 0, 1)

    
    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output png: {}\n'.format(os.path.relpath(maskts_png)), flush=True)


#%% main
if __name__ == "__main__":
    sys.exit(main())
