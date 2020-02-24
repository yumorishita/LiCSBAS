#!/usr/bin/env python3
"""
v1.1 20190829 Yu Morishita, Uni of Leeds and GSI

========
Overview
========
This script applies spatial-temporal filter (HP in time and LP in space with gaussian kernel, same as StaMPS) to the time series of displacement. Deramp can also be applied if specified by -r option.

===============
Input & output files
===============
Inputs in TS_GEOCml* :
 - cum.h5
 - results/mask
 - info/parameters.txt
 
Outputs in TS_GEOCml* directory
 - cum_filt.h5
 - 16filt
   - yyyymmdd_filt.png
   - yyyymmdd_yyyymmdd_filt.png
  [- yyyymmdd_deramp.png]
  [- yyyymmdd_yyyymmdd_deramp.png]
 - results/vel.filt[.mskd][.png]
 - results/vintercept.filt.mskd[.png]

=====
Usage
=====
LiCSBAS16_filt_ts.py -t tsadir [-s filtwidth_km] [-y filtwidth_yr] [-r deg] [--nomask]

 -t  Path to the TS_GEOCml* dir.
 -s  Width of spatial filter in km (Default: 2 km)
 -y  Width of temporal filter in yr (Default: auto, avg_interval*3)
 -r  Degree of deramp [1, bl, 2] (Default: no deramp)
     1: 1d ramp, bl: bilinear, 2: 2d polynomial
 --nomask  Not mask (Default: use mask)
  
"""
#%% Change log
'''
v1.1 20190829 Yu Morishita, Uni of Leeds and GSI
 - Remove cum_filt.h5 if exists before creation
v1.0 20190731 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''

#%% Import
import getopt
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import sys
import time
import glob
import warnings
import numpy as np
import datetime as dt
import h5py as h5
from astropy.convolution import Gaussian2DKernel, convolve_fft
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_inv_lib as inv_lib
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
    ver=1.1; date=20190829; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    tsadir = []
    filtwidth_km = 2
    filtwidth_yr = []
    deg_ramp = []
    maskflag = True
    
    cumname = 'cum.h5'


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "ht:s:y:r:", ["help", "nomask"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-t':
                tsadir = a
            elif o == '-s':
                filtwidth_km = float(a)
            elif o == '-y':
                filtwidth_yr = float(a)
            elif o == '-r':
                deg_ramp = a
            elif o == '--nomask':
                maskflag = False

        if not tsadir:
            raise Usage('No tsa directory given, -t is not optional!')
        elif not os.path.isdir(tsadir):
            raise Usage('No {} dir exists!'.format(tsadir))
        elif not os.path.exists(os.path.join(tsadir, cumname)):
            raise Usage('No {} exists in {}!'.format(cumname, tsadir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Directory and file setting
    tsadir = os.path.abspath(tsadir)
    cumfile = os.path.join(tsadir, cumname)
    resultsdir = os.path.join(tsadir, 'results')
    parmfile = os.path.join(tsadir, 'info',  'parameters.txt')

    pixsp_r = float(io_lib.get_param_par(parmfile, 'pixel_spacing_r'))
    pixsp_a = float(io_lib.get_param_par(parmfile, 'pixel_spacing_a'))
    x_stddev = filtwidth_km*1000/pixsp_r
    y_stddev = filtwidth_km*1000/pixsp_a

    wavelength = float(io_lib.get_param_par(parmfile, 'wavelength')) #meter
    coef_r2m = -wavelength/4/np.pi*1000 #rad -> mm, positive is -LOS

    if wavelength > 0.2: ## L-band
        cycle = 1.5 # 2pi/cycle for comparison png
    elif wavelength <= 0.2: ## C-band
        cycle = 3 # 3*2pi/cycle for comparison png

    filtdir = os.path.join(tsadir, '16filt')
    if not os.path.exists(filtdir): os.mkdir(filtdir)
    
    cumffile = os.path.join(tsadir, 'cum_filt.h5')

    vconstfile = os.path.join(resultsdir, 'vintercept.filt')
    velfile = os.path.join(resultsdir, 'vel.filt')

    cumh5 = h5.File(cumfile,'r')

    if os.path.exists(cumffile): os.remove(cumffile)
    cumfh5 = h5.File(cumffile,'w')


    #%% Dates
    imdates = cumh5['imdates'][()].astype(str).tolist()
    cum_org = cumh5['cum']
    n_im, length, width = cum_org.shape

    ### Calc dt in year
    imdates_dt = ([dt.datetime.strptime(imd, '%Y%m%d').toordinal() for imd in imdates])
    dt_cum = np.float32((np.array(imdates_dt)-imdates_dt[0])/365.25)

    ### Save dates and refarea into cumf
    cumfh5.create_dataset('imdates', data=cumh5['imdates'])
    cumfh5.create_dataset('refarea', data=cumh5['refarea'])
    cumfh5.create_dataset('gap', data=cumh5['gap'])
    try: ## if dummy, no bperp field
        cumfh5.create_dataset('bperp', data=cumh5['bperp'])
    except:
        print('No bperp field found in {}. Skip.'.format(cumname))

    try: 
        cumfh5.create_dataset('corner_lat', data=cumh5['corner_lat'])
        cumfh5.create_dataset('corner_lon', data=cumh5['corner_lon'])
        cumfh5.create_dataset('post_lat', data=cumh5['post_lat'])
        cumfh5.create_dataset('post_lon', data=cumh5['post_lon'])
    except: ## not geocoded
        print('No latlon field found in {}. Skip.'.format(cumname))

    ### temporal filter width
    if not filtwidth_yr and filtwidth_yr != 0:
        filtwidth_yr = dt_cum[-1]/(n_im-1)*3 ## avg interval*3


    #%% Display settings
    print('')
    print('Size of image (w,l)      : {0}, {1}'.format(width, length))
    print('Number of images         : {}'.format(n_im))
    print('Width of filter in space : {} km ({:.1f}x{:.1f} pixel)'.format(filtwidth_km, x_stddev, y_stddev))
    print('Width of filter in time  : {:.3f} yr ({} days)'.format(filtwidth_yr, int(filtwidth_yr*365.25)))
    print('Deramp flag              : {}'.format(deg_ramp), flush=True)


    #%% Load Mask (1: unmask, 0: mask, nan: no cum data)
    if maskflag:
        maskfile = os.path.join(resultsdir, 'mask')
        mask = io_lib.read_img(maskfile, length, width)
        mask[mask==0] = np.nan ## 0->nan
    else:
        mask = np.ones((length, width), dtype=np.float32)
        mask[np.isnan(cum_org[0, :, :])] = np.nan


    #%% First, deramp if indicated
    cum = np.zeros((cum_org.shape), dtype=np.float32)*np.nan
    if not deg_ramp:
        cum = cum_org[()]
        ## Remove past *_deramp.png if exist
        for png in glob.glob(os.path.join(filtdir, '*_deramp.png')):
            os.remove(png)
        
    else:
        print('\nDeramp ifgs with the degree of {}...'.format(deg_ramp), flush=True)
        ramp1 = []

        for i in range(n_im):
            if np.mod(i, 10) == 0:
                print("  {0:3}/{1:3}th image...".format(i, n_im), flush=True)
            
            ramp, _ = tools_lib.fit2d(cum_org[i, :, :], deg=deg_ramp)
            cum[i, :, :] = cum_org[i, :, :]-ramp

            ## Output comparison image of deramp
            data3 = [np.angle(np.exp(1j*(data/coef_r2m/cycle))*cycle) for data in [cum_org[i, :, :]*mask, ramp, cum[i, :, :]*mask]]
            title3 = ['Before deramp ({}pi/cycle)'.format(cycle*2), 'ramp phase (deg:{}, {}pi/cycle)'.format(deg_ramp, cycle*2), 'After deramp ({}pi/cycle)'.format(cycle*2)]
            pngfile = os.path.join(filtdir, imdates[i]+'_deramp.png')
            plot_lib.make_3im_png(data3, pngfile, 'insar', title3, vmin=-np.pi, vmax=np.pi, cbar=False)

            ## Output comparison image of deramp for increment
            if i != 0:
                data3 = [np.angle(np.exp(1j*(data/coef_r2m/cycle))*cycle) for data in [(cum_org[i, :, :]-cum_org[i-1, :, :])*mask, ramp-ramp1, (cum[i, :, :]-cum[i-1, :, :])*mask]]
                title3 = ['Before deramp ({}pi/cycle)'.format(cycle*2), 'ramp phase (deg:{}, {}pi/cycle)'.format(deg_ramp, cycle*2), 'After deramp ({}pi/cycle)'.format(cycle*2)]
                pngfile = os.path.join(filtdir, '{}_{}_deramp.png'.format(imdates[i-1], imdates[i]))
                plot_lib.make_3im_png(data3, pngfile, 'insar', title3, vmin=-np.pi, vmax=np.pi, cbar=False)

            ramp1 = ramp.copy() ## backup last ramp        


    #%% Filter each image
    cum_filt = cumfh5.require_dataset('cum', (n_im, length, width), dtype=np.float32)
    cum_hptlps1 = []

    print('\nHP filter in time, LP filter in space...', flush=True)
    for i in range(n_im):
        if np.mod(i, 10) == 0:
            print("  {0:3}/{1:3}th image...".format(i, n_im), flush=True)


        #%% Second, HP in time
        if filtwidth_yr == 0.0:
            cum_hpt = cum[i, :, :] ## No temporal filter
        else:
            time_diff_sq = (dt_cum[i]-dt_cum)**2
            
            ## Limit reading data within filtwidth_yr**8
            ixs = time_diff_sq < filtwidth_yr*8

            weight_factor = np.tile(np.exp(-time_diff_sq[ixs]/2/filtwidth_yr**2)[:, np.newaxis, np.newaxis], (1, length, width)) #len(ixs), length, width
            
            ## Take into account nan in cum
            weight_factor = weight_factor*(~np.isnan(cum[ixs, :, :]))
            
            ## Normalize weight
            with warnings.catch_warnings(): ## To silence warning by zero division
                warnings.simplefilter('ignore', RuntimeWarning)
                weight_factor = weight_factor/np.sum(weight_factor, axis=0)
            
            cum_lpt = np.nansum(cum[ixs, :, :]*weight_factor, axis=0);
    
            cum_hpt = cum[i, :, :] - cum_lpt


        #%% Third, LP in space and subtract from original
        if filtwidth_km == 0.0:
            cum_filt[i, :, :] = cum[i, :, :] ## No spatial
        else:
            with warnings.catch_warnings(): ## To silence warning
                if i ==0: cum_hpt = cum_hpt+sys.float_info.epsilon ##To distinguish from 0 of filtered nodata

#                warnings.simplefilter('ignore', FutureWarning)
                warnings.simplefilter('ignore', RuntimeWarning)
                kernel = Gaussian2DKernel(x_stddev, y_stddev)
                cum_hptlps = convolve_fft(cum_hpt*mask, kernel, fill_value=np.nan, allow_huge=True) ## fill edge 0 for interpolation
                cum_hptlps[cum_hptlps == 0] = np.nan ## fill 0 with nan
            
            cum_filt[i, :, :] = cum[i, :, :] - cum_hptlps


        #%% Output comparison image
        data3 = [np.angle(np.exp(1j*(data/coef_r2m/cycle))*cycle) for data in [cum[i, :, :]*mask, cum_hptlps, cum_filt[i, :, :]*mask]]
        title3 = ['Before filter ({}pi/cycle)'.format(cycle*2), 'Filter phase ({}pi/cycle)'.format(cycle*2), 'After filter ({}pi/cycle)'.format(cycle*2)]
        pngfile = os.path.join(filtdir, imdates[i]+'_filt.png')
        plot_lib.make_3im_png(data3, pngfile, 'insar', title3, vmin=-np.pi, vmax=np.pi, cbar=False)
        
        ### Output comparison image for increment
        if i != 0:
            data3 = [np.angle(np.exp(1j*(data/coef_r2m/cycle))*cycle) for data in [(cum[i, :, :]-cum[i-1, :, :])*mask, cum_hptlps-cum_hptlps1, (cum_filt[i, :, :]-cum_filt[i-1, :, :])*mask]]
            title3 = ['Before filter ({}pi/cycle)'.format(cycle*2), 'Filter phase ({}pi/cycle)'.format(cycle*2), 'After filter ({}pi/cycle)'.format(cycle*2)]
            pngfile = os.path.join(filtdir, imdates[i]+'_filt.png')
            pngfile = os.path.join(filtdir, '{}_{}_filt.png'.format(imdates[i-1], imdates[i]))
            plot_lib.make_3im_png(data3, pngfile, 'insar', title3, vmin=-np.pi, vmax=np.pi, cbar=False)

        cum_hptlps1 = cum_hptlps.copy() ## backup last filt phase


    #%% Calc filtered velocity
    print('\nCalculate velocity of filtered time series...', flush=True)
    G = np.stack((np.ones_like(dt_cum), dt_cum), axis=1)
    vconst = np.zeros((length, width), dtype=np.float32)*np.nan
    vel = np.zeros((length, width), dtype=np.float32)*np.nan
    
    bool_unnan = ~np.isnan(cum_filt[0, :, :]).reshape(length, width) ## not all nan
    cum_pt = cum_filt[()].reshape(n_im, length*width)[:, bool_unnan.ravel()] #n_im x n_pt
    n_pt_unnan = bool_unnan.sum()
    vconst_tmp = np.zeros((n_pt_unnan), dtype=np.float32)*np.nan
    vel_tmp = np.zeros((n_pt_unnan), dtype=np.float32)*np.nan
    
    bool_nonan_pt = np.all(~np.isnan(cum_pt), axis=0) 

    ### First, calc vel point without nan
    print('  First, solving {0:6}/{1:6}th points with full cum...'.format(bool_nonan_pt.sum(), n_pt_unnan), flush=True)
    vconst_tmp[bool_nonan_pt], vel_tmp[bool_nonan_pt] = np.linalg.lstsq(G, cum_pt[:, bool_nonan_pt], rcond=None)[0]

    ### Next, calc vel point with nan
    print('  Next, solving {0:6}/{1:6}th points with nan in cum...'.format((~bool_nonan_pt).sum(), n_pt_unnan), flush=True)
    
    mask_cum = ~np.isnan(cum_pt[:, ~bool_nonan_pt])
    vconst_tmp[~bool_nonan_pt], vel_tmp[~bool_nonan_pt] = inv_lib.censored_lstsq_slow(G, cum_pt[:, ~bool_nonan_pt], mask_cum)
    vconst[bool_unnan], vel[bool_unnan] = vconst_tmp, vel_tmp

    vconst.tofile(vconstfile)
    vel.tofile(velfile)

    if maskflag:
        vel_mskd = vel*mask
        vconst_mskd = vconst*mask
        vconst_mskd.tofile(vconstfile+'.mskd')
        vel_mskd.tofile(velfile+'.mskd')

    cumfh5.create_dataset('vel', data=vel.reshape(length, width))
    cumfh5.create_dataset('vintercept', data=vconst.reshape(length, width))


    #%% Add info and close
    cumfh5.create_dataset('filtwidth_yr', data=filtwidth_yr)
    cumfh5.create_dataset('filtwidth_km', data=filtwidth_km)
    cumfh5.create_dataset('deramp_flag', data=deg_ramp)
    
    cumh5.close()
    cumfh5.close()

 
    #%% Output image
    cmap = 'jet'
    pngfile = os.path.join(resultsdir,'vel.filt.png')
    title = 'Filtered velocity (mm/yr)'
    vmin = np.nanpercentile(vel, 1)
    vmax = np.nanpercentile(vel, 99)
    plot_lib.make_im_png(vel, pngfile, cmap, title, vmin, vmax)

    ## vintercept
    pngfile = os.path.join(resultsdir,'vintercept.filt.png')
    title = 'Intercept of filtered velocity (mm)'
    vmin = np.nanpercentile(vconst, 1)
    vmax = np.nanpercentile(vconst, 99)
    plot_lib.make_im_png(vconst, pngfile, cmap, title, vmin, vmax)

    if maskflag:
        pngfile = os.path.join(resultsdir,'vel.filt.mskd.png')
        title = 'Masked filtered velocity (mm/yr)'
        vmin = np.nanpercentile(vel_mskd, 1)
        vmax = np.nanpercentile(vel_mskd, 99)
        plot_lib.make_im_png(vel_mskd, pngfile, cmap, title, vmin, vmax)
    
        ## vintercept
        pngfile = os.path.join(resultsdir,'vintercept.filt.mskd.png')
        title = 'Masked intercept of filtered velocity (mm)'
        vmin = np.nanpercentile(vconst_mskd, 1)
        vmax = np.nanpercentile(vconst_mskd, 99)
        plot_lib.make_im_png(vconst_mskd, pngfile, cmap, title, vmin, vmax)


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output: {}\n'.format(os.path.relpath(cumffile)), flush=True)

    print('To plot the time-series:')
    print('LiCSBAS_plot_ts.py -i {} &\n'.format(os.path.relpath(cumffile)))


#%% main
if __name__ == "__main__":
    sys.exit(main())
