#!/usr/bin/env python3
"""
v1.2 20190807 Yu Morishita, Uni of Leeds and GSI

========
Overview
========
This script calculates velocity and its standard deviation from cum*.h5 and outputs them as a float32 file. Amplitude and time offset of the annual displacement can also be calculated by --sin option.

=====
Usage
=====
LiCSBAS_cum2vel.py [-s yyyymmdd] [-e yyyymmdd] [-i infile] [-o outfile] [-r x1:x2/y1:y2] [--vstd] [--sin] [--mask maskfile] [--png] 

 -s  Start date of period to calculate velocity (Default: first date)
 -e  End date of period to calculate velocity (Default: last date)
 -i  Path to input cum file (Default: cum_filt.h5)
 -o  Output vel file (Default: yyyymmdd_yyyymmdd.vel[.mskd])
 -r  Reference area (Default: same as info/*ref.txt)
     Note: x1/y1 range 0 to width-1, while x2/y2 range 1 to width
     0 for x2/y2 means all. (i.e., 0:0/0:0 means whole area).
 --vstd  Calculate vstd (Default: No)
 --sin   Add sin (annual) funcsion to linear model (Default: No)
         *.amp and *.dt (time difference wrt Jan 1) are output
 --mask  Path to mask file for ref phase calculation (Default: No mask)
 --png   Make png file (Default: Not make png)

"""
#%% Change log
'''
v1.2 20190807 Yu Morishita, Uni of Leeds and GSI
 - Add sin option
v1.1 20190802 Yu Morishita, Uni of Leeds and GSI
 - Make vstd optional
v1.0 20190730 Yu Morishita, Uni of Leeds and GSI
 - Original implementationf
'''

#%% Import
import getopt
import os
import sys
import re
import time
import numpy as np
import datetime as dt
import h5py as h5
import LiCSBAS_io_lib as io_lib
import LiCSBAS_inv_lib as inv_lib
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
    ver=1.2; date=20190807; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    #%% Set default
    imd_s = []
    imd_e = []
    cumfile = 'cum_filt.h5'
    outfile = []
    refarea = []
    maskfile = []
    vstdflag = False
    sinflag = False
    pngflag = False


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hs:e:i:o:r:", ["help", "vstd", "sin", "png", "mask="])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-s':
                imd_s = a
            elif o == '-e':
                imd_e = a
            elif o == '-i':
                cumfile = a
            elif o == '-o':
                outfile = a
            elif o == '-r':
                refarea = a
            elif o == '--vstd':
                vstdflag = True
            elif o == '--sin':
                sinflag = True
            elif o == '--mask':
                maskfile = a
            elif o == '--png':
                pngflag = True

        if not os.path.exists(cumfile):
            raise Usage('No {} exists! Use -i option.'.format(cumfile))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Read info
    ### Read cumfile
    cumh5 = h5.File(cumfile,'r')
    imdates = cumh5['imdates'][()].astype(str).tolist()
    cum = cumh5['cum']
    n_im_all, length, width = cum.shape

    if not refarea:
        refarea = cumh5['refarea'][()]
        refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]
    else:
        if not tools_lib.read_range(refarea, width, length):
            print('\nERROR in {}\n'.format(refarea), file=sys.stderr)
            return 2
        else:
            refx1, refx2, refy1, refy2 = tools_lib.read_range(refarea, width, length)
    
    #%% Setting
    ### Dates
    if not imd_s:
        imd_s = imdates[0]
        
    if not imd_e:
        imd_e = imdates[-1]
        
    ### mask
    if maskfile:
        mask = io_lib.read_img(maskfile, length, width)
        mask[mask==0] = np.nan
        suffix_mask = '.mskd'
    else:
        mask = np.ones((length, width), dtype=np.float32)
        suffix_mask = ''
        
    ### Find date index if not exist in imdates
    if not imd_s in imdates:
        for imd in imdates:
            if int(imd) >= int(imd_s): ## First larger one than imd_s
                imd_s = imd
                break
        
    if not imd_e in imdates:
        for imd in imdates[::-1]:
            if int(imd) <= int(imd_e): ## Last smaller one than imd_e
                imd_e = imd
                break
        
    ix_s = imdates.index(imd_s)
    ix_e = imdates.index(imd_e)+1 #+1 for python custom
    n_im = ix_e-ix_s

    ### Calc dt in year
    imdates_dt = ([dt.datetime.strptime(imd, '%Y%m%d').toordinal() for imd in imdates[ix_s:ix_e]])
    dt_cum = np.float32((np.array(imdates_dt)-imdates_dt[0])/365.25)
    
    ### Outfile
    if not outfile:
        outfile = '{}_{}.vel{}'.format(imd_s, imd_e, suffix_mask)
        

    #%% Display info
    print('')
    print('Start date  : {}'.format(imdates[ix_s]))
    print('End date    : {}'.format(imdates[ix_e-1]))
    print('# of images : {}'.format(n_im))
    print('Ref area    : {}:{}/{}:{}'.format(refx1, refx2, refy1, refy2))
    print('')


    #%% Calc velocity and vstd
    vconst = np.zeros((length, width), dtype=np.float32)*np.nan
    vel = np.zeros((length, width), dtype=np.float32)*np.nan

    ### Read cum data
    cum_tmp = cum[ix_s:ix_e, :, :]*mask
    cum_ref = np.nanmean(cum[ix_s:ix_e, refy1:refy2, refx1:refx2]*mask[refy1:refy2, refx1:refx2], axis=(1, 2))
    
    if np.all(np.isnan(cum_ref)):
        print('\nERROR: Ref area has only NaN value!\n', file=sys.stderr)
        return 2
    
    cum_tmp = cum_tmp-cum_ref[:, np.newaxis, np.newaxis]

    ### Extract not nan points
    bool_allnan = np.all(np.isnan(cum_tmp), axis=0)
    cum_tmp = cum_tmp.reshape(n_im, length*width)[:, ~bool_allnan.ravel()].transpose()
        
    
    if not sinflag: ## Linear function
        print('Calc velocity...')
        vel[~bool_allnan], vconst[~bool_allnan] = inv_lib.calc_vel(cum_tmp, dt_cum)
        vel.tofile(outfile)
    else: ## Linear+sin function
        print('Calc velocity and annual components...')
        amp = np.zeros((length, width), dtype=np.float32)*np.nan
        delta_t = np.zeros((length, width), dtype=np.float32)*np.nan
        ampfile = outfile.replace('vel', 'amp')
        dtfile = outfile.replace('vel', 'dt')
        
        vel[~bool_allnan], vconst[~bool_allnan], amp[~bool_allnan], delta_t[~bool_allnan] = inv_lib.calc_velsin(cum_tmp, dt_cum, imdates[0])
        vel.tofile(outfile)
        amp.tofile(ampfile)
        delta_t.tofile(dtfile)
    
    ### vstd
    if vstdflag:
        vstdfile = outfile.replace('vel', 'vstd')
        vstd = np.zeros((length, width), dtype=np.float32)*np.nan

        print('Calc vstd...')
        vstd[~bool_allnan] = inv_lib.calc_velstd_withnan(cum_tmp, dt_cum)
        vstd.tofile(vstdfile)


    #%% Make png if specified
    if pngflag:
        pngfile = outfile+'.png'
        title = 'n_im: {}, Ref X/Y {}:{}/{}:{}'.format(n_im, refx1, refx2, refy1, refy2)
        plot_lib.make_im_png(vel, pngfile, 'jet', title)

        if sinflag:
            amp_max = np.nanpercentile(amp, 99)
            plot_lib.make_im_png(amp, ampfile+'.png', 'viridis', title, vmax=amp_max)
            plot_lib.make_im_png(delta_t, dtfile+'.png', 'hsv', title)

        if vstdflag:
            plot_lib.make_im_png(vstd, vstdfile+'.png', 'jet', title)
    

    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output: {}'.format(outfile), flush=True)
    if vstdflag:
        print('       {}'.format(vstdfile), flush=True)
    print('')


#%% main
if __name__ == "__main__":
    sys.exit(main())
