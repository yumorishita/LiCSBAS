#!/usr/bin/env python3
"""
v1.5.3 20201118 Yu Morishita, GSI

========
Overview
========
This script applies a tropospheric correction to unw data using GACOS data. GACOS data may be automatically downloaded from COMET-LiCS web at step01 (if available), or could be externally obtained by requesting on a GACOS web. 
If you request the GACOS data through the GACOS web, the dates and time of interest can be found in baselines and slc.mli.par, respectively. These are also available on the COMET-LiCS web portal. Once the GACOS data are ready, download the tar.gz, uncompress it, and put into GACOS dir. 
Existing files are not re-created to save time, i.e., only the newly available data will be processed. The impact of the correction can be visually checked by showing GACOS_info.png and */*.gacos.png. This step is optional.

===============
Input & output files
===============
Inputs in GEOCml*/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw
   - yyyymmdd_yyyymmdd.cc
 - U.geo
 - EQA.dem_par
 - slc.mli.par

Inputs in GACOS/ :
 - yyyymmdd.sltd.geo.tif  and/or 
 - yyyymmdd.ztd[.rsc]

Outputs in GEOCml*GACOS/
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw[.png] : Corrected unw
   - yyyymmdd_yyyymmdd.gacos.png : Comparison image
   - yyyymmdd_yyyymmdd.cc        : Coherence (symbolic link)
 - GACOS_info.txt : List of noise reduction rates
 - GACOS_info.png : Correlation diagram of STD between before and after
 - no_gacos_ifg.txt : List of removed ifg because no GACOS data available
 - no_gacos_im.txt  : List of images with no available GACOS data
 - sltd/
   - yyyymmdd.sltd.geo : Slantrange tropospheric delay in rad
 - other files needed for following time series analysis

=====
Usage
=====
LiCSBAS03op_GACOS.py -i in_dir -o out_dir [-g gacosdir] [--fillhole] [--n_para int]

 -i  Path to the GEOCml* dir containing stack of unw data
 -o  Path to the output dir
 -g  Path to the dir containing GACOS data (Default: GACOS)
 --fillhole  Fill holes of GACOS data at hgt=0 in SRTM3 by averaging surrounding pixels
 --n_para  Number of parallel processing (Default: # of usable CPU)

"""
#%% Change log
'''
v1.5.3 20201118 Yu Morishita, GSI
 - Again Bug fix of multiprocessing
v1.5.2 20201116 Yu Morishita, GSI
 - Bug fix of multiprocessing in Mac python>=3.8
v1.5.1 20201028 Yu Morishita, GSI
 - Update how to get n_para
v1.5 20200909 Yu Morishita, GSI
 - Parallel processing
v1.4 20200703 Yu Morioshita, GSI
 - Replace problematic terms
v1.3 20200609 Yu Morishita, GSI
 - Avoid reading error of ztd (unkown cause) by addding subtle value
v1.2 20200228 Yu Morishita, Uni of Leeds and GSI
 - Compatible with GACOS data provided from LiCSAR-portal
 - Output correlation plot
 - Output no_gacos_im.txt
 - Change option from -z (still available) to -g and set GACOS as default
v1.1 20190812 Yu Morishita, Uni of Leeds and GSI
 - Add fillhole option
v1.0 20190729 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''


#%% Import
import getopt
import os
import sys
import time
import shutil
import glob
import numpy as np
import gdal
import multiprocessing as multi
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg

#%% fill hole function
def fillhole(ztd):
    """
    Fill holes (no data) surrounded by valid data by averaging surrounding pixels.
    0 in ztd means no data.
    """
    length, width = ztd.shape
    
    ### Add 1 pixel margin to ztd data filled with 0
    ztd1 = np.zeros((length+2, width+2), dtype=np.float32)
    ztd1[1:length+1, 1:width+1] = ztd
    n_ztd1 = np.int16(ztd1!=0) # 1 if exist, 0 if no data

    ### Average 8 srrounding pixels. [1, 1] is center
    pixels = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
    _ztd = np.zeros_like(ztd)
    _n_ztd = np.zeros_like(ztd)

    for pixel in pixels:
        ### Adding data and number of data
        _ztd = _ztd + ztd1[pixel[0]:length+pixel[0],pixel[1]:width+pixel[1]]
        _n_ztd = _n_ztd + n_ztd1[pixel[0]:length+pixel[0],pixel[1]:width+pixel[1]]

    _n_ztd[_n_ztd==0] = 1 # avoid 0 division
    _ztd = _ztd/_n_ztd

    ### Fill hole 
    ztd[ztd==0] = _ztd[ztd==0]
    
    return ztd


#%% make_hdr
def make_hdr(ztdpar, hdrfile):
    ### Get ztd info. Grid registration
    width_ztd = int(io_lib.get_param_par(ztdpar, 'WIDTH'))
    length_ztd = int(io_lib.get_param_par(ztdpar, 'FILE_LENGTH'))
    dlat_ztd = float(io_lib.get_param_par(ztdpar, 'Y_STEP')) #minus
    dlon_ztd = float(io_lib.get_param_par(ztdpar, 'X_STEP'))
    latn_ztd = float(io_lib.get_param_par(ztdpar, 'Y_FIRST'))
    lonw_ztd = float(io_lib.get_param_par(ztdpar, 'X_FIRST'))

    ### Make hdr file of ztd
    strings = ["NROWS          {}".format(length_ztd),
               "NCOLS          {}".format(width_ztd),
               "NBITS          32",
               "PIXELTYPE      FLOAT",
               "BYTEORDER      I",
               "LAYOUT         BIL",
               "ULXMAP         {}".format(lonw_ztd),
               "ULYMAP         {}".format(latn_ztd),
               "XDIM           {}".format(dlon_ztd),
               "YDIM           {}".format(np.abs(dlat_ztd))]
    with open(hdrfile, "w") as f:
        f.write("\n".join(strings))


#%% Main
def main(argv=None):
    
    #%% Check argv
    if argv == None:
        argv = sys.argv
        
    start = time.time()
    ver="1.5.3"; date=20201118; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    ### For parallel processing
    global imdates2, gacosdir, outputBounds, width_geo, length_geo, resampleAlg,\
        sltddir, LOSu, m2r_coef, fillholeflag, ifgdates2,\
        in_dir, out_dir, length_unw, width_unw, cycle


    #%% Set default
    in_dir = []
    out_dir = []
    gacosdir = 'GACOS'
    resampleAlg = 'cubicspline'# None # 'cubic' 
    fillholeflag = False
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()

    q = multi.get_context('fork')


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:g:z:", ["fillhole", "help", "n_para="])
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
            elif o == '-z': ## for backward-compatible
                gacosdir = a
            elif o == '-g':
                gacosdir = a
            elif o == "--fillhole":
                fillholeflag = True
            elif o == '--n_para':
                n_para = int(a)

        if not in_dir:
            raise Usage('No input directory given, -i is not optional!')
        elif not os.path.isdir(in_dir):
            raise Usage('No {} dir exists!'.format(in_dir))
        elif not os.path.exists(os.path.join(in_dir, 'slc.mli.par')):
            raise Usage('No slc.mli.par file exists in {}!'.format(in_dir))
        if not out_dir:
            raise Usage('No output directory given, -o is not optional!')
        if not os.path.isdir(gacosdir):
            raise Usage('No {} dir exists!'.format(gacosdir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2
    
    
    #%% Read data information
    ### Directory
    in_dir = os.path.abspath(in_dir)
    gacosdir = os.path.abspath(gacosdir)

    out_dir = os.path.abspath(out_dir)
    if not os.path.exists(out_dir): os.mkdir(out_dir)

    sltddir = os.path.join(os.path.join(out_dir),'sltd')
    if not os.path.exists(sltddir): os.mkdir(sltddir)

    ### Get general info
    mlipar = os.path.join(in_dir, 'slc.mli.par')
    width_unw = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length_unw = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    speed_of_light = 299792458 #m/s
    radar_frequency = float(io_lib.get_param_par(mlipar, 'radar_frequency')) #Hz
    wavelength = speed_of_light/radar_frequency #meter
    m2r_coef = 4*np.pi/wavelength
    
    if wavelength > 0.2: ## L-band
        cycle = 1.5  # 2pi/cycle for png
    else: ## C-band
        cycle = 3  # 2pi*3/cycle for png

    ### Get geo info. Grid registration
    dempar = os.path.join(in_dir, 'EQA.dem_par')
    width_geo = int(io_lib.get_param_par(dempar, 'width'))
    length_geo = int(io_lib.get_param_par(dempar, 'nlines'))
    dlat_geo = float(io_lib.get_param_par(dempar, 'post_lat')) #minus
    dlon_geo = float(io_lib.get_param_par(dempar, 'post_lon'))
    latn_geo = float(io_lib.get_param_par(dempar, 'corner_lat'))
    lonw_geo = float(io_lib.get_param_par(dempar, 'corner_lon'))
    lats_geo = latn_geo+dlat_geo*(length_geo-1)
    lone_geo = lonw_geo+dlon_geo*(width_geo-1)
    outputBounds = (lonw_geo, lats_geo, lone_geo, latn_geo)
    
    ### Check coordinate
    if width_unw!=width_geo or length_unw!=length_geo:
        print('\n{} seems to contain files in radar coordinate!!\n'.format(in_dir), file=sys.stderr)
        print('Not supported.\n', file=sys.stderr)
        return 1

    ### Calc incidence angle from U.geo
    ufile = os.path.join(in_dir, 'U.geo')
    LOSu = io_lib.read_img(ufile, length_geo, width_geo)
    LOSu[LOSu==0] = np.nan

    ### Get ifgdates and imdates
    ifgdates = tools_lib.get_ifgdates(in_dir)
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    n_ifg = len(ifgdates)
    n_im = len(imdates)


    #%% Process ztd files 
    print('\nConvert ztd/sltd.geo.tif files to sltd.geo files...', flush=True)

    no_gacos_imfile = os.path.join(out_dir, 'no_gacos_im.txt')
    if os.path.exists(no_gacos_imfile): os.remove(no_gacos_imfile)

    ### First check if sltd already exist
    imdates2 = []
    for imd in imdates:
        sltd_geofile = os.path.join(sltddir, imd+'.sltd.geo')
        if not os.path.exists(sltd_geofile):
            imdates2.append(imd)

    n_im2 = len(imdates2)
    if n_im-n_im2 > 0:
        print("  {0:3}/{1:3} sltd already exist. Skip".format(n_im-n_im2, n_im), flush=True)

    if n_im2 > 0:
        ### Convert with parallel processing
        if n_para > n_im2:
            _n_para = n_im2
        else:
            _n_para = n_para
            
        print('  {} parallel processing...'.format(_n_para), flush=True)
        p = q.Pool(_n_para)
        no_gacos_imds = p.map(convert_wrapper, range(n_im2))
        p.close()
    
        for imd in no_gacos_imds:
            if imd is not None:
                with open(no_gacos_imfile, mode='a') as fnogacos:
                    print('{}'.format(imd), file=fnogacos)
    
    
    #%% Correct unw files
    print('\nCorrect unw data...', flush=True)
    ### Information files    
    gacinfofile = os.path.join(out_dir, 'GACOS_info.txt')
    if not os.path.exists(gacinfofile):
        ### Add header
        with open(gacinfofile, "w") as f:
            print(' Phase STD (rad) Before After  ReductionRate', file=f)
    
    no_gacos_ifgfile = os.path.join(out_dir, 'no_gacos_ifg.txt')
    if os.path.exists(no_gacos_ifgfile): os.remove(no_gacos_ifgfile)

    ### First check if already corrected unw exist
    ifgdates2 = []
    for i, ifgd in enumerate(ifgdates): 
        out_dir1 = os.path.join(out_dir, ifgd)
        unw_corfile = os.path.join(out_dir1, ifgd+'.unw')
        if not os.path.exists(unw_corfile):
            ifgdates2.append(ifgd)

    n_ifg2 = len(ifgdates2)
    if n_ifg-n_ifg2 > 0:
        print("  {0:3}/{1:3} corrected unw already exist. Skip".format(n_ifg-n_ifg2, n_ifg), flush=True)

    if n_ifg2 > 0:
        ### Correct with parallel processing
        if n_para > n_ifg2:
            _n_para = n_ifg2
        else:
            _n_para = n_para
            
        print('  {} parallel processing...'.format(_n_para), flush=True)
        p = q.Pool(_n_para)
        _return = p.map(correct_wrapper, range(n_ifg2))
        p.close()
    
        for i in range(n_ifg2):
            if _return[i][0] == 1:
                with open(no_gacos_ifgfile, mode='a') as fnogacos:
                    print('{}'.format(_return[i][1]), file=fnogacos)
            elif _return[i][0] == 2:
                with open(gacinfofile, "a") as f:
                    print('{0}  {1:4.1f}  {2:4.1f} {3:5.1f}%'.format(*_return[i][1]), file=f)
    
    print("", flush=True)
    
    
    #%% Create correlation png
    pngfile = os.path.join(out_dir, 'GACOS_info.png')
    plot_lib.plot_gacos_info(gacinfofile, pngfile)
    
    
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

    if os.path.exists(no_gacos_ifgfile):
        print('Caution: Some ifgs below are excluded due to GACOS unavailable')
        with open(no_gacos_ifgfile) as f:
            for line in f:
                print(line, end='')
        print('')

    if os.path.exists(no_gacos_imfile):
        print('GACOS data for the following dates are missing:')
        with open(no_gacos_imfile) as f:
            for line in f:
                print(line, end='')
        print('')

#%%
def convert_wrapper(ix_im):
    imd = imdates2[ix_im]
    if np.mod(ix_im, 10)==0:
        print('  Finished {0:4}/{1:4}th sltd...'.format(ix_im, len(imdates2)), flush=True)

    ztdfile = os.path.join(gacosdir, imd+'.ztd')
    sltdtiffile = os.path.join(gacosdir, imd+'.sltd.geo.tif')

    if os.path.exists(sltdtiffile):
        infile = os.path.basename(sltdtiffile)
        try: ### Cut and resapmle. Already in rad.
            sltd_geo = gdal.Warp("", sltdtiffile, format='MEM', outputBounds=outputBounds, width=width_geo, height=length_geo, resampleAlg=resampleAlg, srcNodata=0).ReadAsArray()
        except: ## if broken
            print ('  {} cannot open. Skip'.format(infile), flush=True)
            return imd

    elif os.path.exists(ztdfile):
        infile = os.path.basename(ztdfile)
        hdrfile = os.path.join(sltddir, imd+'.hdr')
        bilfile = os.path.join(sltddir, imd+'.bil')
        if os.path.exists(hdrfile): os.remove(hdrfile)
        if os.path.exists(bilfile): os.remove(bilfile)
        make_hdr(ztdfile+'.rsc', hdrfile)
        os.symlink(os.path.relpath(ztdfile, sltddir), bilfile)
        
        ## Check read error with unkown cause
        if gdal.Info(bilfile) is None: 
            ### Create new ztd by adding 0.0001m
            print('{} cannot open, but trying minor update. You can ignore this error unless this script stops.'.format(ztdfile))
            shutil.copy2(ztdfile, ztdfile+'.org') ## Backup
            _ztd = np.fromfile(ztdfile, dtype=np.float32)
            _ztd[_ztd!=0] = _ztd[_ztd!=0]+0.001
            _ztd.tofile(ztdfile)

        ### Cut and resapmle ztd to geo
        ztd_geo = gdal.Warp("", bilfile, format='MEM', outputBounds=outputBounds,\
            width=width_geo, height=length_geo, \
            resampleAlg=resampleAlg, srcNodata=0).ReadAsArray()
        os.remove(hdrfile)
        os.remove(bilfile)

        ### Meter to rad, slantrange
        sltd_geo = ztd_geo/LOSu*m2r_coef ## LOSu=cos(inc)

    else:
        print('  There is no ztd|sltd.geo.tif for {}!'.format(imd), flush=True)
        return imd ## Next imd

    ### Skip if no data in the area
    if np.all((sltd_geo==0)|np.isnan(sltd_geo)):
        print('  There is no valid data in {}!'.format(infile), flush=True)
        return imd ## Next imd

    ### Fill hole is specified
    if fillholeflag:
        sltd_geo = fillhole(sltd_geo)
    
    ### Output as sltd.geo
    sltd_geofile = os.path.join(sltddir, imd+'.sltd.geo')
    sltd_geo.tofile(sltd_geofile)

    return


#%%
def correct_wrapper(i):
    ifgd = ifgdates2[i]
    if np.mod(i, 10)==0:
        print('  Finished {0:4}/{1:4}th unw...'.format(i, len(ifgdates2)), flush=True)

    md = ifgd[:8]
    sd = ifgd[-8:]
    msltdfile = os.path.join(sltddir, md+'.sltd.geo')
    ssltdfile = os.path.join(sltddir, sd+'.sltd.geo')
    
    in_dir1 = os.path.join(in_dir, ifgd)
    out_dir1 = os.path.join(out_dir, ifgd)
    
    ### Check if sltd available for both primary and secondary. If not continue
    ## Not use in tsa because loop cannot be closed
    if not (os.path.exists(msltdfile) and os.path.exists(ssltdfile)):
        print('  ztd file not available for {}'.format(ifgd), flush=True)
        return 1, ifgd

    ### Prepare directory and file
    if not os.path.exists(out_dir1): os.mkdir(out_dir1)
    unwfile = os.path.join(in_dir1, ifgd+'.unw')
    unw_corfile = os.path.join(out_dir1, ifgd+'.unw')
    
    ### Calculate dsltd
    msltd = io_lib.read_img(msltdfile, length_unw, width_unw)
    ssltd = io_lib.read_img(ssltdfile, length_unw, width_unw)

    msltd[msltd==0] = np.nan
    ssltd[ssltd==0] = np.nan
    
    dsltd = ssltd-msltd
    
    ### Correct unw
    unw = io_lib.read_img(unwfile, length_unw, width_unw)
    
    unw[unw==0] = np.nan
    unw_cor = unw-dsltd
    unw_cor.tofile(unw_corfile)
    
    ### Calc std
    std_unw = np.nanstd(unw)
    std_unwcor = np.nanstd(unw_cor)
    rate = (std_unw-std_unwcor)/std_unw*100

    ### Link cc
    if not os.path.exists(os.path.join(out_dir1, ifgd+'.cc')):
        os.symlink(os.path.relpath(os.path.join(in_dir1, ifgd+'.cc'), out_dir1), os.path.join(out_dir1, ifgd+'.cc'))
        
    ### Output png for comparison
    data3 = [np.angle(np.exp(1j*(data/cycle))*cycle) for data in [unw, unw_cor, dsltd]]
    title3 = ['unw_org (STD: {:.1f} rad)'.format(std_unw), 'unw_cor (STD: {:.1f} rad)'.format(std_unwcor), 'dsltd ({:.1f}% reduced)'.format(rate)]
    pngfile = os.path.join(out_dir1, ifgd+'.gacos.png')
    plot_lib.make_3im_png(data3, pngfile, 'insar', title3, vmin=-np.pi, vmax=np.pi, cbar=False)
    
    ## Output png for corrected unw
    pngfile = os.path.join(out_dir1, ifgd+'.unw.png')
    title = '{} ({}pi/cycle)'.format(ifgd, cycle*2)
    plot_lib.make_im_png(np.angle(np.exp(1j*unw_cor/cycle)*cycle), pngfile, 'insar', title, -np.pi, np.pi, cbar=False)

    return 2, [ifgd, std_unw, std_unwcor, rate]


#%% main
if __name__ == "__main__":
    sys.exit(main())
