#!/usr/bin/env python3
"""
========
Overview
========
This script applies a tropospheric correction to unw data using GACOS data. GACOS data must be prepared beforehand by requesting on GACOS web. This step is optional.

=========
Changelog
=========
v1.0 20190729 Yu Morishita, Uni of Leeds and GSI
 - Original implementation

===============
Input & output files
===============
Inputs in GEOCml* :
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.unw
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.cc
 - U.geo
 - EQA.dem_par
 - slc.mli.par

Inputs in ztddir :
 - yyyymmdd.ztd[.rsc]

Outputs in GEOCml*GACOS directory
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.unw[.png] (Corrected)
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.gacos.png (Comparison)
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.cc (Symbolic link)
 - GACOS_info.txt (List of noise reduction rates)
 - no_gacos_ifg.txt (List of removed ifg because no GACOS data available)
 - sltd (Slantrange tropospheric delay in rad taking into account incidence angle)
 - other files needed for following time series analysis

=====
Usage
=====
LiCSBAS03op_GACOS.py -i in_dir -o out_dir -z ztddir

 -i  Path to the GEOCml* dir containing stack of unw data.
 -o  Path to the output dir.
 -z  Path to the dir containing ztd files.
 
"""


#%% Import
import getopt
import os
import sys
import time
import shutil
import glob
import numpy as np
import gdal
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
    in_dir = []
    out_dir = []
    ztddir = []
    resampleAlg = 'cubicspline'# None # 'cubic' 


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:z:", ["version", "help"])
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
            elif o == '-z':
                ztddir = a

        if not in_dir:
            raise Usage('No input directory given, -i is not optional!')
        elif not os.path.isdir(in_dir):
            raise Usage('No {} dir exists!'.format(in_dir))
        elif not os.path.exists(os.path.join(in_dir, 'slc.mli.par')):
            raise Usage('No slc.mli.par file exists in {}!'.format(in_dir))
        if not out_dir:
            raise Usage('No output directory given, -o is not optional!')
        elif not ztddir:
            raise Usage('No ztd directory given, -z is not optional!')
        elif not os.path.isdir(ztddir):
            raise Usage('No {} dir exists!'.format(ztddir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2
    
    
    #%% Read data information
    ### Directory
    in_dir = os.path.abspath(in_dir)
    ztddir = os.path.abspath(ztddir)

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

    ### Check coordinate
    if width_unw!=width_geo or length_unw!=length_geo:
        print('\n{} seems to contain files in radar coordinate!!\n'.format(in_dir), file=sys.stderr)
        print('Not supported.\n'.format(in_dir), file=sys.stderr)
        return 1

    ### Get ztd info. Grid registration
    ztdpar = glob.glob(os.path.join(ztddir, '*.ztd.rsc'))[0]
    width_ztd = int(io_lib.get_param_par(ztdpar, 'WIDTH'))
    length_ztd = int(io_lib.get_param_par(ztdpar, 'FILE_LENGTH'))
    dlat_ztd = float(io_lib.get_param_par(ztdpar, 'Y_STEP')) #minus
    dlon_ztd = float(io_lib.get_param_par(ztdpar, 'X_STEP'))
    latn_ztd = float(io_lib.get_param_par(ztdpar, 'Y_FIRST'))
    lonw_ztd = float(io_lib.get_param_par(ztdpar, 'X_FIRST'))

    ### Calc incidence angle from U.geo
    ufile = os.path.join(in_dir, 'U.geo')
    LOSu = io_lib.read_img(ufile, length_geo, width_geo)
    LOSu[LOSu==0] = np.nan

    ### Get ifgdates and imdates
    ifgdates = tools_lib.get_ifgdates(in_dir)
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    n_ifg = len(ifgdates)
    n_im = len(imdates)


    #%% Make hdr file of ztd
    hdrfile1 = os.path.join(sltddir, 'ztd.hdr')
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
    with open(hdrfile1, "w") as f:
        f.write("\n".join(strings))


    #%% Process ztd files 
    print('\nConvert ztd files to sltd files...', flush=True)
    ### First check if sltd already exist
    imdates2 = []
    for imd in imdates:
        sltd_geofile = os.path.join(sltddir, imd+'.sltd.geo')
        if not os.path.exists(sltd_geofile):
            imdates2.append(imd)

    n_im2 = len(imdates2)
    if n_im-n_im2 > 0:
        print("  {0:3}/{1:3} sltd already exist. Skip".format(n_im-n_im2, n_im), flush=True)
    
    for ix_im, imd in enumerate(imdates2):
        if np.mod(ix_im, 10)==0:
            print('  Finished {0:4}/{1:4}th ztd...'.format(ix_im, n_im2), flush=True)

        ztdfile = os.path.join(ztddir, imd+'.ztd')
        if not os.path.exists(ztdfile):
            print('  There is no {}!'.format(ztdfile), flush=True)
            continue ## Next imd
        
        hdrfile = os.path.join(sltddir, imd+'.hdr')
        bilfile = os.path.join(sltddir, imd+'.bil')
        if os.path.exists(hdrfile): os.remove(hdrfile)
        if os.path.exists(bilfile): os.remove(bilfile)
        os.symlink(os.path.relpath(hdrfile1, sltddir), hdrfile)
        os.symlink(os.path.relpath(ztdfile, sltddir), bilfile)

        ### Cut and resapmle ztd to geo
        ztd_geo = gdal.Warp("", bilfile, format='MEM', outputBounds=(lonw_geo, lats_geo, lone_geo, latn_geo), width=width_geo, height=length_geo, resampleAlg=resampleAlg, srcNodata=0).ReadAsArray()

        ### Skip if no data in the area
        if np.all(ztd_geo==0):
            print('  There is no valid data in {}!'.format(ztdfile), flush=True)
            continue ## Next imd

        ### Meter to rad, slantrange
        sltd_geo = ztd_geo*m2r_coef/LOSu ## LOSu=cos(inc)
        
        ### Output as sltd.geo
        sltd_geofile = os.path.join(sltddir, imd+'.sltd.geo')
        sltd_geo.tofile(sltd_geofile)

        os.remove(hdrfile)
        os.remove(bilfile)

    
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

    ### Correct
    for i, ifgd in enumerate(ifgdates2):
        if np.mod(i, 100)==0:
            print('  Finished {0:4}/{1:4}th unw...'.format(i, n_ifg2), flush=True)

        md = ifgd[:8]
        sd = ifgd[-8:]
        msltdfile = os.path.join(sltddir, md+'.sltd.geo')
        ssltdfile = os.path.join(sltddir, sd+'.sltd.geo')
        
        in_dir1 = os.path.join(in_dir, ifgd)
        out_dir1 = os.path.join(out_dir, ifgd)
        
        ### Check if sltd available for both master and slave. If not continue
        ## Not use in tsa because loop cannot be closed
        if not (os.path.exists(msltdfile) and os.path.exists(ssltdfile)):
            print('  ztd file not available for {}'.format(ifgd), flush=True)
            with open(no_gacos_ifgfile, mode='a') as fnogacos:
                print('{}'.format(ifgd), file=fnogacos)
            continue

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
        
        ### Output std
        std_unw = np.nanstd(unw)
        std_unwcor = np.nanstd(unw_cor)
        rate = (std_unw-std_unwcor)/std_unw*100
        with open(gacinfofile, "a") as f:
            print('{0}  {1:4.1f}  {2:4.1f} {3:5.1f}%'.format(ifgd, std_unw, std_unwcor, rate), file=f)

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

    print("", flush=True)
    
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


#%% main
if __name__ == "__main__":
    sys.exit(main())
