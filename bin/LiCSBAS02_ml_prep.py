#!/usr/bin/env python3
"""
v1.7.2 202011116 Yu Morishita, GSI

========
Overview
========
This script converts GeoTIFF files of unw and cc to float32 and uint8 format, respectively, for further time series analysis, and also downsamples (multilooks) data if specified. Existing files are not re-created to save time, i.e., only the newly available data will be processed.

====================
Input & output files
====================
Inputs:
 - GEOC/    
   - yyyymmdd_yyyymmdd/
     - yyyymmdd_yyyymmdd.geo.unw.tif
     - yyyymmdd_yyyymmdd.geo.cc.tif
  [- *.geo.mli.tif]
  [- *.geo.hgt.tif]
  [- *.geo.[E|N|U].tif]
  [- baselines]
  [- metadata.txt]

Outputs in GEOCml*/ (downsampled if indicated):
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw[.png] (float32)
   - yyyymmdd_yyyymmdd.cc (uint8)
 - baselines (may be dummy)
 - EQA.dem_par
 - slc.mli.par
 - slc.mli[.png] (if input exists)
 - hgt[.png] (if input exists)
 - [E|N|U].geo (if input exists)
 - no_unw_list.txt (if there are unavailable unw|cc)

=====
Usage
=====
LiCSBAS02_ml_prep.py -i GEOCdir [-o GEOCmldir] [-n nlook] [--freq float] [--n_para int]

 -i  Path to the input GEOC dir containing stack of geotiff data
 -o  Path to the output GEOCml dir (Default: GEOCml[nlook])
 -n  Number of donwsampling factor (Default: 1, no donwsampling)
 --freq    Radar frequency in Hz (Default: 5.405e9 for Sentinel-1)
           (e.g., 1.27e9 for ALOS, 1.2575e9 for ALOS-2/U, 1.2365e9 for ALOS-2/{F,W})
 --n_para  Number of parallel processing (Default: # of usable CPU)

"""
#%% Change log
'''
v1.7.2 20201116 Yu Morishita, GSI
 - Bug fix of multiprocessing in Mac python>=3.8
v1.7.1 20201028 Yu Morishita, GSI
 - Update how to get n_para
v1.7 20201020 Yu Morishita, GSI
 - Remove -f option and not download tifs here
v1.6.1 20201016 Yu Morishita, GSI
 - Deal with mli and hgt in other dtype
v1.6 20201008 Yu Morishita, GSI
 - Add --freq option
v1.5.1 20200916 Yu Morishita, GSI
 - Bug fix in handling cc float
v1.5 20200909 Yu Morishita, GSI
 - Parallel processing
v1.4 20200228 Yu Morishita, Uni of Leeds and GSI
 - Change format of output cc from float32 to uint8
 - Add center_time into slc.mli.par
v1.3 20191115 Yu Morishita, Uni of Leeds and GSI
 - Use mli and hgt
v1.2 20191014 Yu Morishita, Uni of Leeds and GSI
 - Deal with format of uint8 of cc.tif
 - Not available mli
v1.1 20190824 Yu Morishita, Uni of Leeds and GSI
 - Skip broken geotiff
v1.0 20190731 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''


#%% Import
import getopt
import os
import re
import sys
import time
import shutil
import gdal
import glob
import numpy as np
import subprocess as subp
import multiprocessing as multi
multi.set_start_method('fork') # for python >=3.8 in Mac
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
    ver="1.7.2"; date=202011116; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    ### For parallel processing
    global ifgdates2, geocdir, outdir, nlook, n_valid_thre, cycle, cmap


    #%% Set default
    geocdir = []
    outdir = []
    nlook = 1
    radar_freq = 5.405e9
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()

    cmap = 'insar'
    cycle = 3
    n_valid_thre = 0.5


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:n:", ["help", "freq=", "n_para="])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                geocdir = a
            elif o == '-o':
                outdir = a
            elif o == '-n':
                nlook = int(a)
            elif o == '--freq':
                radar_freq = float(a)
            elif o == '--n_para':
                n_para = int(a)

        if not geocdir:
            raise Usage('No GEOC directory given, -d is not optional!')
        elif not os.path.isdir(geocdir):
            raise Usage('No {} dir exists!'.format(geocdir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2
 

    #%% Directory and file setting
    geocdir = os.path.abspath(geocdir)
    if not outdir:
        outdir = os.path.join(os.path.dirname(geocdir), 'GEOCml{}'.format(nlook))
    if not os.path.exists(outdir): os.mkdir(outdir)

    mlipar = os.path.join(outdir, 'slc.mli.par')
    dempar = os.path.join(outdir, 'EQA.dem_par')

    no_unw_list = os.path.join(outdir, 'no_unw_list.txt')
    if os.path.exists(no_unw_list): os.remove(no_unw_list)

    bperp_file_in = os.path.join(geocdir, 'baselines')
    bperp_file_out = os.path.join(outdir, 'baselines')

    metadata_file = os.path.join(geocdir, 'metadata.txt')
    if os.path.exists(metadata_file):
        center_time = subp.check_output(['grep', 'center_time', metadata_file]).decode().split('=')[1].strip()
    else:
        center_time = None


    #%% ENU
    for ENU in ['E', 'N', 'U']:
        print('\nCreate {}'.format(ENU+'.geo'), flush=True)
        enutif = glob.glob(os.path.join(geocdir, '*.geo.{}.tif'.format(ENU)))

        ### Download if not exist
        if len(enutif)==0:
            print('  No *.geo.{}.tif found in {}'.format(ENU, os.path.basename(geocdir)), flush=True)
            continue

        else:
            enutif = enutif[0] ## first one
                
        ### Create float
        data = gdal.Open(enutif).ReadAsArray()
        data[data==0] = np.nan

        if nlook != 1:
            ### Multilook
            data = tools_lib.multilook(data, nlook, nlook)

        outfile = os.path.join(outdir, ENU+'.geo')
        data.tofile(outfile)
        print('  {}.geo created'.format(ENU), flush=True)


    #%% mli
    print('\nCreate slc.mli', flush=True)
    mlitif = glob.glob(os.path.join(geocdir, '*.geo.mli.tif'))
    if len(mlitif)>0:
        mlitif = mlitif[0] ## First one
        mli = np.float32(gdal.Open(mlitif).ReadAsArray())
        mli[mli==0] = np.nan
    
        if nlook != 1:
            ### Multilook
            mli = tools_lib.multilook(mli, nlook, nlook)
    
        mlifile = os.path.join(outdir, 'slc.mli')
        mli.tofile(mlifile)
        mlipngfile = mlifile+'.png'
        mli = np.log10(mli)
        vmin = np.nanpercentile(mli, 5)
        vmax = np.nanpercentile(mli, 95)
        plot_lib.make_im_png(mli, mlipngfile, 'gray', 'MLI (log10)', vmin, vmax, cbar=True)
        print('  slc.mli[.png] created', flush=True)
    else:
        print('  No *.geo.mli.tif found in {}'.format(os.path.basename(geocdir)), flush=True)


    #%% hgt
    print('\nCreate hgt', flush=True)
    hgttif = glob.glob(os.path.join(geocdir, '*.geo.hgt.tif'))
    if len(hgttif)>0:
        hgttif = hgttif[0] ## First one
        hgt = np.float32(gdal.Open(hgttif).ReadAsArray())
        hgt[hgt==0] = np.nan

        if nlook != 1:
            ### Multilook
            hgt = tools_lib.multilook(hgt, nlook, nlook)
    
        hgtfile = os.path.join(outdir, 'hgt')
        hgt.tofile(hgtfile)
        hgtpngfile = hgtfile+'.png'
        vmax = np.nanpercentile(hgt, 99)
        vmin = -vmax/3 ## bnecause 1/4 of terrain is blue
        plot_lib.make_im_png(hgt, hgtpngfile, 'terrain', 'DEM (m)', vmin, vmax, cbar=True)
        print('  hgt[.png] created', flush=True)
    else:
        print('  No *.geo.hgt.tif found in {}'.format(os.path.basename(geocdir)), flush=True)


    #%% tif -> float (with multilook/downsampling)
    print('\nCreate unw and cc', flush=True)
    ifgdates = tools_lib.get_ifgdates(geocdir)
    n_ifg = len(ifgdates)
    
    ### First check if float already exist
    ifgdates2 = []
    for i, ifgd in enumerate(ifgdates): 
        ifgdir1 = os.path.join(outdir, ifgd)
        unwfile = os.path.join(ifgdir1, ifgd+'.unw')
        ccfile = os.path.join(ifgdir1, ifgd+'.cc')
        if not (os.path.exists(unwfile) and os.path.exists(ccfile)):
            ifgdates2.append(ifgd)

    n_ifg2 = len(ifgdates2)
    if n_ifg-n_ifg2 > 0:
        print("  {0:3}/{1:3} unw and cc already exist. Skip".format(n_ifg-n_ifg2, n_ifg), flush=True)

    if n_ifg2 > 0:
        if n_para > n_ifg2:
            n_para = n_ifg2
            
        ### Create float with parallel processing
        print('  {} parallel processing...'.format(n_para), flush=True)
        p = multi.Pool(n_para)
        rc = p.map(convert_wrapper, range(n_ifg2))
        p.close()
        
        ifgd_ok = []
        for i, _rc in enumerate(rc):
            if _rc == 1:
                with open(no_unw_list, 'a') as f:
                    print('{}'.format(ifgdates2[i]), file=f)
            elif _rc == 0:
                ifgd_ok = ifgdates2[i] ## readable tiff
        
        ### Read info
        ## If all float already exist, this will not be done, but no problem because
        ## par files should alerady exist!
        if ifgd_ok:
            unw_tiffile = os.path.join(geocdir, ifgd_ok, ifgd_ok+'.geo.unw.tif')
            geotiff = gdal.Open(unw_tiffile)
            width = geotiff.RasterXSize
            length = geotiff.RasterYSize
            lon_w_p, dlon, _, lat_n_p, _, dlat = geotiff.GetGeoTransform()
            ## lat lon are in pixel registration. dlat is negative
            lon_w_g = lon_w_p + dlon/2
            lat_n_g = lat_n_p + dlat/2
            ## to grit registration by shifting half pixel inside
            if nlook != 1:
                width = int(width/nlook)
                length = int(length/nlook)
                dlon = dlon*nlook
                dlat = dlat*nlook


    #%% EQA.dem_par, slc.mli.par
    if not os.path.exists(mlipar):
        print('\nCreate slc.mli.par', flush=True)
#        radar_freq = 5.405e9 ## fixed for Sentnel-1

        with open(mlipar, 'w') as f:
            print('range_samples:   {}'.format(width), file=f)
            print('azimuth_lines:   {}'.format(length), file=f)
            print('radar_frequency: {} Hz'.format(radar_freq), file=f)
            if center_time is not None:
                print('center_time: {}'.format(center_time), file=f)

    if not os.path.exists(dempar):
        print('\nCreate EQA.dem_par', flush=True)

        text = ["Gamma DIFF&GEO DEM/MAP parameter file",
              "title: DEM", 
              "DEM_projection:     EQA",
              "data_format:        REAL*4",
              "DEM_hgt_offset:          0.00000",
              "DEM_scale:               1.00000",
              "width: {}".format(width), 
              "nlines: {}".format(length), 
              "corner_lat:     {}  decimal degrees".format(lat_n_g), 
              "corner_lon:    {}  decimal degrees".format(lon_w_g), 
              "post_lat: {} decimal degrees".format(dlat), 
              "post_lon: {} decimal degrees".format(dlon), 
              "", 
              "ellipsoid_name: WGS 84", 
              "ellipsoid_ra:        6378137.000   m",
              "ellipsoid_reciprocal_flattening:  298.2572236",
              "",
              "datum_name: WGS 1984",
              "datum_shift_dx:              0.000   m",
              "datum_shift_dy:              0.000   m",
              "datum_shift_dz:              0.000   m",
              "datum_scale_m:         0.00000e+00",
              "datum_rotation_alpha:  0.00000e+00   arc-sec",
              "datum_rotation_beta:   0.00000e+00   arc-sec",
              "datum_rotation_gamma:  0.00000e+00   arc-sec",
              "datum_country_list: Global Definition, WGS84, World\n"]
    
        with open(dempar, 'w') as f:
            f.write('\n'.join(text))


    #%% bperp
    print('\nCopy baselines file', flush=True)
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    if os.path.exists(bperp_file_in):
        ## Check exisiting bperp_file
        if not io_lib.read_bperp_file(bperp_file_in, imdates):
            print('  baselines file found, but not complete. Make dummy', flush=True)
            io_lib.make_dummy_bperp(bperp_file_out, imdates)
        else:
            shutil.copyfile(bperp_file_in, bperp_file_out)
    else:
        print('  No valid baselines file exists. Make dummy.', flush=True)
        io_lib.make_dummy_bperp(bperp_file_out, imdates)


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(outdir)))


#%%
def convert_wrapper(i):
    ifgd = ifgdates2[i]
    if np.mod(i,10) == 0:
        print("  {0:3}/{1:3}th IFG...".format(i, len(ifgdates2)), flush=True)

    unw_tiffile = os.path.join(geocdir, ifgd, ifgd+'.geo.unw.tif')
    cc_tiffile = os.path.join(geocdir, ifgd, ifgd+'.geo.cc.tif')

    ### Check if inputs exist
    if not os.path.exists(unw_tiffile):
        print ('  No {} found. Skip'.format(ifgd+'.geo.unw.tif'), flush=True)
        return 1
    elif not os.path.exists(cc_tiffile):
        print ('  No {} found. Skip'.format(ifgd+'.geo.cc.tif'), flush=True)
        return 1

    ### Output dir and files
    ifgdir1 = os.path.join(outdir, ifgd)
    if not os.path.exists(ifgdir1): os.mkdir(ifgdir1)
    unwfile = os.path.join(ifgdir1, ifgd+'.unw')
    ccfile = os.path.join(ifgdir1, ifgd+'.cc')

    ### Read data from geotiff
    try:
        unw = gdal.Open(unw_tiffile).ReadAsArray()
        unw[unw==0] = np.nan
    except: ## if broken
        print ('  {} cannot open. Skip'.format(ifgd+'.geo.unw.tif'), flush=True)
        shutil.rmtree(ifgdir1)
        return 1

    try:
        cc = gdal.Open(cc_tiffile).ReadAsArray()
        if cc.dtype == np.float32:
            cc = cc*255 ## 0-1 -> 0-255 to output in uint8
    except: ## if broken
        print ('  {} cannot open. Skip'.format(ifgd+'.geo.cc.tif'), flush=True)
        shutil.rmtree(ifgdir1)
        return 1

    ### Multilook
    if nlook != 1:
        unw = tools_lib.multilook(unw, nlook, nlook, n_valid_thre)
        cc = cc.astype(np.float32)
        cc[cc==0] = np.nan
        cc = tools_lib.multilook(cc, nlook, nlook, n_valid_thre)

    ### Output float
    unw.tofile(unwfile)
    cc = cc.astype(np.uint8) ##nan->0, max255, auto-floored
    cc.tofile(ccfile)

    ### Make png
    unwpngfile = os.path.join(ifgdir1, ifgd+'.unw.png')
    plot_lib.make_im_png(np.angle(np.exp(1j*unw/cycle)*cycle), unwpngfile, cmap, ifgd+'.unw', vmin=-np.pi, vmax=np.pi, cbar=False)
    
    return 0


#%% main
if __name__ == "__main__":
    sys.exit(main())

