#!/usr/bin/env python3
"""
========
Overview
========
This script converts geotiff files to float format for further time series analysis, and also downsamples (multilooks) data if specified.

=========
Changelog
=========
v1.1 20190824 Yu Morishita, Uni of Leeds and GSI
 - Skip broken geotiff
v1.0 20190731 Yu Morishita, Uni of Leeds and GSI
 - Original implementation

===============
Input & output files
===============
Inputs:
 - GEOC/    
   - yyyymmdd_yyyymmdd
     - yyyymmdd_yyyymmdd.geo.unw.tif
     - yyyymmdd_yyyymmdd.geo.cc.tif
    [- yyyymmdd_yyyymmdd.geo.diff_mag.tif] (if exist, just one file is used for slc.mli)
  [- *.geo.[ENU].tif] (if not exist, try to download from LiCSAR portal)
  [- baselines] (if not exist, try to download from LiCSAR portal or make dummy)

Outputs in GEOCml? directory (all binary files are 4byte float little endian):
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.unw[.png] (downsampled if indicated)
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.cc[.png] (downsampled if indicated)
 - baselines (may be dummy)
 - EQA.dem_par
 - slc.mli[.par|.png]
 - E.geo (if exist)
 - N.geo (if exist)
 - U.geo (if exist)
 - no_unw_list.txt (if exist)

=====
Usage
=====
LiCSBAS02_ml_prep.py -i GEOCdir [-o GEOCmldir] [-n nlook] [-f FRAME]

 -i  Path to the input GEOC dir containing stack of geotiff data
 -o  Path to the output GEOCml dir (Default: GEOCml[nlook])
 -n  Number of donwsampling factor (Default: 1, no donwsampling)
 -f  Frame ID (e.g., 021D_04972_131213). Used only for downloading ENU
     (Default: Read from directory name)

"""


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
    geocdir = []
    outdir = []
    nlook = 1
    frameID = []
    cmap = 'insar'
    cycle = 3
    n_valid_thre = 0.5


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:n:f:", ["help"])
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
            elif o == '-f':
                frameID = a

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

    mlifile = os.path.join(outdir, 'slc.mli')

    mlipar = os.path.join(outdir, 'slc.mli.par')
    dempar = os.path.join(outdir, 'EQA.dem_par')

    no_unw_list = os.path.join(outdir, 'no_unw_list.txt')
    if os.path.exists(no_unw_list): os.remove(no_unw_list)

    bperp_file_in = os.path.join(geocdir, 'baselines')
    bperp_file_out = os.path.join(outdir, 'baselines')

    LiCSARweb = 'http://gws-access.ceda.ac.uk/public/nceo_geohazards/LiCSAR_products/'

    ### Frame ID even if not used
    if not frameID: ## if not specified
        _tmp = re.findall(r'\d{3}[AD]_\d{5}_\d{6}', geocdir)
        ##e.g., 021D_04972_131213
        if len(_tmp)!=0: ## if not found, keep []
             frameID = _tmp[0]
             trackID = str(int(frameID[0:3]))
    else:
        trackID = str(int(frameID[0:3]))


    #%% ENU
    for ENU in ['E', 'N', 'U']:
        print('\nCreate {}'.format(ENU+'.geo'), flush=True)
        enutif = glob.glob(os.path.join(geocdir, '*.geo.{}.tif'.format(ENU)))

        ### Download if not exist
        if len(enutif)==0:
            print('  No *.geo.{}.tif found in {}'.format(ENU, os.path.basename(geocdir)), flush=True)

            if not frameID: ## if frameID not found above
                print('  Frame ID cannot be identified from dir name!', file=sys.stderr)
                print('  Use -f option if you need {}.geo'.format(ENU), file=sys.stderr)
                continue

            ### Download tif
            url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', '{}.geo.{}.tif'.format(frameID, ENU))
            enutif = os.path.join(geocdir, '{}.geo.{}.tif'.format(frameID, ENU))
            if not tools_lib.download_data(url, enutif):
                print('  Error while downloading from {}'.format(url), file=sys.stderr, flush=True)
                continue
            else:
                print('  {} dowonloaded from LiCSAR-portal'.format(os.path.basename(url)), flush=True)
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

    ### Create
    for i, ifgd in enumerate(reversed(ifgdates2)): ## From latest for mli
        if np.mod(i,10) == 0:
            print("  {0:3}/{1:3}th IFG...".format(i, n_ifg2), flush=True)

        unw_tiffile = os.path.join(geocdir, ifgd, ifgd+'.geo.unw.tif')
        cc_tiffile = os.path.join(geocdir, ifgd, ifgd+'.geo.cc.tif')

        ### Check if inputs exist
        if not os.path.exists(unw_tiffile) or not os.path.exists(cc_tiffile):
            print ('  No {} found. Skip'.format(ifgd+'.geo.[unw|cc].tif'), flush=True)
            with open(no_unw_list, 'a') as f:
                print('{}'.format(ifgd), file=f)
            continue

        ### Output dir and files
        ifgdir1 = os.path.join(outdir, ifgd)
        if not os.path.exists(ifgdir1): os.mkdir(ifgdir1)
        unwfile = os.path.join(ifgdir1, ifgd+'.unw')
        ccfile = os.path.join(ifgdir1, ifgd+'.cc')

       ### Read data from geotiff
        try:
            unw = gdal.Open(unw_tiffile).ReadAsArray()
            unw[unw==0] = np.nan
            cc = gdal.Open(cc_tiffile).ReadAsArray()
            cc[cc==0] = np.nan
        except: ## if broken
            print ('  {} cannot open. Skip'.format(ifgd+'.geo.[unw|cc].tif'), flush=True)
            with open(no_unw_list, 'a') as f:
                print('{}'.format(ifgd), file=f)
            shutil.rmtree(ifgdir1)
            continue

        ### Make mli (only once)
        if not os.path.exists(mlifile):
            diffmag_tiffile = os.path.join(geocdir, ifgd, ifgd+'.geo.diff_mag.tif')
            if os.path.exists(diffmag_tiffile):
                mag = gdal.Open(diffmag_tiffile).ReadAsArray()
                mag[mag==0] = np.nan
                if nlook != 1:
                    mag = tools_lib.multilook(mag, nlook, nlook, n_valid_thre)
                mag.tofile(mlifile)
                mlipngfile = mlifile+'.png'
                plot_lib.make_im_png(mag, mlipngfile, 'gray', 'MLI', cbar=False)

        ### Read info (only once)
        ## If all float already exist, this is not done, but no problem because
        ## par files should alerady be exits!
        if not 'length' in locals():
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

        ### Multilook
        if nlook != 1:
            unw = tools_lib.multilook(unw, nlook, nlook, n_valid_thre)
            cc = tools_lib.multilook(cc, nlook, nlook, n_valid_thre)

        ### Output float
        unw.tofile(unwfile)
        cc.tofile(ccfile)

        ### Make png
        unwpngfile = os.path.join(ifgdir1, ifgd+'.unw.png')
        plot_lib.make_im_png(np.angle(np.exp(1j*unw/cycle)*cycle), unwpngfile, cmap, ifgd+'.unw', vmin=-np.pi, vmax=np.pi, cbar=False)



    #%% EQA.dem_par, slc.mli.par
    if not os.path.exists(mlipar):
        print('\nCreate slc.mli.par', flush=True)
        radar_freq = 5.405e9 ## fixed for Sentnel-1

        with open(mlipar, 'w') as f:
            print('range_samples:   {}'.format(width), file=f)
            print('azimuth_lines:   {}'.format(length), file=f)
            print('radar_frequency: {} Hz'.format(radar_freq), file=f)

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
        print('  No valid baselines exists.', flush=True)
        if not frameID: ## if frameID not found above
            print('  Frame ID cannot be identified from dir name!')
            print('  Make dummy.', flush=True)
            io_lib.make_dummy_bperp(bperp_file_out, imdates)
        else:
            print('  Try download.', flush=True)
            url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', 'baselines')
            if not tools_lib.download_data(url, bperp_file_out):
                print('  Error while downloading from {}.\n  Make dummy.'.format(url), file=sys.stderr, flush=True)
                io_lib.make_dummy_bperp(bperp_file_out, imdates)
            else:
                print('  {} dowonloaded from LiCSAR-portal'.format(os.path.basename(url)), flush=True)
                if not io_lib.read_bperp_file(bperp_file_out, imdates):
                    print('  but not complete. Make dummy.', flush=True)
                    io_lib.make_dummy_bperp(bperp_file_out, imdates)


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(outdir)))


#%% main
if __name__ == "__main__":
    sys.exit(main())

