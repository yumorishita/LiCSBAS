#!/usr/bin/env python3
"""
========
Overview
========
This script downloads geotiff files of unw (unwrapped interferogram) and cc (coherence) for the specified frame ID from COMET-LiCS web. The -f option is not necessary because the frame ID can be automatically identified from the name of the working directory.

=========
Changelog
=========
v1.0 20190729 Yu Morishita, Uni of Leeds and GSI
 - Original implementation

============
Output files
============
 - GEOC/    
   - yyyymmdd_yyyymmdd
     - yyyymmdd_yyyymmdd.geo.unw.tif
     - yyyymmdd_yyyymmdd.geo.cc.tif
     - yyyymmdd_yyyymmdd.geo.diff_mag.tif (for just one latest ifg)
   - *.geo.E.tif
   - *.geo.N.tif
   - *.geo.U.tif
   - baselines

=====
Usage
=====
LiCSBAS01_get_geotiff.py [-f FRAME] [-s start_date] [-e end_date]

 -f  Frame ID (e.g., 021D_04972_131213). (Default: Read from directory name)
 -s  Start date (Default: 20141001)
 -e  End date (Default: Today)
 
"""


#%% Import
import getopt
import os
import re
import sys
import time
import requests
from bs4 import BeautifulSoup
import numpy as np
import datetime as dt
import LiCSBAS_tools_lib as tools_lib

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
    frameID = []
    startdate = 20141001
    enddate = int(dt.date.today().strftime("%Y%m%d"))


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hf:s:e:", ["help"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-f':
                frameID = a
            elif o == '-s':
                startdate = int(a)
            elif o == '-e':
                enddate = int(a)

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2
 

    #%% Determine frameID
    wd = os.getcwd()
    if not frameID: ## if frameID not indicated
        _tmp = re.findall(r'\d{3}[AD]_\d{5}_\d{6}', wd)
        ##e.g., 021D_04972_131213
        if len(_tmp)==0:
            print('\nFrame ID cannot be identified from dir name!', file=sys.stderr)
            print('Use -f option', file=sys.stderr)
            return
        else:
            frameID = _tmp[0]
            print('\nFrame ID is {}\n'.format(frameID), flush=True)
    trackID = str(int(frameID[0:3]))


    #%% Directory and file setting
    outdir = os.path.join(wd, 'GEOC')
    if not os.path.exists(outdir): os.mkdir(outdir)
    os.chdir(outdir)

    LiCSARweb = 'http://gws-access.ceda.ac.uk/public/nceo_geohazards/LiCSAR_products/'


    #%% ENU
    for ENU in ['E', 'N', 'U']:
        enutif = '{}.geo.{}.tif'.format(frameID, ENU)
        if os.path.exists(enutif):
            print('{} already exist. Skip download.'.format(enutif), flush=True)
            continue
        
        print('Download {}'.format(enutif), flush=True)

        url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', enutif)
        if not tools_lib.download_data(url, enutif):
            print('  Error while downloading from {}'.format(url), file=sys.stderr, flush=True)
            continue

    #%% baselines
    print('Download baselines', flush=True)
    url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', 'baselines')
    if not tools_lib.download_data(url, 'baselines'):
        print('  Error while downloading from {}'.format(url), file=sys.stderr, flush=True)


    #%% unw and cc
    ### Get available dates
    print('\nDownload geotiff of unw and cc', flush=True)
    url = os.path.join(LiCSARweb, trackID, frameID, 'products')
    response = requests.get(url)
    response.encoding = response.apparent_encoding #avoid garble
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "html.parser")
    tags = soup.find_all(href=re.compile(r"\d{8}_\d{8}"))
    ifgdates_all = [tag.get("href")[0:17] for tag in tags]
    
    ### Extract during start_date to end_date
    ifgdates = []
    for ifgd in ifgdates_all:
        mimd = int(ifgd[:8])
        simd = int(ifgd[-8:])
        if mimd >= startdate and simd <= enddate:
            ifgdates.append(ifgd)
    
    n_ifg = len(ifgdates)
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    print('{} IFGs available from {} to {}'.format(n_ifg, imdates[0], imdates[-1]), flush=True)
    
    ### Download
    for i, ifgd in enumerate(ifgdates):
        print('  Donwnloading {} ({}/{})...'.format(ifgd, i+1, n_ifg), flush=True)
        url_unw = os.path.join(url, ifgd, ifgd+'.geo.unw.tif')
        path_unw = os.path.join(ifgd, ifgd+'.geo.unw.tif')
        if not os.path.exists(ifgd): os.mkdir(ifgd)
        if os.path.exists(path_unw):
            print('    {}.geo.unw.tif already exist. Skip'.format(ifgd), flush=True)
        elif not tools_lib.download_data(url_unw, path_unw):
            print('    Error while downloading from {}'.format(url_unw), file=sys.stderr, flush=True)

        url_cc = os.path.join(url, ifgd, ifgd+'.geo.cc.tif')
        path_cc = os.path.join(ifgd, ifgd+'.geo.cc.tif')
        if os.path.exists(path_cc):
            print('    {}.geo.cc.tif already exist. Skip.'.format(ifgd), flush=True)
        if not tools_lib.download_data(url_cc, path_cc):
            print('    Error while downloading from {}'.format(url_cc), file=sys.stderr, flush=True)
   

    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(outdir))


#%% main
if __name__ == "__main__":
    sys.exit(main())

