#!/usr/bin/env python3
"""
v1.4 20200503 Yu Morishita, GSI

========
Overview
========
This script downloads GeoTIFF files of unw (unwrapped interferogram) and cc (coherence) in the specified frame ID from COMET-LiCS web portal. The -f option is not necessary when the frame ID can be automatically identified from the name of the working directory. GACOS data can also be downloaded if available. Existing GeoTIFF files are not re-downloaded to save time, i.e., only the newly available data will be downloaded.

============
Output files
============
 - GEOC/
   - yyyymmdd_yyyymmdd/
     - yyyymmdd_yyyymmdd.geo.unw.tif
     - yyyymmdd_yyyymmdd.geo.cc.tif
  [- *.geo.mli.tif (using just one first epoch)]
   - *.geo.E.tif
   - *.geo.N.tif
   - *.geo.U.tif
   - *.geo.hgt.tif
   - baselines
   - metadata.txt
[- GACOS/] (if --get_gacos is used and GACOS data available on COMET-LiCS web)
  [- yyyymmdd.sltd.geo.tif]

=====
Usage
=====
LiCSBAS01_get_geotiff.py [-f frameID] [-s yyyymmdd] [-e yyyymmdd] [--get_gacos]

 -f  Frame ID (e.g., 021D_04972_131213). (Default: Read from directory name)
 -s  Start date (Default: 20141001)
 -e  End date (Default: Today)
 --get_gacos  Download GACOS data as well if available
 
"""
#%% Change log
'''
v1.4 20200503 Yu Morishita, GSI
 - Update download_data (thanks to sahitono)
v1.3 20200311 Yu Morishita, Uni of Leeds and GSI
 - Deal with only new LiCSAR file structure
v1.2 20200302 Yu Morishita, Uni of Leeds and GSI
 - Compatible with new LiCSAR file structure (backward-compatible)
 - Add --get_gacos option
v1.1 20191115 Yu Morishita, Uni of Leeds and GSI
 - Download mli and hgt
v1.0 20190729 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''


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
    ver=1.2; date=20200227; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    frameID = []
    startdate = 20141001
    enddate = int(dt.date.today().strftime("%Y%m%d"))
    get_gacos = False


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hf:s:e:", ["help", "get_gacos"])
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
            elif o == '--get_gacos':
                get_gacos = True

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


    #%% ENU and hgt
    for ENU in ['E', 'N', 'U', 'hgt']:
        enutif = '{}.geo.{}.tif'.format(frameID, ENU)
        if os.path.exists(enutif):
            print('{} already exist. Skip download.'.format(enutif), flush=True)
            continue
        
        print('Download {}'.format(enutif), flush=True)

        url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', enutif)
        tools_lib.download_data(url, enutif)

    #%% baselines and metadata.txt
    print('Download baselines', flush=True)
    url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', 'baselines')
    tools_lib.download_data(url, 'baselines')

    print('Download metadata.txt', flush=True)
    url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', 'metadata.txt')
    tools_lib.download_data(url, 'metadata.txt')


    #%% mli
    ### Get available dates
    url = os.path.join(LiCSARweb, trackID, frameID, 'epochs')
    response = requests.get(url)
    
    response.encoding = response.apparent_encoding #avoid garble
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "html.parser")
    tags = soup.find_all(href=re.compile(r"\d{8}"))
    imdates_all = [tag.get("href")[0:8] for tag in tags]
    _imdates = np.int32(np.array(imdates_all))
    _imdates = (_imdates[(_imdates>=startdate)*(_imdates<=enddate)]).astype('str').tolist()
    
    ## Find earliest date in which mli is available
    imd1 = []
    for imd in _imdates: 
        url_mli = os.path.join(url, imd, imd+'.geo.mli.tif')
        response = requests.head(url_mli)
        if  response.ok:
            imd1 = imd
            break

    ### Download
    if imd1:
        print('Donwnloading {}.geo.mli.tif as {}.geo.mli.tif...'.format(imd1, frameID), flush=True)
        url_mli = os.path.join(url, imd1, imd1+'.geo.mli.tif')
        mlitif = frameID+'.geo.mli.tif'
        if os.path.exists(mlitif):
            print('    {} already exist. Skip'.format(mlitif), flush=True)
        else:
            tools_lib.download_data(url_mli, mlitif)
    else:
        print('No mli available on {}'.format(url), file=sys.stderr, flush=True)


    #%% GACOS if specified
    if get_gacos:
        gacosdir = os.path.join(wd, 'GACOS')
        if not os.path.exists(gacosdir): os.mkdir(gacosdir)

        ### Get available dates
        print('\nDownload GACOS data', flush=True)
        url = os.path.join(LiCSARweb, trackID, frameID, 'epochs')
        response = requests.get(url)
        response.encoding = response.apparent_encoding #avoid garble
        html_doc = response.text
        soup = BeautifulSoup(html_doc, "html.parser")
        tags = soup.find_all(href=re.compile(r"\d{8}"))
        imdates_all = [tag.get("href")[0:8] for tag in tags]
        _imdates = np.int32(np.array(imdates_all))
        _imdates = (_imdates[(_imdates>=startdate)*(_imdates<=enddate)]).astype('str').tolist()

        ### Extract available dates
        imdates = []
        for imd in _imdates:
            url_sltd = os.path.join(url, imd, imd+'.sltd.geo.tif')
            response = requests.get(url_sltd)
            if response.ok:
                imdates.append(imd)

        n_im = len(imdates)
        if n_im > 0:
            print('{} GACOS data available from {} to {}'.format(n_im, imdates[0], imdates[-1]), flush=True)
        else:
            print('No GACOS data available from {} to {}'.format(startdate, enddate), flush=True)
        
        ### Download
        for i, imd in enumerate(imdates):
            print('  Donwnloading {} ({}/{})...'.format(imd, i+1, n_im), flush=True)
            url_sltd = os.path.join(url, imd, imd+'.sltd.geo.tif')
            path_sltd = os.path.join(gacosdir, imd+'.sltd.geo.tif')
            if os.path.exists(path_sltd):
                print('    {}.sltd.geo.tif already exist. Skip'.format(imd), flush=True)
            else:
                tools_lib.download_data(url_sltd, path_sltd)
    

    #%% unw and cc
    ### Get available dates
    print('\nDownload geotiff of unw and cc', flush=True)
    url = os.path.join(LiCSARweb, trackID, frameID, 'interferograms')
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
        else:
            tools_lib.download_data(url_unw, path_unw)

        url_cc = os.path.join(url, ifgd, ifgd+'.geo.cc.tif')
        path_cc = os.path.join(ifgd, ifgd+'.geo.cc.tif')
        if os.path.exists(path_cc):
            print('    {}.geo.cc.tif already exist. Skip.'.format(ifgd), flush=True)
        tools_lib.download_data(url_cc, path_cc)
   

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

