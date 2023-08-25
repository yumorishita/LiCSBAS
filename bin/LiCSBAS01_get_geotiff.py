#!/usr/bin/env python3
"""
v1.14.1 20230608 Milan Lazecky, UoL
v1.6.3 20201207 Yu Morishita, GSI

========
Overview
========
This script downloads GeoTIFF files in the specified frame ID from COMET-LiCS web portal.
By default, unw (unwrapped interferogram) and cc (coherence) files are downloaded
The -f option is not necessary when the frame ID can be automatically identified from the name of the working directory.
GACOS data can also be downloaded if available. Existing GeoTIFF files are not re-downloaded to save time, i.e., only the newly available data will be downloaded.

============
Output files
============
 - GEOC/
   - yyyymmdd_yyyymmdd/
     - yyyymmdd_yyyymmdd.geo.unw.tif
     - yyyymmdd_yyyymmdd.geo.cc.tif
    [- yyyymmdd_yyyymmdd.geo.diff_pha.tif] (if --get_pha is used)
    [- yyyymmdd_yyyymmdd.geo.diff_unfiltered_pha.tif] (if --get_pha is used)
  [- *.geo.mli.tif (using just one first epoch)]
   - *.geo.E.tif
   - *.geo.N.tif
   - *.geo.U.tif
   - *.geo.hgt.tif
   - baselines
   - metadata.txt
[- GACOS/] (if --get_gacos is used and GACOS data available on COMET-LiCS web)
  [- yyyymmdd.sltd.geo.tif]
[- GEOC.MLI/] (if --get_mli is used)
  [- yyyymmdd.mli.geo.tif]
=====
Usage
=====
LiCSBAS01_get_geotiff.py [-f frameID] [-s yyyymmdd] [-e yyyymmdd] [--get_gacos] [--get_mli] [--n_para int]

 -f  Frame ID (e.g., 021D_04972_131213). (Default: Read from directory name)
 -s  Start date (Default: 20141001)
 -e  End date (Default: Today)
 --get_pha    Download also wrapped phase data (if available)
 --get_gacos  Download GACOS data as well if available
 --get_mli  Download MLI (multilooked intensity) data as well if available
 --n_para  Number of parallel downloading (Default: 4)

"""
#%% Change log
'''
v1.14.1 20230608 Milan Lazecky, UoL
 - added download of phase (for reunw) and mli
v1.6.3 20201207 Yu Morishita, GSI
 - Download network.png
 - Search latest epoch for mli to save times
v1.6.2 20201118 Yu Morishita, GSI
 - Again Bug fix of multiprocessing
v1.6.1 20201116 Yu Morishita, GSI
 - Bug fix of multiprocessing in Mac python>=3.8
v1.6 20200911 Yu Morishita, GSI
 - Parallel downloading
 - Check time stamp and size
 - Log size, elapsed time, and download speed
v1.5 20200623 Yu Morishita, GSI
 - Speed up (small bug fix) when re-downloading
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
import multiprocessing as multi
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
    ver='1.14.1'; date=20230628; author="Y. Morishita, M. Lazecky"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    frameID = []
    startdate = 20141001
    enddate = int(dt.date.today().strftime("%Y%m%d"))
    get_gacos = False
    get_mli = False
    get_pha = False
    n_para = 4

    q = multi.get_context('fork')


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hf:s:e:", ["help", "get_gacos", "get_mli", "get_pha", "n_para="])
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
            elif o == '--get_mli':
                get_mli = True
            elif o == '--get_pha':
                get_pha = True
            elif o == '--n_para':
                n_para = int(a)

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
    else:
        print('\nFrame ID is {}\n'.format(frameID), flush=True)

    trackID = str(int(frameID[0:3]))


    #%% Directory and file setting
    outdir = os.path.join(wd, 'GEOC')
    if not os.path.exists(outdir): os.mkdir(outdir)
    os.chdir(outdir)

    LiCSARweb = 'https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/'


    #%% ENU and hgt
    for ENU in ['E', 'N', 'U', 'hgt']:
        enutif = '{}.geo.{}.tif'.format(frameID, ENU)
        url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', enutif)
        if os.path.exists(enutif):
            rc = tools_lib.comp_size_time(url, enutif)
            if rc == 0:
                print('{} already exist. Skip download.'.format(enutif), flush=True)
                continue
            elif rc == 3:
                print('{} not available. Skip download.'.format(enutif), flush=True)
                continue
            else:
                if rc == 1:
                    print("Size of {} is not identical.".format(enutif))
                elif rc == 2:
                    print("Newer {} available.".format(enutif))
        
        print('Download {}'.format(enutif), flush=True)
        tools_lib.download_data(url, enutif)

    print('', flush=True)


    #%% baselines, network.png and metadata.txt
    print('Download baselines', flush=True)
    url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', 'baselines')
    tools_lib.download_data(url, 'baselines')

    print('Download network.png', flush=True)
    url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', 'network.png')
    tools_lib.download_data(url, 'network.png')

    print('Download metadata.txt', flush=True)
    url = os.path.join(LiCSARweb, trackID, frameID, 'metadata', 'metadata.txt')
    tools_lib.download_data(url, 'metadata.txt')

    print('', flush=True)


    #%% mli
    mlitif = frameID+'.geo.mli.tif'
    if os.path.exists(mlitif):
        print('{} already exist. Skip.'.format(mlitif), flush=True)
    else:
        ### Get available dates
        print('Searching latest epoch for mli...', flush=True)
        url = os.path.join(LiCSARweb, trackID, frameID, 'epochs')
        response = requests.get(url)
        
        response.encoding = response.apparent_encoding #avoid garble
        html_doc = response.text
        soup = BeautifulSoup(html_doc, "html.parser")
        tags = soup.find_all(href=re.compile(r"\d{8}"))
        imdates_all = [tag.get("href")[0:8] for tag in tags]
        _imdates = np.int32(np.array(imdates_all))
        _imdates = (_imdates[(_imdates>=startdate)*(_imdates<=enddate)]).astype('str').tolist()
        
        ## Find latest date in which mli is available
        imd1 = []
        for i, imd in enumerate(reversed(_imdates)):
            if np.mod(i, 10) == 0:
                print("\r  {0:3}/{1:3}".format(i, len(_imdates)), end='', flush=True)
            url_epoch = os.path.join(url, imd)
            response = requests.get(url_epoch)
            response.encoding = response.apparent_encoding #avoid garble
            html_doc = response.text
            soup = BeautifulSoup(html_doc, "html.parser")
            tag = soup.find(href=re.compile(r"\d{8}.geo.mli.tif"))
            if tag is not None:
                print('\n{} found as latest.'.format(imd))
                imd1 = imd
                break
    
        ### Download
        if imd1:
            print('Downloading {}.geo.mli.tif as {}.geo.mli.tif...'.format(imd1, frameID), flush=True)
            url_mli = os.path.join(url, imd1, imd1+'.geo.mli.tif')
            tools_lib.download_data(url_mli, mlitif)
        else:
            print('\nNo mli available on {}'.format(url), file=sys.stderr, flush=True)

    print('', flush=True)


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
        print('  There are {} epochs from {} to {}'.format(len(_imdates),
                                       startdate, enddate), flush=True)

        ### Extract available dates
        print('  Searching available epochs ({} parallel)...'.format(n_para), flush=True)

        args = [(i, len(_imdates),
                 os.path.join(url, imd, '{}.sltd.geo.tif'.format(imd)),
                 os.path.join(gacosdir, imd+'.sltd.geo.tif')
                 ) for i, imd in enumerate(_imdates)]
    
        p = q.Pool(n_para)
        rc = p.map(check_gacos_wrapper, args)
        p.close()

        n_im_existing = 0
        n_im_unavailable = 0
        imdates_dl = []
        for i, rc1 in enumerate(rc):
            if rc1 == 0:  ## No need to download
                n_im_existing = n_im_existing + 1
            if rc1 == 3 or rc1 == 5:  ## Can not download
                n_im_unavailable = n_im_unavailable + 1
            elif rc1 == 1 or rc1 == 2  or rc1 == 4:  ## Need download
                imdates_dl.append(_imdates[i])

        n_im_dl = len(imdates_dl)

        if n_im_existing > 0:
            print('  {} GACOS data already downloaded'.format(n_im_existing), flush=True)
        if n_im_unavailable > 0:
            print('  {} GACOS data unavailable'.format(n_im_unavailable), flush=True)

        ### Download
        if n_im_dl > 0:
            print('{} GACOS data will be downloaded'.format(n_im_dl), flush=True)
            print('Download GACOS ({} parallel)...'.format(n_para), flush=True)
            ### Download
            args = [(i, imd, n_im_dl,
                     os.path.join(url, imd, '{}.sltd.geo.tif'.format(imd)),
                     os.path.join(gacosdir, '{}.sltd.geo.tif'.format(imd))
                     ) for i, imd in enumerate(imdates_dl)]
            
            p = q.Pool(n_para)
            p.map(download_wrapper, args)
            p.close()
        else:
            print('No GACOS data available from {} to {}'.format(startdate, enddate), flush=True)
    
    print('', flush=True)


    #%% InSAR data
    ### Get available dates
    print('\nDownload geotiff of InSAR products', flush=True)
    url_ifgdir = os.path.join(LiCSARweb, trackID, frameID, 'interferograms')
    response = requests.get(url_ifgdir)
    
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

    ### Check if both unw and cc already donwloaded, new, and same size
    print('Checking and downloading ({} parallel, may take time)...'.format(n_para), flush=True)
    exts = ['unw', 'cc']
    if get_pha:
        exts = exts + ['diff_pha', 'diff_unfiltered_pha'] # some ifgs do not have unfiltered version, so getting both
    for ext in exts:
        print(ext + ' data:')
        args = [(i, n_ifg,
                 os.path.join(url_ifgdir, ifgd, '{0}.geo.{1}.tif'.format(ifgd, ext)),
                 os.path.join(ifgd, '{0}.geo.{1}.tif'.format(ifgd, ext))
                 ) for i, ifgd in enumerate(ifgdates)]

        p = q.Pool(n_para)
        rc = p.map(check_exist_wrapper, args)
        p.close()

        n_existing = 0
        dates_dl = []
        for i, rc1 in enumerate(rc):
            if rc1 == 0:  ## No need to download
                n_existing = n_existing + 1
            if rc1 == 3 or rc1 == 5:  ## Can not download
                print('  {0}.geo.{1}.tif not available.'.format(str(ifgdates[i]), ext), flush=True)
            elif rc1 == 1 or rc1 == 2  or rc1 == 4:  ## Need download
                dates_dl.append(ifgdates[i])

        n_dl = len(dates_dl)
        print('{} already downloaded'.format(n_existing), flush=True)
        ### Download with parallel
        if n_dl != 0:
            print('\nDownload {0} ({1} parallel)...'.format(ext, str(n_para)), flush=True)
            args = [(i, ifgd, n_dl,
                     os.path.join(url_ifgdir, ifgd, '{0}.geo.{1}.tif'.format(ifgd, ext)),
                     os.path.join(ifgd, '{0}.geo.{1}.tif'.format(ifgd, ext))
                     ) for i, ifgd in enumerate(dates_dl)]

            p = q.Pool(n_para)
            p.map(download_wrapper, args)
            p.close()

    # %% MLIs if specified
    if get_mli:
        mlidir = os.path.join(wd, 'GEOC.MLI')
        if not os.path.exists(mlidir): os.mkdir(mlidir)

        ### Get available dates
        print('\nDownload MLI data', flush=True)
        url = os.path.join(LiCSARweb, trackID, frameID, 'epochs')
        response = requests.get(url)
        response.encoding = response.apparent_encoding  # avoid garble
        html_doc = response.text
        soup = BeautifulSoup(html_doc, "html.parser")
        tags = soup.find_all(href=re.compile(r"\d{8}"))
        imdates_all = [tag.get("href")[0:8] for tag in tags]
        _imdates = np.int32(np.array(imdates_all))
        _imdates = (_imdates[(_imdates >= startdate) * (_imdates <= enddate)]).astype('str').tolist()
        print('  There are {} epochs from {} to {}'.format(len(_imdates),
                                                           startdate, enddate), flush=True)

        ### Extract available dates
        print('  Searching available epochs ({} parallel)...'.format(n_para), flush=True)

        args = [(i, len(_imdates),
                 os.path.join(url, imd, '{}.mli.geo.tif'.format(imd)),
                 os.path.join(mlidir, imd + '.mli.geo.tif')
                 ) for i, imd in enumerate(_imdates)]

        # will use the same for gacos
        p = q.Pool(n_para)
        rc = p.map(check_gacos_wrapper, args)
        p.close()

        n_im_existing = 0
        n_im_unavailable = 0
        imdates_dl = []
        for i, rc1 in enumerate(rc):
            if rc1 == 0:  ## No need to download
                n_im_existing = n_im_existing + 1
            if rc1 == 3 or rc1 == 5:  ## Can not download
                n_im_unavailable = n_im_unavailable + 1
            elif rc1 == 1 or rc1 == 2 or rc1 == 4:  ## Need download
                imdates_dl.append(_imdates[i])

        n_im_dl = len(imdates_dl)

        if n_im_existing > 0:
            print('  {} MLI data already downloaded'.format(n_im_existing), flush=True)
        if n_im_unavailable > 0:
            print('  {} MLI data unavailable'.format(n_im_unavailable), flush=True)

        ### Download
        if n_im_dl > 0:
            print('{} MLI data will be downloaded'.format(n_im_dl), flush=True)
            print('Download MLI ({} parallel)...'.format(n_para), flush=True)
            ### Download
            args = [(i, imd, n_im_dl,
                     os.path.join(url, imd, '{}.mli.geo.tif'.format(imd)),
                     os.path.join(mlidir, '{}.mli.geo.tif'.format(imd))
                     ) for i, imd in enumerate(imdates_dl)]

            p = q.Pool(n_para)
            p.map(download_wrapper, args)
            p.close()
        else:
            print('No MLI data available from {} to {}'.format(startdate, enddate), flush=True)

    print('', flush=True)

    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(outdir))


#%%
def download_wrapper(args):
    i, ifgd, n_dl, url_data, path_data = args
    dir_data = os.path.dirname(path_data)
    print('  Donwnloading {} ({}/{})...'.format(ifgd, i+1, n_dl), flush=True)
    if not os.path.exists(dir_data): os.mkdir(dir_data)
    tools_lib.download_data(url_data, path_data)
    return


#%%
def check_exist_wrapper(args):
    """
    Returns :
        0 : Local exist, complete, and new (no need to donwload)
        1 : Local incomplete (need to re-donwload)
        2 : Local old (no need to re-donwload)
        3 : Remote not exist  (can not compare, no download)
        4 : Local not exist (need to download)
    """

    i, n_data, url_data, path_data = args
    bname_data = os.path.basename(path_data)
    
#    if np.mod(i, 10) == 0:
#        print("  {0:3}/{1:3}".format(i, n_data), flush=True)

    if os.path.exists(path_data):
        rc = tools_lib.comp_size_time(url_data, path_data)
        if rc == 1:
            print("Size of {} is not identical.".format(bname_data), flush=True)
        elif rc == 2:
            print("Newer {} available.".format(bname_data), flush=True)
        return rc
    else:
        return 4
    

#%%
def check_gacos_wrapper(args):
    """
    Returns :
        0 : Local exist, complete, and new (no need to donwload)
        1 : Local incomplete (need to re-donwload)
        2 : Local old (no need to re-donwload)
        3 : Remote not exist  (can not compare, no download)
        4 : Local not exist and remote exist (need to download)
        5 : Local not exist but remote not exist (can not download)
    """
    i, n_data, url_data, path_data = args
    bname_data = os.path.basename(path_data)
    
    if np.mod(i, 10) == 0:
        print("  {0:3}/{1:3}".format(i, n_data), flush=True)

    if os.path.exists(path_data):
        rc = tools_lib.comp_size_time(url_data, path_data)
        if rc == 1:
            print("Size of {} is not identical.".format(bname_data), flush=True)
        elif rc == 2:
            print("Newer {} available.".format(bname_data), flush=True)
        return rc
    else:
        response = requests.head(url_data, allow_redirects=True)
        if response.status_code == 200:
            return 4
        else:
            return 5
    

#%% main
if __name__ == "__main__":
    sys.exit(main())

