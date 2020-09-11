#!/usr/bin/env python3
"""
========
Overview
========
Python3 library of time series analysis tools for LiCSBAS.

=========
Changelog
=========
v1.6 20200911 Yu Morioshita, GSI
 - STDOUT time and size in download_data
 - Add convert_size
v1.5 20200828 Yu Morioshita, GSI
 - Update for matplotlib >= 3.3
 - Use nearest interpolation for insar cmap to avoid aliasing
v1.4 20200703 Yu Morioshita, GSI
 - Replace problematic terms
 - Small bug (shift) fix in read_range_geo
v1.3 20200503 Yu Morioshita, GSI
 - Update download_data (thanks to sahitono)
v1.2 20200227 Yu Morioshita, Uni of Leeds and GSI
 - Add read_point and fit2dh
v1.1 20190916 Yu Morioshita, Uni of Leeds and GSI
 - Add read_range_line and read_range_line_geo
v1.0 20190730 Yu Morioshita, Uni of Leeds and GSI
 - Original implementation
"""
import os
import sys
import re
import time
import requests
import dateutil
import datetime as dt
import numpy as np
import statsmodels.api as sm
import warnings

#%%
def bl2xy(lon, lat, width, length, lat1, postlat, lon1, postlon):
    """
    lat1 is north edge and postlat is negative value.
    lat lon values are in grid registration
    x/y index start from 0, end with width-1
    """
    x = int(np.round((lon - lon1)/postlon))
    y = int(np.round((lat - lat1)/postlat))
    
    return [x, y]


#%%
def comp_size_time(file_remote, file_local):
    """
    Compare size and time of remote and local files.
    Returns:
        0 : Size is identical and local time is new
        1 : Size is not identical
        2 : Size is identical but remote time is new
        3 : Remote not exist
    """

    response = requests.head(file_remote, allow_redirects=True)
    
    if response.status_code != 200:
        return 3

    size_remote = int(response.headers.get("Content-Length"))
    size_local = os.path.getsize(file_local)
    if size_remote != size_local: ### Different size
        return 1

    time_remote = dateutil.parser.parse(response.headers.get("Last-Modified"))
    time_local = dt.datetime.fromtimestamp(os.path.getmtime(file_local), dt.timezone.utc)
    if time_remote > time_local: ### New file
        return 2

    return 0


#%%
def cmap_insar():
    """
    How to use cmap_insar:
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        cdict = cmap_insar()
        plt.register_cmap(cmap=mpl.colors.LinearSegmentedColormap('insar', cdict))
        plt.register_cmap(name='insar', data=cdict)
        plt.imshow(array, cmap='insar', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
        
    Note:
        - Input array should be wrapped and in radian
        - To wrap unwrapped phase, np.angle(np.exp(1j*unw/cycle)*cycle)
    """

#    These are for 0-2pi, not -pi to pi
    red = [255,255,255,255,240,203,165,127,90,55,92,130,167,205,243,255,255,255]
    green = [118,156,193,231,255,255,255,255,255,255,217,179,142,104,66,80,118,156]
    blue = [191,153,116,78,69,106,144,182,219,255,255,255,255,255,255,229,191,153]
    phase = [k/32 for k in range(1,33,2)]
    phase = [0] + phase + [1]

    red_norm = [ k/255 for k in red ] + [ red[0]/255 ]
    green_norm = [ k/255 for k in green ] + [ green[0]/255 ]
    blue_norm = [ k/255 for k in blue ] + [ blue[0]/255 ]

    redtuple=[]
    greentuple=[]
    bluetuple=[]    
    for j in range(18):
        redtuple.append((phase[j],red_norm[j],red_norm[j+1]))
        greentuple.append((phase[j],green_norm[j],green_norm[j+1]))
        bluetuple.append((phase[j],blue_norm[j],blue_norm[j+1]))
   
    redtuple=tuple(redtuple)
    greentuple=tuple(greentuple)    
    bluetuple=tuple(bluetuple)
        
    cdict = { 'red': redtuple, 'green': greentuple, 'blue': bluetuple }
    
    return cdict


#%%
def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(np.floor(np.log(size_bytes)/np.log(1024)))
   p = np.power(1024, i)
   s = round(size_bytes / p, 2)
   return "%s%s" % (s, size_name[i])


#%% 
def download_data(url, file):
    try:
       start = time.time() 
       with requests.get(url) as res:
            res.raise_for_status()
            with open(file, "wb") as output:
                output.write(res.content)
       elapsed = int(time.time()-start)
       if elapsed==0: elapsed=elapsed+1
       fsize = convert_size(os.path.getsize(file))
       speed = convert_size(int(os.path.getsize(file)/elapsed))
       print('  {}, {}, {}s, {}/s'.format(os.path.basename(file),
                                          fsize, elapsed, speed))

    except:
        print(
            "    Error while downloading from {}".format(url),
            file=sys.stderr,
            flush=True,
        )
        return



#%%
def fit2d(A,w=None,deg="1"):
    """
    Estimate best fit plain with indicated degree of polynomial.
    
    Inputs:
        A : Input ndarray (can include nan)
        w : Wieight (1/std**2) for each element of A (with the same dimention as A)
        deg : degree of polynomial of fitting plain
         - 1   -> a+bx+cy (ramp)
         - bl -> a+bx+cy+dxy (biliner)
         - 2   -> a+bx+cy+dxy+ex**2_fy**2 (2d polynomial)
     
    Returns:
        Afit : Best fit plain with the same demention as A
        m    : set of parameters of best fit plain (a,b,c...)
    
    """

    ### Make design matrix G
    length,width = A.shape #read dimension
    Xgrid,Ygrid = np.meshgrid(np.arange(width),np.arange(length)) #mesh grid
    
    if str(deg) == "1":
        G = np.stack((np.ones((length*width)), Xgrid.flatten(), Ygrid.flatten())).T
    elif str(deg) == "bl":
        G = np.stack((np.ones((length*width)), Xgrid.flatten(), Ygrid.flatten(), Xgrid.flatten()*Ygrid.flatten())).T
    elif str(deg) == "2":
        G = np.stack((np.ones((length*width)), Xgrid.flatten(), Ygrid.flatten(), Xgrid.flatten()*Ygrid.flatten(), Xgrid.flatten()**2, Ygrid.flatten()**2)).T
    else:
        print('\nERROR: Not proper deg ({}) is used\n'.format(deg), file=sys.stderr)
        return False
        
    ### Handle nan by 0 padding and 0 weight
    # Not drop in sm because cannot return Afit
    if np.any(np.isnan(A)):
        bool_nan = np.isnan(A)
        A = A.copy() # to avoid change original value in main
        A[bool_nan] = 0
        if w is None:
            w = np.ones_like(A)
        w = w.copy() # to avoid change original value in main
        w[bool_nan] = 0

    ### Invert
    if w is None: ## Ordinary LS
        results = sm.OLS(A.ravel(), G).fit()
    else: ## Weighted LS
        results = sm.WLS(A.ravel(), G, weights=w.ravel()).fit()
       
    m = results.params
    Afit = np.float32(results.predict().reshape((length,width)))

    return Afit,m


#%%
def fit2dh(A, deg, hgt, hgt_min, hgt_max):
    """
    Estimate best fit 2d ramp and topography-correlated component simultaneously.
    
    Inputs:
        A   : Input ndarray (can include nan)
        deg : degree of polynomial for fitting ramp
          - 1  -> a+bx+cy (ramp, default)
          - bl -> a+bx+cy+dxy (biliner)
          - 2  -> a+bx+cy+dxy+ex**2_fy**2 (2d polynomial)
          - []  -> a (must be used with hgt)
        hgt : Input hgt to estimate coefficient of topo-corr component
              If blank, don*t estimate topo-corr component.
        hgt_min : Minimum hgt to take into account in hgt-linear
        hgt_max : Maximum hgt to take into account in hgt-linear
    
    Returns:
        Afit : Best fit solution with the same demention as A
        m    : Set of parameters of best fit plain (a,b,c...)
    
    """

    ### Make design matrix G
    length, width = A.shape
    Xgrid, Ygrid = np.meshgrid(np.arange(width),np.arange(length))
    Xgrid1 = Xgrid.ravel()
    Ygrid1 = Ygrid.ravel()

    if not deg:
        G = np.ones((length*width))
    else:
        if str(deg) == "1":
            G = np.stack((Xgrid1, Ygrid1)).T
        elif str(deg) == "bl":
            G = np.stack((Xgrid1, Ygrid1, Xgrid1*Ygrid1)).T
        elif str(deg) == "2":
            G = np.stack((Xgrid1, Ygrid1, Xgrid1*Ygrid1, Xgrid1**2, Ygrid1**2)).T
        else:
            print('\nERROR: Not proper deg ({}) is used\n'.format(deg), file=sys.stderr)
            return False
        G = sm.add_constant(G)
    
    if len(hgt) > 0:
        _hgt = hgt.copy()  ## Not to overwrite hgt in main
        _hgt[np.isnan(_hgt)] = 0
        _hgt[_hgt<hgt_min] = 0
        _hgt[_hgt>hgt_max] = 0
        G2 = np.vstack((G.T, hgt.ravel())).T ## for Afit
        G = np.vstack((G.T, _hgt.ravel())).T
    else:
        G2 = G

    ### Invert
    results = sm.OLS(A.ravel(), G, missing='drop').fit()
    m = results.params

    Afit = np.float32((G2@m).reshape(((length,width))))

    return Afit, m


#%%
def get_ifgdates(ifgdir):
    """
    Get ifgdates and imdates in ifgdir.
    
    Returns:
        ifgdates : List of dates of ifgs 
        imdates  : List of dates of images
    """
    ifgdates = [str(k) for k in sorted(os.listdir(ifgdir))
                if len(k) == 17
                and k[0] =='2'
                and k[8] =='_'
                and k[9] =='2'
                and os.path.isdir(os.path.join(ifgdir, k))]

    return ifgdates


#%%
def get_patchrow(width, length, n_data, memory_size):
    """
    Get patch number of rows for memory size (in MB).
    
    Returns:
        n_patch : Number of patches (int)
        patchrow : List of the number of rows for each patch.
                ex) [[0, 1234], [1235, 2469],... ]
    """
    data_size = width*length*n_data*4/2**20  #in MiB, 4byte float
    n_patch = int(np.ceil(data_size/memory_size))
    ### Devide only row direction

    patchrow= []
    for i in range(n_patch):
        rowspacing = int(np.ceil(length/n_patch))
        patchrow.append([i*rowspacing,(i+1)*rowspacing])
        if i == n_patch-1:
            patchrow[-1][-1] = length

    return n_patch, patchrow



#%%
def ifgdates2imdates(ifgdates):
    primarylist = []
    secondarylist = []
    for ifgd in ifgdates:
        primarylist.append(ifgd[:8])
        secondarylist.append(ifgd[-8:])

    imdates = list(set(primarylist+secondarylist)) # set is a unique operator
    imdates.sort()

    return imdates


#%%
def multilook(array, nlook_r, nlook_c, n_valid_thre=0.5):
    """
    Nodata in input array must be filled with nan beforehand.
    if the number of valid data is less than n_valid_thre*nlook_r*nlook_c, return nan.
    """
    length, width = array.shape
    length_ml = int(np.floor(length/nlook_r))
    width_ml = int(np.floor(width/nlook_c))

    array_reshape = array[:length_ml*nlook_r,:width_ml*nlook_c].reshape(length_ml, nlook_r, width_ml, nlook_c)

    with warnings.catch_warnings(): ## To silence RuntimeWarning: Mean of empty slice
        warnings.simplefilter('ignore', RuntimeWarning)
        array_ml = np.nanmean(array_reshape, axis=(1, 3))

    n_valid = np.sum(~np.isnan(array_reshape), axis=(1, 3))
    bool_invalid = n_valid < n_valid_thre*nlook_r*nlook_c
    array_ml[bool_invalid] = np.nan
    
    return array_ml


#%%
def read_point(point_str, width, length):
    if re.match('[0-9]*/[0-9]*', point_str):
        x, y = [int(s) for s in re.split('[/]', point_str)]
        if x > width-1 or y > length-1:
            print("\nERROR:", file=sys.stderr)
            print("Index exceed input dimension ({0},{1})!".format(width,length), file=sys.stderr)
            return False
    else:
        print("\nERROR:", file=sys.stderr)
        print("Point format seems to be wrong (should be x/y)", file=sys.stderr)
        return False
    
    return [x, y]


#%%
def read_range(range_str, width, length):
    if re.match('[0-9]*:[0-9]*/[0-9]*:[0-9]', range_str):
        x1, x2, y1, y2 = [int(s) for s in re.split('[:/]', range_str)]
        if x2 == 0:
            x2 = width
        if y2 == 0:
            y2 = length
        if x1 > width-1 or x2 > width or y1 > length-1 or y2 > length:
            print("\nERROR:", file=sys.stderr)
            print("Index exceed input dimension ({0},{1})!".format(width,length), file=sys.stderr)
            return False
        if x1 >= x2 or y1 >= y2:
            print("\nERROR: x2/y2 must be larger than x1/y1", file=sys.stderr)
            return False
    else:
        print("\nERROR:", file=sys.stderr)
        print("Range format seems to be wrong (should be x1:x2/y1:y2)", file=sys.stderr)
        return False
    
    return [x1, x2, y1, y2]


#%%
def read_range_line(range_str, width, length):
    if re.match('[0-9]*,[0-9]*/[0-9]*,[0-9]', range_str):
        x1, y1, x2, y2 = [int(s) for s in re.split('[,/]', range_str)]
        if x1 > width-1 or x2 > width-1 or y1 > length-1 or y2 > length-1:
            print("\nERROR:", file=sys.stderr)
            print("Index exceed input dimension ({0},{1})!".format(width ,length), file=sys.stderr)
            return False
    else:
        print("\nERROR:", file=sys.stderr)
        print("Range format seems to be wrong (must be x1,y1/x2,y2)", file=sys.stderr)
        return False
    
    return [x1, x2, y1, y2]



#%%
def read_range_geo(range_str, width, length, lat1, postlat, lon1, postlon):
    """
    lat1 is north edge and postlat is negative value.
    lat lon values are in grid registration
    Note: x1/y1 range 0 to width-1, while x2/y2 range 1 to width
    """
    lat2 = lat1+postlat*(length-1)
    lon2 = lon1+postlon*(width-1)
    
    if re.match('[+-]?\d+(?:\.\d+)?/[+-]?\d+(?:\.\d+)?/[+-]?\d+(?:\.\d+)?/[+-]?\d+(?:\.\d+)?', range_str):
        lon_w, lon_e, lat_s, lat_n = [float(s) for s in range_str.split('/')]
        x1 = int(np.round((lon_w - lon1)/postlon)) if lon_w > lon1 else 0
        x2 = int(np.round((lon_e - lon1)/postlon))+1 if lon_e < lon2 else width
        y1 = int(np.round((lat_n - lat1)/postlat)) if lat_n < lat1 else 0
        y2 = int(np.round((lat_s - lat1)/postlat))+1 if lat_s > lat2 else length
    else:
        print("\nERROR:", file=sys.stderr)
        print("Range format seems to be wrong (should be lon1/lon2/lat1/lat2)", file=sys.stderr)
        return False
    
    return [x1, x2, y1, y2]


#%%
def read_range_line_geo(range_str, width, length, lat_n, postlat, lon_w, postlon):
    """
    lat_n is north edge and postlat is negative value.
    lat lon values are in grid registration
    """

    if re.match('[+-]?\d+(?:\.\d+)?,[+-]?\d+(?:\.\d+)?/[+-]?\d+(?:\.\d+)?,[+-]?\d+(?:\.\d+)?', range_str):
        lon1, lat1, lon2, lat2 = [float(s) for s in re.split('[,/]', range_str)]
        x1 = int(np.round((lon1 - lon_w)/postlon))
        x2 = int(np.round((lon2 - lon_w)/postlon))
        y1 = int(np.round((lat1 - lat_n)/postlat))
        y2 = int(np.round((lat2 - lat_n)/postlat))
    else:
        print("\nERROR:", file=sys.stderr)
        print("Range format seems to be wrong (should be lon1,lat1/lon2,lat2)", file=sys.stderr)
        return False
    
    return [x1, x2, y1, y2]


#%%
def xy2bl(x, y, lat1, dlat, lon1, dlon):
    """
    xy index starts from 0, end with width/length-1
    """
    lat = lat1+dlat*y
    lon = lon1+dlon*x
    
    return lat, lon
