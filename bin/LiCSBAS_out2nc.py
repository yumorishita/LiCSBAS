#!/usr/bin/env python3
"""
v1.0 20200901 Milan Lazecky, Leeds Uni

========
Overview
========
This script outputs a standard NetCDF4 file using LiCSBAS results


=====
Usage
=====
LiCSBAS_out2nc.py [-i infile] [-o outfile] [-m yyyymmdd]
     [--ref_geo lon1/lon2/lat1/lat2] [--clip_geo lon1/lon2/lat1/lat2]

 -i  Path to input cum file (Default: cum_filt.h5)
 -o  Output netCDF4 file (Default: output.nc)
 -m  Master (reference) date (Default: first date)
 --ref_geo  Reference area in geographical coordinates as: lon1/lon2/lat1/lat2
 --clip_geo  Area to clip in geographical coordinates as: lon1/lon2/lat1/lat2
 --compress, -C  use zlib compression (very small files but time series may take long to load in GIS)
 
 TODO:
 --mask  Path to mask file for ref phase calculation (Default: No mask)

"""
#%% Change log
'''
v1.0 20200901 Milan Lazecky, Uni of Leeds
 - Original implementation
'''

#%% Import
import getopt
import os
import sys
import time
import numpy as np
import datetime as dt
import xarray as xr
import rioxarray
import subprocess as subp
from scipy.ndimage import gaussian_filter
from scipy import interpolate

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


#just an eye candy layer
def interp_and_smooth(da, sigma=0.8):
    dar = da.copy()
    array = np.ma.masked_invalid(dar.values)
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    xx, yy = np.meshgrid(x, y)
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    GD1 = interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='linear')
    #, fill_value=0)
    GD1 = np.array(GD1)
    GD1 = gaussian_filter(GD1, sigma=sigma)
    dar.values = GD1
    return dar


def loadall2cube(cumfile):
    cumdir = os.path.dirname(cumfile)
    cohfile = os.path.join(cumdir,'results/coh_avg')
    rmsfile = os.path.join(cumdir,'results/resid_rms')
    vstdfile = os.path.join(cumdir,'results/vstd')
    stcfile = os.path.join(cumdir,'results/stc')
    metafile = os.path.join(cumdir,'../../metadata.txt')
    #h5datafile = 'cum.h5'
    cum = xr.load_dataset(cumfile)
    
    sizex = len(cum.vel[0])
    sizey = len(cum.vel)
    
    lon = cum.corner_lon.values+cum.post_lon.values*np.arange(sizex)
    lat = cum.corner_lat.values+cum.post_lat.values*np.arange(sizey)
    
    time = np.array(([dt.datetime.strptime(str(imd), '%Y%m%d') for imd in cum.imdates.values]))
    
    velxr = xr.DataArray(cum.vel.values.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
    #LiCSBAS uses 0 instead of nans...
    velxr = velxr.where(velxr!=0)
    velxr.attrs['unit'] = 'mm/year'
    #vinterceptxr = xr.DataArray(cum.vintercept.values.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
    
    cumxr = xr.DataArray(cum.cum.values, coords=[time, lat, lon], dims=["time","lat", "lon"])
    cumxr.attrs['unit'] = 'mm'
    #bperpxr = xr.DataArray(cum.bperp.values, coords=[time], dims=["time"])
    
    cube = xr.Dataset()
    cube['cum'] = cumxr
    cube['vel'] = velxr
    #cube['vintercept'] = vinterceptxr
    
    #add coh_avg resid_rms vstd
    if os.path.exists(cohfile):
        infile = np.fromfile(cohfile, 'float32')
        cohxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        cube['coh'] = cohxr
    else: print('No coh_avg file detected, skipping')
    if os.path.exists(rmsfile):
        infile = np.fromfile(rmsfile, 'float32')
        rmsxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        rmsxr.attrs['unit'] = 'mm'
        cube['rms'] = rmsxr
    else: print('No RMS file detected, skipping')
    if os.path.exists(vstdfile):
        infile = np.fromfile(vstdfile, 'float32')
        vstdxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        vstdxr.attrs['unit'] = 'mm/year'
        cube['vstd'] = vstdxr
    else: print('No vstd file detected, skipping')
    if os.path.exists(stcfile):
        infile = np.fromfile(stcfile, 'float32')
        stcxr = xr.DataArray(infile.reshape(sizey,sizex), coords=[lat, lon], dims=["lat", "lon"])
        #vstdxr.attrs['unit'] = 'mm/year'
        cube['stc'] = stcxr
    else: print('No stc file detected, skipping')
    # add inc_angle
    if os.path.exists(metafile):
        a = subp.run(['grep','inc_angle', metafile], stdout=subp.PIPE)
        inc_angle = float(a.stdout.decode('utf-8').split('=')[1])
        cube.attrs['inc_angle'] = inc_angle
    else: print('')#'warning, metadata file not found. using general inc angle value')
        #inc_angle = 39
    
    #cube['bperp'] = bperpxr
    cube.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    cube.rio.write_crs("EPSG:4326", inplace=True)
    cube = cube.sortby(['time','lon','lat'])
    return cube

#not in use now
def maskit(clipped, cohthres = 0.62, rmsthres = 5, vstdthres = 0.3):
    da = clipped.copy()
    out = da.where(clipped.coh>=cohthres) \
    .where(np.abs(clipped.rms)<=rmsthres) \
    .where(np.abs(clipped.vstd)<=vstdthres)
    return out


#%% Main
def main(argv=None):
   
    #%% Check argv
    if argv == None:
        argv = sys.argv
        
    start = time.time()
    ver=1.0; date=20200904; author="M.Lazecky"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    cumfile = 'cum.h5'
    outfile = 'output.nc'
    imd_m = []
    #refarea = []
    refarea_geo = []
    maskfile = []
    cliparea_geo = []
    compress = False
    
    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:m:r:c:C", ["help", "compress","clip_geo=", "ref_geo=", "mask="])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                cumfile = a
            elif o == '-o':
                outfile = a
            elif o == '-m':
                imd_m = a
            elif (o == '-C') or (o=='compress'):
                compress = True
                print('will use zlib compression')
            elif o == '-r':
                refarea = a
                print('ref area in radar coords not implemented yet')
            elif o == '--clip_geo':
                cliparea_geo = a
                minclipx, maxclipx, minclipy, maxclipy = cliparea_geo.split('/')
                minclipx, maxclipx, minclipy, maxclipy = float(minclipx), float(maxclipx), float(minclipy), float(maxclipy)
            elif o == '--ref_geo':
                refarea_geo = a
                minrefx, maxrefx, minrefy, maxrefy = refarea_geo.split('/')
                minrefx, maxrefx, minrefy, maxrefy = float(minrefx), float(maxrefx), float(minrefy), float(maxrefy)
            elif o == '--mask':
                maskfile = a

        if not os.path.exists(cumfile):
            raise Usage('No {} exists! Use -i option.'.format(cumfile))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    cube = loadall2cube(cumfile)
    
    
    #reference cum to time (first date will be 0)
    if not imd_m:
        imd_m = cube.time.isel(time=0).astype('str')
    cube['cum'] = cube['cum'] - cube['cum'].sel(time=imd_m)
    
    #reference it
    if refarea_geo:
        #ref = cube.rio.clip_box(minrefx, minrefy, maxrefx, maxrefy)
        ref = cube.sel(lon=slice(minrefx, maxrefx), lat=slice(minrefy, maxrefy))
        if len(ref.vel) == 0:
            print('warning, no points in the reference area - will export without referencing')
        else:
            refcoh = ref.where(ref.coh >0.6)
            if refcoh.vel.count() < 2:
                print('warning, the ref area has low coherence! continuing anyway')
                refcoh = ref
            #for v in refcoh.data_vars.variables:
            #for v in ['cum', 'vel', 'vel_filt']:
            for v in ['cum', 'vel']:
                cube[v] = cube[v] - refcoh[v].median(["lat", "lon"])
    
    #only now will clip - this way the reference area can be outside the clip, if needed
    if cliparea_geo:
        cube = cube.sel(lon=slice(minclipx, maxclipx), lat=slice(minclipy, maxclipy))
    
    #do filtered (it is nice)
    cube['vel_filt'] = interp_and_smooth(cube['vel'], 0.5)
    
    #masked = maskit(clipped)
    #masked['vel_filt'] = clipped['vel_filt']
    

    #masked.to_netcdf(outfile)


    #just to make sure it is written..
    #check if it does not invert data!
    cube.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    cube.rio.write_crs("EPSG:4326", inplace=True)
    if compress:
        encode = {'cum': {'zlib': True, 'complevel': 9}, 'vel': {'zlib': True, 'complevel': 9}, 
        'coh': {'zlib': True, 'complevel': 9}, 'rms': {'zlib': True, 'complevel': 9}, 
        'stc': {'zlib': True, 'complevel': 9}, 'vel_filt': {'zlib': True, 'complevel': 9}, 
        'time': {'dtype': 'i4'}}
        cube.to_netcdf(outfile, encoding=encode)
    else:
        cube.to_netcdf(outfile, encoding={'time': {'dtype': 'i4'}})
    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output: {}\n'.format(outfile), flush=True)


#%% main
if __name__ == "__main__":
    sys.exit(main())
