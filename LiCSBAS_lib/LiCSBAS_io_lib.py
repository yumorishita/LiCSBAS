#!/usr/bin/env python3
"""
========
Overview
========
Python3 library of input/output functions for LiCSBAS.

=========
Changelog
=========
v1.1 20200227 Yu Morioshita, Uni of Leeds and GSI
 - Add hgt_linear_flag to make_tstxt
v1.0 20190730 Yu Morioshita, Uni of Leeds and GSI
 - Original implementation

"""
import sys
import numpy as np
import subprocess as subp
import datetime as dt
import statsmodels.api as sm


#%%
def make_dummy_bperp(bperp_file, imdates):
    with open(bperp_file, 'w') as f:
        for i, imd in enumerate(imdates):
            if i==0: bp = 0
            elif np.mod(i, 4)==1: bp = np.random.rand()/2+0.5 #0.5~1
            elif np.mod(i, 4)==2: bp = -np.random.rand()/2-0.5 #-1~-0.5
            elif np.mod(i, 4)==3: bp = np.random.rand()/2 #0~0.5
            elif np.mod(i, 4)==0: bp = -np.random.rand()/2 #-0.5~0

            ifg_dt = dt.datetime.strptime(imd, '%Y%m%d').toordinal() - dt.datetime.strptime(imdates[0], '%Y%m%d').toordinal()
            
            print('{:3d} {} {} {:5.2f} {:4d} {} {:4d} {} {:5.2f}'.format(i, imdates[0], imd, bp, ifg_dt, 0, ifg_dt, 0, bp), file=f)


#%%
def make_tstxt(x, y, imdates, ts, tsfile, refx1, refx2, refy1, refy2, gap, lat=None, lon=None, reflat1=None, reflat2=None, reflon1=None, reflon2=None, deramp_flag=None, hgt_linear_flag=None, filtwidth_km=None, filtwidth_yr=None):
    """
    Make txt of time series.
    Format example:
    # x, y    : 432, 532
    # lat, lon: 34.65466, 136.65432
    # ref     : 21:22/54:55
    # refgeo  : 136.98767/136.98767/34.95364/34.95364
    # deramp, filtwidth_km, filtwidth_yr: 1, 2, 0.653
    # hgt_linear_flag: 1
    # gap     : 20160104_20160116, 20170204_20170216
    # linear model: -3.643*t+4.254
    20141030    0.00
    20150216   -3.50
    20160716   -3.5
    """
    ### Calc model
    imdates_ordinal = np.array(([dt.datetime.strptime(imd, '%Y%m%d').toordinal() for imd in imdates])) ##73????
    imdates_yr = (imdates_ordinal-imdates_ordinal[0])/365.25
    A = sm.add_constant(imdates_yr) #[1, t]
    vconst, vel = sm.OLS(ts, A, missing='drop').fit().params
    
    ### Identify gaps
    ixs_gap = np.where(gap==1)[0] # n_im-1, bool
    gap_str = ''
    for ix_gap in ixs_gap:
        gap_str = gap_str+imdates[ix_gap]+'_'+imdates[ix_gap+1]+' '

    ### Output
    with open(tsfile, 'w') as f:
        print('# x, y    : {}, {}'.format(x, y), file=f)
        if all(v is not None for v in [lat, lon]):
            print('# lat, lon: {:.5f}, {:.5f}'.format(lat, lon), file=f)
        print('# ref     : {}:{}/{}:{}'.format(refx1, refx2, refy1, refy2), file=f)
        if all(v is not None for v in [reflon1, reflon2, reflat1, reflat2]):
            print('# refgeo  : {:.5f}/{:.5f}/{:.5f}/{:.5f}'.format(reflon1, reflon2, reflat1, reflat2), file=f)
        if filtwidth_yr is not None:
            print('# deramp, filtwidth_km, filtwidth_yr : {}, {}, {:.3f}'.format(deramp_flag, filtwidth_km, filtwidth_yr), file=f)
        if hgt_linear_flag is not None:
            print('# hgt_linear_flag : {}'.format(hgt_linear_flag), file=f)
        print('# gap     : {}'.format(gap_str), file=f)
        print('# linear model: {:.3f}*t{:+.3f}'.format(vel, vconst), file=f)

        for i, imd in enumerate(imdates):
            print('{} {:7.2f}'.format(imd, ts[i]), file=f)


#%%
def read_bperp_file(bperp_file, imdates):
    """
    bperp_file (baselines) contains (m:master, s:slave, sm: single master):
          smdate    sdate    bp    dt
        20170302 20170326 130.9  24.0
        20170302 20170314  32.4  12.0

    Old bperp_file contains (m:master, s:slave, sm: single master):
        num    mdate    sdate   bp   dt  dt_m_sm dt_s_sm bp_m_sm bp_s_sm 
          1 20170218 20170326 96.6 36.0    -12.0    24.0    34.2   130.9
          2 20170302 20170314 32.4 12.0      0.0    12.0     0.0    32.4

    Return: bperp
    """
    bperp = []
    bperp_dict = {}
    
    ### Determine type of bperp_file; old or not
    with open(bperp_file) as f:
        line = f.readline().split() #list

    if len(line) == 4: ## new format
        bperp_dict[line[0]] = '0.00' ## single master. unnecessary?
        with open(bperp_file) as f:
            for l in f:
                bperp_dict[l.split()[1]] = l.split()[2]
        
    else: ## old format
        with open(bperp_file) as f:
            for l in f:
                bperp_dict[l.split()[1]] = l.split()[-2]
                bperp_dict[l.split()[2]] = l.split()[-1]
            
    for imd in imdates:
        if imd in bperp_dict:
            bperp.append(float(bperp_dict[imd]))
        else: ## If no key exists
            print('ERROR: bperp for {} not found!'.format(imd), file=sys.stderr)
            return False
    
    return bperp


#%%
def read_img(file, length, width, dtype=np.float32, endian='little'):
    """
    Read image data into numpy array.
    endian: 'little' or 'big' (not 'little' is regarded as 'big')
    """
    
    if endian == 'little':
        data = np.fromfile(file, dtype=dtype).reshape((length, width))
    else:
        data = np.fromfile(file, dtype=dtype).byteswap().reshape((length, width))
    
    return data


#%%
def read_ifg_list(ifg_listfile):
    ifgdates = []
    f = open(ifg_listfile)
    line = f.readline()
    while line:
        ifgd = line.split()[0]
        if ifgd == "#":
            line = f.readline()
            continue # Comment
        else:
            ifgdates.append(ifgd)
            line = f.readline()
    f.close()
    
    return ifgdates


#%%
def get_param_par(mlipar, field):
    """
    Get parameter from mli.par or dem_par file. Examples of fields are;
     - range_samples
     - azimuth_lines
     - range_looks
     - azimuth_looks
     - range_pixel_spacing (m)
     - azimuth_pixel_spacing (m)
     - radar_frequency  (Hz)
    """
    value = subp.check_output(['grep', field,mlipar]).decode().split()[1].strip()
    return value

