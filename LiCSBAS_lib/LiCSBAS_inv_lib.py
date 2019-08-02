#!/usr/bin/env python3
"""
========
Overview
========
Python3 library of time series inversion functions for LiCSBAS.

=========
Changelog
=========
v1.0 20190730 Yu Morioshita, Uni of Leeds and GSI
 - Original implementation
"""

import warnings
import numpy as np
import multiprocessing as multi
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext
import LiCSBAS_tools_lib as tools_lib


#%%
def make_sb_matrix(ifgdates):
    """
    Make small baseline incidence-like matrix.
    Composed of 1 between master and slave. (n_ifg, n_im-1)
    Unknown is incremental displacement.
    """
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    n_im = len(imdates)
    n_ifg = len(ifgdates)

    G = np.zeros((n_ifg, n_im-1), dtype=np.int16)
    for ifgix, ifgd in enumerate(ifgdates):
        masterdate = ifgd[:8]
        masterix = imdates.index(masterdate)
        slavedate = ifgd[-8:]
        slaveix = imdates.index(slavedate)
        G[ifgix, masterix:slaveix] = 1

    return G


#%%
def make_sb_matrix2(ifgdates):
    """
    Make small baseline incidence-like matrix.
    Composed of -1 at master and 1 at slave. (n_ifg, n_im)
    Unknown is cumulative displacement.
    """
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    n_im = len(imdates)
    n_ifg = len(ifgdates)

    A = np.zeros((n_ifg, n_im), dtype=np.int16)
    for ifgix, ifgd in enumerate(ifgdates):
        masterdate = ifgd[:8]
        masterix = imdates.index(masterdate)
        slavedate = ifgd[-8:]
        slaveix = imdates.index(slavedate)
        A[ifgix, masterix] = -1
        A[ifgix, slaveix] = 1
    return A



#%%
def invert_nsbas(unw, G, dt_cum, gamma, n_core):
    """
    Calculate increment displacement difference by NSBAS inversion. Points with all unw data are solved by simple SB inversion firstly at a time.

    Inputs:
      unw : Unwrapped data block for each point (n_pt, n_ifg)
            Still include nan to keep dimention
      G    : Design matrix (1 between master and slave) (n_ifg, n_im-1)
      dt_cum : Cumulative years(or days) for each image (n_im)
      gamma  : Gamma value for NSBAS inversion, should be small enough (e.g., 0.0001)
      n_core : Number of cores for parallel processing

    Returns:
      inc     : Incremental displacement (n_im-1, n_pt)
      vel     : Velocity (n_pt)
      vconst  : Constant part of linear velocity (c of vt+c) (n_pt)
    """
    if n_core != 1:
        global Gall, unw_tmp, mask ## for para_wrapper

    ### Settings
    n_pt, n_ifg = unw.shape
    n_im = G.shape[1]+1

    result = np.zeros((n_im+1, n_pt), dtype=np.float32)*np.nan #[inc, vel, const]

    ### Set matrix of NSBAS part (bottom)
    Gbl = np.tril(np.ones((n_im, n_im-1), dtype=np.float32), k=-1) #lower tri matrix without diag
    Gbr = -np.ones((n_im, 2), dtype=np.float32)
    Gbr[:, 0] = -dt_cum
    Gb = np.concatenate((Gbl, Gbr), axis=1)*gamma
    Gt = np.concatenate((G, np.zeros((n_ifg, 2), dtype=np.float32)), axis=1)
    Gall = np.float32(np.concatenate((Gt, Gb)))

    ### Solve points with full unw data at a time. Very fast.
    bool_pt_full = np.all(~np.isnan(unw), axis=1)
    n_pt_full = bool_pt_full.sum()

    if n_pt_full!=0:
        print('  Solving {0:6}/{1:6}th points with full unw at a time...'.format(n_pt_full, n_pt), flush=True)
    
        ## Sovle
        unw_tmp = np.concatenate((unw[bool_pt_full, :], np.zeros((n_pt_full, n_im), dtype=np.float32)), axis=1).transpose()
        result[:, bool_pt_full] = np.linalg.lstsq(Gall, unw_tmp, rcond=None)[0]


    ### Solve other points with nan point by point.
    unw_tmp = np.concatenate((unw[~bool_pt_full, :], np.zeros((n_pt-n_pt_full, n_im), dtype=np.float32)), axis=1).transpose()
    mask = (~np.isnan(unw_tmp))
    unw_tmp[np.isnan(unw_tmp)] = 0
    print('  Next, solve {0} points including nan point-by-point...'.format(n_pt-n_pt_full), flush=True)
    
    if n_core == 1:
        result[:, ~bool_pt_full] = censored_lstsq_slow(Gall, unw_tmp, mask) #(n_im+1, n_pt) 
    else:
        print('  {} parallel processing'.format(n_core), flush=True)

        args = [i for i in range(n_pt-n_pt_full)]
        p = multi.Pool(n_core)
        _result = p.map(censored_lstsq_slow_para_wrapper, args) #list[n_pt][length]
        result[:, ~bool_pt_full] = np.array(_result).T

    inc = result[:n_im-1, :]
    vel = result[n_im-1, :]
    vconst = result[n_im, :]

    return inc, vel, vconst


def censored_lstsq_slow_para_wrapper(i):
    ### Use global value
    if np.mod(i, 1000) == 0:
        print('  Running {0:6}/{1:6}th point...'.format(i, unw_tmp.shape[1]), flush=True)
    m = mask[:,i] # drop rows where mask is zero
    try:
        X = np.linalg.lstsq(Gall[m], unw_tmp[m,i], rcond=None)[0]
    except:
        X = np.zeros((Gall.shape[1]), dtype=np.float32)*np.nan
    return X


#%%
def invert_nsbas_wls(unw, var, G, dt_cum, gamma, n_core):
    """
    Calculate increment displacement difference by NSBAS inversion with WLS.

    Inputs:
      unw : Unwrapped data block for each point (n_pt, n_ifg)
            Still include nan to keep dimention
      var : Variance estimated from coherence (n_pt, n_ifg)
      G    : Design matrix (1 between master and slave) (n_ifg, n_im-1)
      dt_cum : Cumulative years(or days) for each image (n_im)
      gamma  : Gamma value for NSBAS inversion, should be small enough (e.g., 0.0001)
      n_core : Number of cores for parallel processing

    Returns:
      inc     : Incremental displacement (n_im-1, n_pt)
      vel     : Velocity (n_pt)
      vconst  : Constant part of linear velocity (c of vt+c) (n_pt)
    """
    global Gall, unw_tmp, var_tmp, mask ## for para_wrapper

    ### Settings
    n_pt, n_ifg = unw.shape
    n_im = G.shape[1]+1

    result = np.zeros((n_im+1, n_pt), dtype=np.float32)*np.nan #[inc, vel, const]

    ### Set matrix of NSBAS part (bottom)
    Gbl = np.tril(np.ones((n_im, n_im-1), dtype=np.float32), k=-1) #lower tri matrix without diag
    Gbr = -np.ones((n_im, 2), dtype=np.float32)
    Gbr[:, 0] = -dt_cum
    Gb = np.concatenate((Gbl, Gbr), axis=1)*gamma
    Gt = np.concatenate((G, np.zeros((n_ifg, 2), dtype=np.float32)), axis=1)
    Gall = np.float32(np.concatenate((Gt, Gb)))


    ### Make unw_tmp, var_tmp, and mask
    unw_tmp = np.concatenate((unw, np.zeros((n_pt, n_im), dtype=np.float32)), axis=1).transpose()
    mask = (~np.isnan(unw_tmp))
    unw_tmp[np.isnan(unw_tmp)] = 0
    var_tmp = np.concatenate((var, 50*np.ones((n_pt, n_im), dtype=np.float32)), axis=1).transpose() #50 is var for coh=0.1, to scale bottom part of Gall

    if n_core == 1:
        for i in range(n_pt):
            result[:, i] = wls_nsbas(i) #(n_im+1, n_pt) 
    else:
        print('  {} parallel processing'.format(n_core), flush=True)

        args = [i for i in range(n_pt)]
        p = multi.Pool(n_core)
        _result = p.map(wls_nsbas, args) #list[n_pt][length]
        result = np.array(_result).T

    inc = result[:n_im-1, :]
    vel = result[n_im-1, :]
    vconst = result[n_im, :]

    return inc, vel, vconst


def wls_nsbas(i):
    ### Use global value of Gall, unw_tmp, mask
    if np.mod(i, 1000) == 0:
        print('  Running {0:6}/{1:6}th point...'.format(i, unw_tmp.shape[1]), flush=True)
        
    ## Weight unw and G

    Gall_w = Gall/np.sqrt(np.float64(var_tmp[:,i][:,np.newaxis]))
    unw_tmp_w = unw_tmp[:, i]/np.sqrt(np.float64(var_tmp[:,i]))
    m = mask[:,i] # drop rows where mask is zero

    try:
        X = np.linalg.lstsq(Gall_w[m], unw_tmp_w[m], rcond=None)[0]
    except:
        X = np.zeros((Gall.shape[1]), dtype=np.float32)*np.nan
    return X


#%%
def calc_vel(cum, dt_cum):
    """
    Calculate velocity.

    Inputs:
      cum    : cumulative phase block for each point (n_pt, n_im)
      dt_cum : Cumulative days for each image (n_im)

    Returns:
      vel    : Velocity (n_pt)
      vconst : Constant part of linear velocity (c of vt+c) (n_pt)
    """
    n_pt, n_im = cum.shape
    result = np.zeros((2, n_pt), dtype=np.float32)*np.nan #[vconst, vel]
   
    G = np.stack((np.ones_like(dt_cum), dt_cum), axis=1)
    vconst = np.zeros((n_pt), dtype=np.float32)*np.nan
    vel = np.zeros((n_pt), dtype=np.float32)*np.nan

    bool_pt_full = np.all(~np.isnan(cum), axis=1)
    n_pt_full = bool_pt_full.sum()

    if n_pt_full!=0:
        print('  Solving {0:6}/{1:6}th points with full cum at a time...'.format(n_pt_full, n_pt), flush=True)
    
        ## Sovle
        result[:, bool_pt_full] = np.linalg.lstsq(G, cum[bool_pt_full, :].transpose(), rcond=None)[0]

    ### Solve other points with nan point by point.
    cum_tmp = cum[~bool_pt_full, :].transpose()
    mask = (~np.isnan(cum_tmp))
    cum_tmp[np.isnan(cum_tmp)] = 0
    print('  Next, solve {0} points including nan point-by-point...'.format(n_pt-n_pt_full), flush=True)
    
    result[:, ~bool_pt_full] = censored_lstsq_slow(G, cum_tmp, mask) #(n_im+1, n_pt) 
        
    vconst = result[0, :]
    vel = result[1, :]

    return vel, vconst


#%%
def calc_velstd(cum, dt_cum):
    """
    Not use 20190509
    Calculate std of velocity by bootstrap. NaN is not allowed.

    Inputs:
      cum    : Cumulative phase block for each point (n_pt, n_im)
      dt_cum : Cumulative days for each image (n_im)

    Returns:
      vstd   : Std of Velocity for each point (n_pt)
    """
    n_pt, n_im = cum.shape
    bootnum = 100
    n_batch = 1000 # number of processed points at a time

    vstd = np.zeros((n_pt), dtype=np.float32)

    G = np.stack((np.ones_like(dt_cum), dt_cum), axis=1)
    velinv = lambda x : np.linalg.lstsq(G, x, rcond=None)[0][1]
#    velinv = lambda x : sm.OLS(x, G).fit().params[1]
    ## Much much faster than processing each point by sm

    for i in range(int(np.ceil(n_pt/n_batch))):
        ix1 = i*n_batch
        ix2 = (i+1)*n_batch # dont need -1!
        if ix2 > n_pt : ix2 = n_pt
        if np.mod(i, 10) == 0:
            print('\r  Finished {0:6}/{1:6}th point...'.format(ix1, n_pt), end='', flush=True)

        with NumpyRNGContext(1):
            bootresult = bootstrap(cum[ix1:ix2, :].transpose(), bootnum, bootfunc=velinv)

        vstd[ix1:ix2] = np.std(bootresult, axis=0)

    print('')

    return vstd


#%%
def calc_velstd_withnan(cum, dt_cum):
    """
    Calculate std of velocity by bootstrap for each point which may include nan.

    Inputs:
      cum    : Cumulative phase block for each point (n_pt, n_im)
               Can include nan.
      dt_cum : Cumulative days for each image (n_im)

    Returns:
      vstd   : Std of Velocity for each point (n_pt)
    """
    global bootcount, bootnum
    n_pt, n_im = cum.shape
    bootnum = 100
    bootcount = 0 

    vstd = np.zeros((n_pt), dtype=np.float32)
    G = np.stack((np.ones_like(dt_cum), dt_cum), axis=1)
    
    fill_value = -9999 # not 0 because exact 0 already exist in the first image
    data = cum.transpose().copy()
    data[np.isnan(data)] = fill_value # remove nan for censored_lstsq
            
    velinv = lambda x : censored_lstsq2(G, x, fill_value)[1]

    with NumpyRNGContext(1):
        bootresult = bootstrap(data, bootnum, bootfunc=velinv)
        
    vstd = np.std(bootresult, axis=0)

    print('')

    return vstd


def censored_lstsq2(A, B, fill_value):
    ## http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
    global bootcount, bootnum
    print('\r  Running {0:3}/{1:3}th bootstrap...'.format(bootcount, bootnum), end='', flush=True)
    bootcount = bootcount+1

    M = (B != fill_value) # False (0) at nodata

    # if B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return np.linalg.leastsq(A[M], B[M])[0]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    return np.squeeze(np.linalg.solve(T, rhs)).T # transpose to get r x n

    
#%%
def calc_stc(cum):
    """
    Calculate STC (spatio-temporal consistensy; Hanssen et al., 2008, Terrafirma) of time series of displacement.
    Note that isolated pixels (which have no surrounding pixel) have nan of STC.

    Input:
      cum  : Cumulative displacement (n_im, length, width)

    Return:
      stc  : STC (length, width)
    """
    n_im, length, width = cum.shape

    ### Add 1 pixel margin to cum data filled with nan
    cum1 = np.ones((n_im, length+2, width+2), dtype=np.float32)*np.nan
    cum1[:, 1:length+1, 1:width+1] = cum

    ### Calc STC for surrounding 8 pixels
    _stc = np.ones((length, width, 8), dtype=np.float32)*np.nan
    pixels = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
    ## Left Top = [0, 0], Rigth Bottmon = [2, 2], Center = [1, 1]
    
    for i, pixel in enumerate(pixels):
        ### Spatial difference (surrounding pixel-center)
        d_cum = cum1[:, pixel[0]:length+pixel[0], pixel[1]:width+pixel[1]] - cum1[:, 1:length+1, 1:width+1]
        
        ### Temporal difference (double difference)
        dd_cum = d_cum[:-1,:,:]-d_cum[1:,:,:]
        
        ### STC (i.e., RMS of DD)
        sumsq_dd_cum = np.nansum(dd_cum**2, axis=0)
        n_dd_cum = np.float32(np.sum(~np.isnan(dd_cum), axis=0)) #nof non-nan
        n_dd_cum[n_dd_cum==0] = np.nan #to avoid 0 division
        _stc[:, :, i] = np.sqrt(sumsq_dd_cum/n_dd_cum)

    ### Identify minimum value as final STC
    with warnings.catch_warnings(): ## To silence warning by All-Nan slice
        warnings.simplefilter('ignore', RuntimeWarning)
        stc = np.nanmin(_stc, axis=2)

    return stc


#%%
def censored_lstsq(A, B, M):
    ## http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
    ## This is actually slow because matmul does not use multicore...
    ## Need multiprocessing.
    ## Precison is bad widh bad condition, so this is unfortunately useless for NSABS...
    ## But maybe usable for vstd because its condition is good.
    """Solves least squares problem subject to missing data.

    Note: uses a broadcasted solve for speed.

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    # Note: we should check A is full rank but we won't bother...

    # if B is a vector, simply drop out corresponding rows in A
    if B.ndim == 1 or B.shape[1] == 1:
        return np.linalg.leastsq(A[M], B[M])[0]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    return np.squeeze(np.linalg.solve(T, rhs)).T # transpose to get r x n


#%%
def censored_lstsq_slow(A, B, M):
    ## http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
    """Solves least squares problem subject to missing data.

    Note: uses a for loop over the columns of B, leading to a
    slower but more numerically stable algorithm

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    X = np.empty((A.shape[1], B.shape[1]))
    for i in range(B.shape[1]):
        if np.mod(i, 1000) == 0:
             print('\r  Running {0:6}/{1:6}th point...'.format(i, B.shape[1]), end='', flush=True)

        m = M[:,i] # drop rows where mask is zero
        try:
            X[:,i] = np.linalg.lstsq(A[m], B[m,i], rcond=None)[0]
        except:
            X[:,i] = np.nan
    
    print('')
    return X
