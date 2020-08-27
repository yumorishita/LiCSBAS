#!/usr/bin/env python3
"""
========
Overview
========
Python3 library of plot functions for LiCSBAS.

=========
Changelog
=========
v1.2 20200827 Yu Morioshita, GSI
 - Bug fix in plot_network; use datetime instead of ordinal
 - Update for matplotlib >= 3.3
v1.1 20200228 Yu Morioshita, Uni of Leeds and GSI
 - Remove pdf option in plot_network
 - Add plot_hgt_corr
 - Add plot_gacos_info
v1.0 20190729 Yu Morioshita, Uni of Leeds and GSI
 - Original implementation

"""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np
import datetime as dt

import warnings
import matplotlib as mpl
with warnings.catch_warnings(): ## To silence user warning
    warnings.simplefilter('ignore', UserWarning)
    mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import dates as mdates

import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_inv_lib as inv_lib


#%%
def make_im_png(data, pngfile, cmap, title, vmin=None, vmax=None, cbar=True):
    """
    Make png image.
    cmap can be 'insar'. To wrap data, np.angle(np.exp(1j*x/cycle)*cycle)
    """

    if cmap=='insar':
        cdict = tools_lib.cmap_insar()
        plt.register_cmap(cmap=mpl.colors.LinearSegmentedColormap('insar', cdict))
    
    length, width = data.shape
    figsizex = 8
    xmergin = 2 if cbar else 0
    figsizey = int((figsizex-xmergin)*(length/width))+1
    
    ### Plot
    fig, ax = plt.subplots(1, 1, figsize=(figsizex, figsizey))
    plt.tight_layout()
    
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
    if cbar: fig.colorbar(im)

    plt.savefig(pngfile)
    plt.close()
    
    return


#%%
def make_3im_png(data3, pngfile, cmap, title3, vmin=None, vmax=None, cbar=True):
    """
    Make png with 3 images for comparison.
    data3 and title3 must be list with 3 elements.
    cmap can be 'insar'. To wrap data, np.angle(np.exp(1j*x/cycle)*cycle)
    """
    ### Plot setting
    if cmap=='insar':
        cdict = tools_lib.cmap_insar()
        plt.register_cmap(cmap=mpl.colors.LinearSegmentedColormap('insar', cdict))

    length, width = data3[0].shape
    figsizex = 12
    xmergin = 4 if cbar else 0
    figsizey = int((figsizex-xmergin)/3*length/width)+2
    
    fig = plt.figure(figsize = (figsizex, figsizey))

    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1) #index start from 1
        im = ax.imshow(data3[i], vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title3[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if cbar: fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(pngfile)
    plt.close()
   
    return 


#%% 
def plot_gacos_info(gacos_infofile, pngfile):
    figsize = (7, 3) #3x7
    sizec, colorc, markerc, alphac = 2, 'k', 'o', 0.8
    
    ### Figure
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 2, 1) #index start from 1
    ax2 = fig.add_subplot(1, 2, 2) #index start from 1
    
    ### Read data
    with open(gacos_infofile, "r") as f:
        info = f.readlines()[1:]

    std_bf, std_af, rate = [], [], []; 
    
    for line in info:
        date, std_bf1, std_af1, rate1 = line.split()
        if std_bf1=='0.0' or std_bf1=='nan' or std_af1=='0.0' or std_af1=='nan':
            continue
        std_bf.append(float(std_bf1))
        std_af.append(float(std_af1))
        rate.append(float(rate1[:-1]))
    
    std_bf = np.array(std_bf)
    std_af = np.array(std_af)
    rate = np.array(rate)
    rate[rate>99] = 99
    rate[rate< -99] = -99

    ### Plot
    xylim1 = np.max(np.concatenate((std_bf, std_af)))+1
    ax1.scatter(std_bf, std_af, s=sizec, c=colorc, marker=markerc, alpha=alphac, zorder=4)
    ax1.set_xlim(0, xylim1)
    ax1.set_ylim(0, xylim1)
    ax1.plot([0, xylim1], [0, xylim1], linewidth=2, color='grey', alpha=0.5, zorder=2)
    ax1.grid(zorder=0)
    ax1.set_xlabel('STD before GACOS (rad)')
    ax1.set_ylabel('STD after GACOS (rad)')

    ### Plot
    ax2.scatter(std_bf, rate, s=sizec, c=colorc, marker=markerc, alpha=alphac, zorder=4)
    ax2.plot([0, xylim1], [0, 0], linewidth=2, color='grey', alpha=0.5, zorder=2)
    ax2.grid(zorder=0)
    ax2.set_xlim(0, xylim1)
    ax2.set_xlabel('STD before GACOS (rad)')
    ax2.set_ylabel('STD reduction rate (%)')

    fig.tight_layout()
    fig.savefig(pngfile)


#%%
def plot_hgt_corr(data_bf, fit_hgt, hgt, title, pngfile):
    """
    """
    bool_nan = np.isnan(data_bf)
    data_af = data_bf - fit_hgt ### Correction
    ix_hgt0 = np.nanargmin(hgt[~bool_nan])
    ix_hgt1 = np.nanargmax(hgt[~bool_nan])
    hgt0 = hgt[~bool_nan][ix_hgt0]
    hgt1 = hgt[~bool_nan][ix_hgt1]
    fit_hgt0 = fit_hgt[~bool_nan][ix_hgt0]
    fit_hgt1 = fit_hgt[~bool_nan][ix_hgt1]
    
    ### Downsample data to plot large number of scatters fast
    hgt_data_bf = np.stack((np.round(hgt[~bool_nan]), np.round(data_bf[~bool_nan], 1))).T  ## Round values
    hgt_data_bf = np.unique(hgt_data_bf, axis = 0)  ## Keep only uniques
    hgt_data_af = np.stack((np.round(hgt[~bool_nan]), np.round(data_af[~bool_nan], 1))).T  ## Round values
    hgt_data_af = np.unique(hgt_data_af, axis = 0)  ## Keep only uniques

    ### Plot    
    figsize = (5, 4)
    sbf, cbf, mbf, zbf, lbf = 0.2, '0.5', 'p', 4, 'Before'
    saf, caf, maf, zaf, laf = 0.2, 'c', 'p', 6, 'After'

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    ax.scatter(hgt_data_bf[:, 0], hgt_data_bf[:, 1], s=sbf, c=cbf, marker=mbf, zorder=zbf, label=lbf)
    ax.scatter(hgt_data_af[:, 0], hgt_data_af[:, 1], s=saf, c=caf, marker=maf, zorder=zaf, label=laf)
    ax.plot([hgt0, hgt1], [fit_hgt0, fit_hgt1], linewidth=2, color='k', alpha=0.8, zorder=8, label='Correction')

    ax.grid(zorder=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Height (m)')
    ax.set_ylabel('Displacement (mm)')

    ax.legend()
    fig.tight_layout()
    fig.savefig(pngfile)
    plt.close()

    return 


#%%
def plot_network(ifgdates, bperp, rm_ifgdates, pngfile, plot_bad=True):
    """
    Plot network of interferometric pairs.
    
    bperp can be dummy (-1~1).
    Suffix of pngfile can be png, ps, pdf, or svg.
    plot_bad
        True  : Plot bad ifgs by red lines
        False : Do not plot bad ifgs
    """

    imdates_all = tools_lib.ifgdates2imdates(ifgdates)
    n_im_all = len(imdates_all)
    imdates_dt_all = np.array(([dt.datetime.strptime(imd, '%Y%m%d') for imd in imdates_all])) ##datetime

    ifgdates = list(set(ifgdates)-set(rm_ifgdates))
    ifgdates.sort()
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    n_im = len(imdates)
    imdates_dt = np.array(([dt.datetime.strptime(imd, '%Y%m%d') for imd in imdates])) ##datetime
    
    ### Identify gaps    
    G = inv_lib.make_sb_matrix(ifgdates)
    ixs_inc_gap = np.where(G.sum(axis=0)==0)[0]
    
    ### Plot fig
    figsize_x = np.round(((imdates_dt_all[-1]-imdates_dt_all[0]).days)/80)+2
    fig = plt.figure(figsize=(figsize_x, 6))
    ax = fig.add_axes([0.06, 0.12, 0.92,0.85])
    
    ### IFG blue lines
    for i, ifgd in enumerate(ifgdates):
        ix_m = imdates_all.index(ifgd[:8])
        ix_s = imdates_all.index(ifgd[-8:])
        label = 'IFG' if i==0 else '' #label only first
        plt.plot([imdates_dt_all[ix_m], imdates_dt_all[ix_s]], [bperp[ix_m],
                bperp[ix_s]], color='b', alpha=0.6, zorder=2, label=label)

    ### IFG bad red lines
    if plot_bad:
        for i, ifgd in enumerate(rm_ifgdates):
            ix_m = imdates_all.index(ifgd[:8])
            ix_s = imdates_all.index(ifgd[-8:])
            label = 'Removed IFG' if i==0 else '' #label only first
            plt.plot([imdates_dt_all[ix_m], imdates_dt_all[ix_s]], [bperp[ix_m],
                    bperp[ix_s]], color='r', alpha=0.6, zorder=6, label=label)

    ### Image points and dates
    ax.scatter(imdates_dt_all, bperp, alpha=0.6, zorder=4)
    for i in range(n_im_all):
        if bperp[i] > np.median(bperp): va='bottom'
        else: va = 'top'
        ax.annotate(imdates_all[i][4:6]+'/'+imdates_all[i][6:],
                    (imdates_dt_all[i], bperp[i]), ha='center', va=va, zorder=8)

    ### gaps
    if len(ixs_inc_gap)!=0:
        gap_dates_dt = []
        for ix_gap in ixs_inc_gap:
            ddays_td = imdates_dt[ix_gap+1]-imdates_dt[ix_gap]
            gap_dates_dt.append(imdates_dt[ix_gap]+ddays_td/2)
        plt.vlines(gap_dates_dt, 0, 1, transform=ax.get_xaxis_transform(),
                   zorder=1, label='Gap', alpha=0.6, colors='k', linewidth=3)
        
    ### Locater        
    loc = ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    try:  # Only support from Matplotlib 3.1
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    except:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
        for label in ax.get_xticklabels():
            label.set_rotation(20)
            label.set_horizontalalignment('right')
    ax.grid(b=True, which='major')

    ### Add bold line every 1yr
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.grid(b=True, which='minor', linewidth=2)

    ax.set_xlim((imdates_dt_all[0]-dt.timedelta(days=10),
                 imdates_dt_all[-1]+dt.timedelta(days=10)))

    ### Labels and legend
    plt.xlabel('Time')
    if np.all(np.abs(np.array(bperp))<=1): ## dummy
        plt.ylabel('dummy')
    else:
        plt.ylabel('Bperp [m]')
    
    plt.legend()

    ### Save
    plt.savefig(pngfile)
    plt.close()


