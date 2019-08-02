#!/usr/bin/env python3
"""
========
Overview
========
Python3 library of plot functions for LiCSBAS.

=========
Changelog
=========
v1.0 20190729 Yu Morioshita, Uni of Leeds and GSI
 - Original implementation

"""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np
import datetime as dt

import warnings
import matplotlib
with warnings.catch_warnings(): ## To silence user warning
    warnings.simplefilter('ignore', UserWarning)
    matplotlib.use('Agg')
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
        plt.register_cmap(name='insar',data=cdict)
    
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
        plt.register_cmap(name='insar',data=cdict)

    length, width = data3[0].shape
    figsizex = 12
    xmergin = 4 if cbar else 0
    figsizey = int((figsizex-xmergin)/3*length/width)+1
    
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
def plot_network(ifgdates, bperp, rm_ifgdates, pngfile, plot_bad=True, pdf=False):
    """
    Plot network of interferometric pairs.
    
    bperp can be dummy (-1~1).
    plot_bad
        True  : Plot bad ifgs by red lines
        False : Do not plot bad ifgs
    """

    imdates_all = tools_lib.ifgdates2imdates(ifgdates)
    n_im_all = len(imdates_all)
    idlist_all = [dt.datetime.strptime(x, '%Y%m%d').toordinal() for x in imdates_all]

    ifgdates = list(set(ifgdates)-set(rm_ifgdates))
    ifgdates.sort()
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    n_im = len(imdates)
    idlist = [dt.datetime.strptime(x, '%Y%m%d').toordinal() for x in imdates]
    
    ### Identify gaps    
    G = inv_lib.make_sb_matrix(ifgdates)
    ixs_inc_gap = np.where(G.sum(axis=0)==0)[0]
    
    ### Plot fig
    figsize_x = np.round((idlist_all[-1]-idlist_all[0])/80)+2
    fig = plt.figure(figsize=(figsize_x, 6))
    ax = fig.add_axes([0.12, 0.12, 0.85,0.85])
    
    ### IFG blue lines
    for i, ifgd in enumerate(ifgdates):
        ix_m = imdates_all.index(ifgd[:8])
        ix_s = imdates_all.index(ifgd[-8:])
        label = 'IFG' if i==0 else '' #label only first
        plt.plot([idlist_all[ix_m], idlist_all[ix_s]], [bperp[ix_m], bperp[ix_s]], color='b', alpha=0.6, zorder=2, label=label)

    ### IFG bad red lines
    if plot_bad:
        for i, ifgd in enumerate(rm_ifgdates):
            ix_m = imdates_all.index(ifgd[:8])
            ix_s = imdates_all.index(ifgd[-8:])
            label = 'Removed IFG' if i==0 else '' #label only first
            plt.plot([idlist_all[ix_m], idlist_all[ix_s]], [bperp[ix_m], bperp[ix_s]], color='r', alpha=0.6, zorder=6, label=label)

    ### Image points and dates
    ax.scatter(idlist_all, bperp, alpha=0.6, zorder=4)
    for i in range(n_im_all):
        if bperp[i] > np.median(bperp): va='bottom'
        else: va = 'top'
        ax.annotate(imdates_all[i][4:6]+'/'+imdates_all[i][6:], (idlist_all[i], bperp[i]), ha='center', va=va, zorder=8)

    ### gaps
    if len(ixs_inc_gap)!=0:
        gap_idlist = []
        for ix_gap in ixs_inc_gap:
            gap_idlist.append((idlist[ix_gap]+idlist[ix_gap+1])/2)
        plt.vlines(gap_idlist, 0, 1, transform=ax.get_xaxis_transform(), zorder=1, label='Gap', alpha=0.6, colors='k', linewidth=3)
        
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

    ax.set_xlim((idlist_all[0]-10, idlist_all[-1]+10))

    ### Labels and legend
    plt.xlabel('Time')
    if np.all(np.abs(np.array(bperp))<=1): ## dummy
        plt.ylabel('dummy')
    else:
        plt.ylabel('Bperp [m]')
    
    plt.legend()

    ### Save
    plt.savefig(pngfile)
    if pdf:
        plt.savefig(pngfile.replace('.png', '.pdf'))
    
    plt.close()
