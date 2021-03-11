#!/usr/bin/env python3
"""
v1.6 20210311 Yu Morishita, GSI

========
Overview
========
This script identifies bad unw by checking loop closure.
A preliminary reference point that has all valid unw data and the smallest RMS
of loop phases is also determined.

===============
Input & output files
===============
Inputs in GEOCml*/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw[.png]
   - yyyymmdd_yyyymmdd.cc
 - slc.mli.par
 - baselines (may be dummy)

Inputs in TS_GEOCml*/ :
 - info/11bad_ifg.txt  : List of bad ifgs identified in step11

Outputs in TS_GEOCml*/ :
 - 12loop/
   - loop_info.txt : Statistical information of loop phase closure
   - bad_ifg_*.txt : List of bad ifgs identified by loop closure
   - good_loop_png/*.png : png images of good loop phase closure
   - bad_loop_png/*.png  : png images of bad loop phase closure
   - bad_loop_cand_png/*.png : png images of bad loop candidates in which
                               bad ifgs were not identified
   - loop_ph_rms_masked[.png] : RMS of loop phases used for ref selection

 - info/
   - 12ref.txt           : Preliminaly ref point for SB inversion (X/Y)
   - 12removed_image.txt : List of images to be removed in further processing
   - 12bad_ifg.txt       : List of bad ifgs to be removed in further processing
   - 12network_gap_info.txt : Information of gaps in network
   - 12no_loop_ifg.txt   : List of ifgs with no loop
                           Recommend to check the quality manually.
 - results/
   - n_unw[.png]      : Number of available unwrapped data to be used
   - coh_avg[.png]    : Average coherence
   - n_loop_err[.png] : Number of remaining loop errors (>pi) in data to be used
 - 12ifg_ras/*.png     : png (link) of unw to be used
 - 12bad_ifg_cand_ras/*.png : png (link) of unw to be used but candidates of bad
 - 12bad_ifg_ras/*.png : png (link) of unw to be removed
 - 12no_loop_ifg_ras/*.png : png (link) of unw with no loop
 - network/network12*.png  : Figures of the network

=====
Usage
=====
LiCSBAS12_loop_closure.py -d ifgdir [-t tsadir] [-l loop_thre] [--multi_prime]
 [--rm_ifg_list file] [--n_para int]

 -d  Path to the GEOCml* dir containing stack of unw data.
 -t  Path to the output TS_GEOCml* dir. (Default: TS_GEOCml*)
 -l  Threshold of RMS of loop phase (Default: 1.5 rad)
 --multi_prime  Multi Prime mode (take into account bias in loop)
 --rm_ifg_list  Manually remove ifgs listed in a file
 --n_para  Number of parallel processing (Default: # of usable CPU)

"""
#%% Change log
'''
v1.6 20210311 Yu Morishita, GSI
 - Add --rm_ifg_list option
v1.5.3 20201118 Yu Morishita, GSI
 - Again Bug fix of multiprocessing
v1.5.2 20201116 Yu Morishita, GSI
 - Bug fix of multiprocessing in Mac python>=3.8
v1.5.1 20201028 Yu Morishita, GSI
 - Update how to get n_para
v1.5 20201016 Yu Morishita, GSI
 - Bug fix in identifying bad_ifg_cand2
v1.4 20201007 Yu Morishita, GSI
 - Add --multi_prime option
 - Parallel processing in 2-4th loop
v1.3 20200907 Yu Morishita, GSI
 - Parallel processing in 1st loop
v1.2 20200228 Yu Morishita, Uni of Leeds and GSI
 - Not output network pdf
 - Improve bad loop cand identification
 - Change color of png
 - Deal with cc file in uint8 format
 - Change ref.txt name
v1.1 20191106 Yu Morishita, Uni of Leeds and GSI
 - Add iteration during ref search when no ref found
v1.0 20190730 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''

#%% Import
import getopt
import os
import sys
import time
import shutil
import glob
import numpy as np
import datetime as dt
import multiprocessing as multi
import SCM
import LiCSBAS_io_lib as io_lib
import LiCSBAS_loop_lib as loop_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_inv_lib as inv_lib
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
    ver="1.6"; date=20210311; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    global Aloop, ifgdates, ifgdir, length, width, loop_pngdir, cycle, \
        multi_prime, bad_ifg, noref_ifg, bad_ifg_all, refy1, refy2, refx1, refx2 ## for parallel processing

    #%% Set default
    ifgdir = []
    tsadir = []
    loop_thre = 1.5
    multi_prime = False
    rm_ifg_list = []

    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()

    cycle = 3 # 2pi*3/cycle
    cmap_noise = 'viridis'
    cmap_noise_r = 'viridis_r'
    q = multi.get_context('fork')


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hd:t:l:",
                                       ["help", "multi_prime",
                                        "rm_ifg_list=", "n_para="])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-d':
                ifgdir = a
            elif o == '-t':
                tsadir = a
            elif o == '-l':
                loop_thre = float(a)
            elif o == '--multi_prime':
                multi_prime = True
            elif o == '--rm_ifg_list':
                rm_ifg_list = a
            elif o == '--n_para':
                n_para = int(a)

        if not ifgdir:
            raise Usage('No data directory given, -d is not optional!')
        elif not os.path.isdir(ifgdir):
            raise Usage('No {} dir exists!'.format(ifgdir))
        elif not os.path.exists(os.path.join(ifgdir, 'slc.mli.par')):
                raise Usage('No slc.mli.par file exists in {}!'.format(ifgdir))
        if rm_ifg_list and not os.path.exists(rm_ifg_list):
            raise Usage('No {} exists!'.format(rm_ifg_list))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2

    print("\nloop_thre : {} rad".format(loop_thre), flush=True)


    #%% Directory setting
    ifgdir = os.path.abspath(ifgdir)

    if not tsadir:
        tsadir = os.path.join(os.path.dirname(ifgdir), 'TS_'+os.path.basename(ifgdir))

    if not os.path.isdir(tsadir):
        print('\nNo {} exists!'.format(tsadir), file=sys.stderr)
        return 1

    tsadir = os.path.abspath(tsadir)

    loopdir = os.path.join(tsadir, '12loop')
    if not os.path.exists(loopdir): os.mkdir(loopdir)

    loop_pngdir = os.path.join(loopdir ,'good_loop_png')
    bad_loop_pngdir = os.path.join(loopdir,'bad_loop_png')
    bad_loop_cand_pngdir = os.path.join(loopdir,'bad_loop_cand_png')

    if os.path.exists(loop_pngdir):
        shutil.move(loop_pngdir+'/', loop_pngdir+'_old') #move to old dir
    if os.path.exists(bad_loop_pngdir):
        for png in glob.glob(bad_loop_pngdir+'/*.png'):
            shutil.move(png, loop_pngdir+'_old') #move to old dir
        shutil.rmtree(bad_loop_pngdir)
    if os.path.exists(bad_loop_cand_pngdir):
        for png in glob.glob(bad_loop_cand_pngdir+'/*.png'):
            shutil.move(png, loop_pngdir+'_old') #move to old dir
        shutil.rmtree(bad_loop_cand_pngdir)

    os.mkdir(loop_pngdir)
    os.mkdir(bad_loop_pngdir)
    os.mkdir(bad_loop_cand_pngdir)

    ifg_rasdir = os.path.join(tsadir, '12ifg_ras')
    if os.path.isdir(ifg_rasdir): shutil.rmtree(ifg_rasdir)
    os.mkdir(ifg_rasdir)

    bad_ifgrasdir = os.path.join(tsadir, '12bad_ifg_ras')
    if os.path.isdir(bad_ifgrasdir): shutil.rmtree(bad_ifgrasdir)
    os.mkdir(bad_ifgrasdir)

    bad_ifg_candrasdir = os.path.join(tsadir, '12bad_ifg_cand_ras')
    if os.path.isdir(bad_ifg_candrasdir): shutil.rmtree(bad_ifg_candrasdir)
    os.mkdir(bad_ifg_candrasdir)

    no_loop_ifgrasdir = os.path.join(tsadir, '12no_loop_ifg_ras')
    if os.path.isdir(no_loop_ifgrasdir): shutil.rmtree(no_loop_ifgrasdir)
    os.mkdir(no_loop_ifgrasdir)

    infodir = os.path.join(tsadir, 'info')
    if not os.path.exists(infodir): os.mkdir(infodir)

    resultsdir = os.path.join(tsadir, 'results')
    if not os.path.exists(resultsdir): os.mkdir(resultsdir)

    netdir = os.path.join(tsadir, 'network')


    #%% Read date, network information and size
    ### Get dates
    ifgdates = tools_lib.get_ifgdates(ifgdir)

    ### Read bad_ifg11 and rm_ifg
    bad_ifg11file = os.path.join(infodir, '11bad_ifg.txt')
    bad_ifg11 = io_lib.read_ifg_list(bad_ifg11file)

    ### Remove bad ifgs and images from list
    ifgdates = list(set(ifgdates)-set(bad_ifg11))
    ifgdates.sort()

    imdates = tools_lib.ifgdates2imdates(ifgdates)

    n_ifg = len(ifgdates)
    n_im = len(imdates)

    ### Get size
    mlipar = os.path.join(ifgdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

    ### Get loop matrix
    Aloop = loop_lib.make_loop_matrix(ifgdates)
    n_loop = Aloop.shape[0]

    ### Extract no loop ifgs
    ns_loop4ifg = np.abs(Aloop).sum(axis=0)
    ixs_ifg_no_loop = np.where(ns_loop4ifg==0)[0]
    no_loop_ifg = [ifgdates[ix] for ix in ixs_ifg_no_loop]


    #%% 1st loop closure check. First without reference
    _n_para = n_para if n_para < n_loop else n_loop
    print('\n1st Loop closure check and make png for all possible {} loops,'.format(n_loop), flush=True)
    print('with {} parallel processing...'.format(_n_para), flush=True)

    bad_ifg_cand = []
    good_ifg = []

    ### Parallel processing
    p = q.Pool(_n_para)
    loop_ph_rms_ifg = np.array(p.map(loop_closure_1st_wrapper, range(n_loop)), dtype=np.float32)
    p.close()


    for i in range(n_loop):
        ### Find index of ifg
        ix_ifg12, ix_ifg23 = np.where(Aloop[i, :] == 1)[0]
        ix_ifg13 = np.where(Aloop[i, :] == -1)[0][0]
        ifgd12 = ifgdates[ix_ifg12]
        ifgd23 = ifgdates[ix_ifg23]
        ifgd13 = ifgdates[ix_ifg13]

        ### List as good or bad candidate
        if loop_ph_rms_ifg[i] >= loop_thre: #Bad loop including bad ifg.
            bad_ifg_cand.extend([ifgd12, ifgd23, ifgd13])
        else:
            good_ifg.extend([ifgd12, ifgd23, ifgd13])

    if os.path.exists(loop_pngdir+'_old/'):
        shutil.rmtree(loop_pngdir+'_old/')


    #%% Identify bad ifgs and output text
    bad_ifg1 = loop_lib.identify_bad_ifg(bad_ifg_cand, good_ifg)

    bad_ifgfile = os.path.join(loopdir, 'bad_ifg_loop.txt')
    with open(bad_ifgfile, 'w') as f:
        for i in bad_ifg1:
            print('{}'.format(i), file=f)

    ### Drop manually indicated ifg
    if rm_ifg_list:
        rm_ifg = io_lib.read_ifg_list(rm_ifg_list)
        bad_ifg = list(set(bad_ifg1+rm_ifg))

        rm_ifgfile = os.path.join(loopdir, 'rm_ifg_man.txt')
        print("\nFollowing ifgs are manually removed by {}:".format(
            rm_ifg_list), flush=True)
        with open(rm_ifgfile, 'w') as f:
            for i in rm_ifg:
                print('{}'.format(i), file=f)
                print('{}'.format(i), flush=True)
    else:
        rm_ifg = []
        bad_ifg = bad_ifg1

    ### Compute n_unw without bad_ifg11 and bad_ifg
    n_unw = np.zeros((length, width), dtype=np.int16)
    for ifgd in ifgdates:
        if ifgd in bad_ifg:
            continue

        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        unw = io_lib.read_img(unwfile, length, width)

        unw[unw == 0] = np.nan # Fill 0 with nan
        n_unw += ~np.isnan(unw) # Summing number of unnan unw


    #%% 2nd loop closure check without bad ifgs to define stable ref area
    ### Devide n_loop for paralell proc
    _n_para2, args = tools_lib.get_patchrow(1, n_loop, 2**20/4, int(np.ceil(n_loop/n_para)))

    print('\n2nd Loop closure check without bad ifgs to define ref area...', flush=True)
    print('with {} parallel processing...'.format(_n_para2), flush=True)

    ### Parallel processing
    p = q.Pool(_n_para2)
    res = np.array(p.map(loop_closure_2nd_wrapper, args), dtype=np.float32)
    p.close()

    ns_loop_ph = np.sum(res[:, 0, :, :,], axis=0)
    ns_loop_ph[ns_loop_ph==0] = np.nan # To avoid 0 division

    ns_bad_loop = np.sum(res[:, 1, :, :,], axis=0)
    loop_ph_rms_points = np.sum(res[:, 2, :, :,], axis=0)
    loop_ph_rms_points = np.sqrt(loop_ph_rms_points/ns_loop_ph)

    ### Find stable ref area which have all n_unw and minimum ns_bad_loop and loop_ph_rms_points
    mask1 = (n_unw==np.nanmax(n_unw))
    min_ns_bad_loop = np.nanmin(ns_bad_loop)
    while True:
        mask2 = (ns_bad_loop==min_ns_bad_loop)
        if np.all(~(mask1*mask2)): ## All masked
            min_ns_bad_loop = min_ns_bad_loop+1 ## Make mask2 again
        else:
            break
    loop_ph_rms_points_masked = loop_ph_rms_points*mask1*mask2
    loop_ph_rms_points_masked[loop_ph_rms_points_masked==0] = np.nan
    refyx = np.where(loop_ph_rms_points_masked==np.nanmin(loop_ph_rms_points_masked))
    refy1 = refyx[0][0] # start from 0, not 1
    refy2 = refyx[0][0]+1 # shift +1 for python custom. start from 1 end with width
    refx1 = refyx[1][0]
    refx2 = refyx[1][0]+1

    ### Save 12ref.txt
    reffile = os.path.join(infodir, '12ref.txt')
    with open(reffile, 'w') as f:
        print('{0}:{1}/{2}:{3}'.format(refx1, refx2, refy1, refy2), file=f)

    ### Save loop_ph_rms_masked and png
    loop_ph_rms_maskedfile = os.path.join(loopdir, 'loop_ph_rms_masked')
    loop_ph_rms_points_masked.tofile(loop_ph_rms_maskedfile)

    cmax = np.nanpercentile(loop_ph_rms_points_masked, 95)
    pngfile = loop_ph_rms_maskedfile+'.png'
    title = 'RMS of loop phase (rad)'
    plot_lib.make_im_png(loop_ph_rms_points_masked, pngfile, cmap_noise_r, title, None, cmax)

    ### Check ref exist in unw. If not, list as noref_ifg
    noref_ifg = []
    for ifgd in ifgdates:
        if ifgd in bad_ifg:
            continue

        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        unw_ref = io_lib.read_img(unwfile, length, width)[refy1:refy2, refx1:refx2]

        unw_ref[unw_ref == 0] = np.nan # Fill 0 with nan
        if np.all(np.isnan(unw_ref)):
            noref_ifg.append(ifgd)

    bad_ifgfile = os.path.join(loopdir, 'bad_ifg_noref.txt')
    with open(bad_ifgfile, 'w') as f:
        for i in noref_ifg:
            print('{}'.format(i), file=f)


    #%% 3rd loop closure check without bad ifgs wrt ref point
    print('\n3rd loop closure check taking into account ref phase...', flush=True)
    print('with {} parallel processing...'.format(_n_para), flush=True)

    ### Parallel processing
    p = q.Pool(_n_para)
    loop_ph_rms_ifg2 = list(np.array(p.map(loop_closure_3rd_wrapper, range(n_loop)), dtype=np.float32))
    p.close()

    bad_ifg_cand2 = []
    good_ifg2 = []
    ### List as good or bad candidate
    for i in range(n_loop):
        ### Find index of ifg
        ix_ifg12, ix_ifg23 = np.where(Aloop[i, :] == 1)[0]
        ix_ifg13 = np.where(Aloop[i, :] == -1)[0][0]
        ifgd12 = ifgdates[ix_ifg12]
        ifgd23 = ifgdates[ix_ifg23]
        ifgd13 = ifgdates[ix_ifg13]

        if np.isnan(loop_ph_rms_ifg2[i]): # Skipped
            loop_ph_rms_ifg2[i] = '--' ## Replace
        elif loop_ph_rms_ifg2[i] >= loop_thre: #Bad loop including bad ifg.
            bad_ifg_cand2.extend([ifgd12, ifgd23, ifgd13])
        else:
            good_ifg2.extend([ifgd12, ifgd23, ifgd13])


    #%% Identify additional bad ifgs and output text
    bad_ifg2 = loop_lib.identify_bad_ifg(bad_ifg_cand2, good_ifg2)

    bad_ifgfile = os.path.join(loopdir, 'bad_ifg_loopref.txt')
    with open(bad_ifgfile, 'w') as f:
        for i in bad_ifg2:
            print('{}'.format(i), file=f)


    #%% Output all bad ifg list and identify remaining candidate of bad ifgs
    ### Merge bad ifg, bad_ifg2, noref_ifg
    bad_ifg_all = list(set(bad_ifg+bad_ifg2+noref_ifg))  # Remove multiple
    bad_ifg_all.sort()

    ifgdates_good = list(set(ifgdates)-set(bad_ifg_all))
    ifgdates_good.sort()

    bad_ifgfile = os.path.join(infodir, '12bad_ifg.txt')
    with open(bad_ifgfile, 'w') as f:
        for i in bad_ifg_all:
            print('{}'.format(i), file=f)

    ### Identify removed image and output file
    imdates_good = tools_lib.ifgdates2imdates(ifgdates_good)
    imdates_bad = list(set(imdates)-set(imdates_good))
    imdates_bad.sort()

    bad_imfile = os.path.join(infodir, '12removed_image.txt')
    with open(bad_imfile, 'w') as f:
        for i in imdates_bad:
            print('{}'.format(i), file=f)

    ### Remaining candidate of bad ifg
    bad_ifg_cand_res = list(set(bad_ifg_cand2)-set(bad_ifg_all))
    bad_ifg_cand_res.sort()

    bad_ifg_candfile = os.path.join(infodir, '12bad_ifg_cand.txt')
    with open(bad_ifg_candfile, 'w') as f:
        for i in bad_ifg_cand_res:
            print('{}'.format(i), file=f)


    #%% 4th loop to be used to calc n_loop_err and n_ifg_noloop
    print('\n4th loop to compute statistics...', flush=True)
    print('with {} parallel processing...'.format(_n_para2), flush=True)

    ### Parallel processing
    p = q.Pool(_n_para2)
    res = np.array(p.map(loop_closure_4th_wrapper, args), dtype=np.int16)
    p.close()

    ns_loop_err = np.sum(res[:, :, :,], axis=0)


    #%% Output loop info, move bad_loop_png
    loop_info_file = os.path.join(loopdir, 'loop_info.txt')
    f = open(loop_info_file, 'w')
    print('# loop_thre: {} rad. *: Removed w/o ref, **: Removed w/ ref'.format(
        loop_thre), file=f)
    if rm_ifg_list:
        print('# +: Removed by manually indicating in {}'.format(rm_ifg_list),
              file=f)
    print('# /: Candidates of bad loops but causative ifgs unidentified',
          file=f)
    print('# image1   image2   image3 RMS w/oref  w/ref', file=f)

    for i in range(n_loop):
        ### Find index of ifg
        ix_ifg12, ix_ifg23 = np.where(Aloop[i, :] == 1)[0]
        ix_ifg13 = np.where(Aloop[i, :] == -1)[0][0]
        ifgd12 = ifgdates[ix_ifg12]
        ifgd23 = ifgdates[ix_ifg23]
        ifgd13 = ifgdates[ix_ifg13]
        imd1 = ifgd12[:8]
        imd2 = ifgd23[:8]
        imd3 = ifgd23[-8:]

        ## Move loop_png if bad ifg or bad ifg_cand is included
        looppngfile = os.path.join(loop_pngdir, '{0}_{1}_{2}_loop.png'.format(imd1, imd2, imd3))
        badlooppngfile = os.path.join(bad_loop_pngdir, '{0}_{1}_{2}_loop.png'.format(imd1, imd2, imd3))
        badloopcandpngfile = os.path.join(bad_loop_cand_pngdir, '{0}_{1}_{2}_loop.png'.format(imd1, imd2, imd3))

        badloopflag1 = ' '
        badloopflag2 = '  '
        if ifgd12 in bad_ifg1 or ifgd23 in bad_ifg1 or ifgd13 in bad_ifg1:
            badloopflag1 = '*'
            shutil.move(looppngfile, badlooppngfile)
        elif ifgd12 in rm_ifg or ifgd23 in rm_ifg or ifgd13 in rm_ifg:
            badloopflag1 = '+'
            shutil.move(looppngfile, badlooppngfile)
        elif ifgd12 in bad_ifg2 or ifgd23 in bad_ifg2 or ifgd13 in bad_ifg2:
            badloopflag2 = '**'
            shutil.move(looppngfile, badlooppngfile)
        elif ifgd12 in bad_ifg_cand_res or ifgd23 in bad_ifg_cand_res or ifgd13 in bad_ifg_cand_res:
            badloopflag1 = '/'
            if os.path.exists(looppngfile):
                shutil.move(looppngfile, badloopcandpngfile)

        if type(loop_ph_rms_ifg2[i]) == np.float32:
            str_loop_ph_rms_ifg2 = "{:.2f}".format(loop_ph_rms_ifg2[i])
        else: ## --
            str_loop_ph_rms_ifg2 = loop_ph_rms_ifg2[i]

        print('{0} {1} {2}    {3:5.2f} {4}  {5:5s} {6}'.format(imd1, imd2, imd3, loop_ph_rms_ifg[i], badloopflag1, str_loop_ph_rms_ifg2, badloopflag2), file=f)

    f.close()


    #%% Saving coh_avg, n_unw, and n_loop_err only for good ifgs
    print('\nSaving coh_avg, n_unw, and n_loop_err...', flush=True)
    ### Calc coh avg and n_unw
    coh_avg = np.zeros((length, width), dtype=np.float32)
    n_coh = np.zeros((length, width), dtype=np.int16)
    n_unw = np.zeros((length, width), dtype=np.int16)
    for ifgd in ifgdates_good:
        ccfile = os.path.join(ifgdir, ifgd, ifgd+'.cc')
        if os.path.getsize(ccfile) == length*width:
            coh = io_lib.read_img(ccfile, length, width, np.uint8)
            coh = coh.astype(np.float32)/255
        else:
            coh = io_lib.read_img(ccfile, length, width)
            coh[np.isnan(coh)] = 0 # Fill nan with 0

        coh_avg += coh
        n_coh += (coh!=0)

        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        unw = io_lib.read_img(unwfile, length, width)

        unw[unw == 0] = np.nan # Fill 0 with nan
        n_unw += ~np.isnan(unw) # Summing number of unnan unw

    coh_avg[n_coh==0] = np.nan
    n_coh[n_coh==0] = 1 #to avoid zero division
    coh_avg = coh_avg/n_coh

    ### Save files
    n_unwfile = os.path.join(resultsdir, 'n_unw')
    np.float32(n_unw).tofile(n_unwfile)

    coh_avgfile = os.path.join(resultsdir, 'coh_avg')
    coh_avg.tofile(coh_avgfile)

    n_loop_errfile = os.path.join(resultsdir, 'n_loop_err')
    np.float32(ns_loop_err).tofile(n_loop_errfile)


    ### Save png
    title = 'Average coherence'
    plot_lib.make_im_png(coh_avg, coh_avgfile+'.png', cmap_noise, title)
    title = 'Number of used unw data'
    plot_lib.make_im_png(n_unw, n_unwfile+'.png', cmap_noise, title, n_im)

    title = 'Number of unclosed loops'
    plot_lib.make_im_png(ns_loop_err, n_loop_errfile+'.png', cmap_noise_r, title)


    #%% Link ras
    ### First, identify suffix of raster image (ras, bmp, or png?)
    unwfile = os.path.join(ifgdir, ifgdates[0], ifgdates[0]+'.unw')
    if os.path.exists(unwfile+'.ras'):
        suffix = '.ras'
    elif os.path.exists(unwfile+'.bmp'):
        suffix = '.bmp'
    elif os.path.exists(unwfile+'.png'):
        suffix = '.png'


    for ifgd in ifgdates:
        rasname = ifgd+'.unw'+suffix
        rasorg = os.path.join(ifgdir, ifgd, rasname)
        ### Bad ifgs
        if ifgd in bad_ifg_all:
            os.symlink(os.path.relpath(rasorg, bad_ifgrasdir), os.path.join(bad_ifgrasdir, rasname))
        ### Remaining bad ifg candidates
        elif ifgd in bad_ifg_cand_res:
            os.symlink(os.path.relpath(rasorg, bad_ifg_candrasdir), os.path.join(bad_ifg_candrasdir, rasname))
        ### Good ifgs
        else:
            os.symlink(os.path.relpath(rasorg, ifg_rasdir), os.path.join(ifg_rasdir, rasname))

        if ifgd in no_loop_ifg:
            os.symlink(os.path.relpath(rasorg, no_loop_ifgrasdir), os.path.join(no_loop_ifgrasdir, rasname))


    #%% Plot network
    ## Read bperp data or dummy
    bperp_file = os.path.join(ifgdir, 'baselines')
    if os.path.exists(bperp_file):
        bperp = io_lib.read_bperp_file(bperp_file, imdates)
    else: #dummy
        bperp = np.random.random(n_im).tolist()

    pngfile = os.path.join(netdir, 'network12_all.png')
    plot_lib.plot_network(ifgdates, bperp, [], pngfile)

    pngfile = os.path.join(netdir, 'network12.png')
    plot_lib.plot_network(ifgdates, bperp, bad_ifg_all, pngfile)

    pngfile = os.path.join(netdir, 'network12_nobad.png')
    plot_lib.plot_network(ifgdates, bperp, bad_ifg_all, pngfile, plot_bad=False)

    ### Network info
    ## Identify gaps
    G = inv_lib.make_sb_matrix(ifgdates_good)
    ixs_inc_gap = np.where(G.sum(axis=0)==0)[0]

    ## Connected network
    ix1 = 0
    connected_list = []
    for ix2 in np.append(ixs_inc_gap, len(imdates_good)-1): #append for last image
        imd1 = imdates_good[ix1]
        imd2 = imdates_good[ix2]
        dyear = (dt.datetime.strptime(imd2, '%Y%m%d').toordinal() - dt.datetime.strptime(imd1, '%Y%m%d').toordinal())/365.25
        n_im_connect = ix2-ix1+1
        connected_list.append([imdates_good[ix1], imdates_good[ix2], dyear, n_im_connect])
        ix1 = ix2+1 # Next connection


    #%% Caution about no_loop ifg, remaining large RMS loop and gap
    ### no_loop ifg
    if len(no_loop_ifg)!=0:
        no_loop_ifgfile = os.path.join(infodir, '12no_loop_ifg.txt')
        with open(no_loop_ifgfile, 'w') as f:
            print("\nThere are {} ifgs without loop, recommend to check manually in no_loop_ifg_ras12".format(len(no_loop_ifg)), flush=True)
            for ifgd in no_loop_ifg:
                print('{}'.format(ifgd), flush=True)
                print('{}'.format(ifgd), file=f)

    ### Remaining candidates of bad ifgs
    if len(bad_ifg_cand_res)!=0:
        print("\nThere are {} remaining candidates of bad ifgs but not identified.".format(len(bad_ifg_cand_res)), flush=True)
        print("Check 12bad_ifg_cand_ras and loop/bad_loop_cand_png.", flush=True)
#        for ifgd in bad_ifg_cand_res:
#            print('{}'.format(ifgd))

    print('\n{0}/{1} ifgs are discarded from further processing.'.format(len(bad_ifg_all), n_ifg), flush=True)
    for ifgd in bad_ifg_all:
        print('{}'.format(ifgd), flush=True)

    ### Gap
    gap_infofile = os.path.join(infodir, '12network_gap_info.txt')
    with open(gap_infofile, 'w') as f:
        if ixs_inc_gap.size!=0:
            print("Gaps between:", file=f)
            print("\nGaps in network between:", flush=True)
            for ix in ixs_inc_gap:
                print("{} {}".format(imdates_good[ix], imdates_good[ix+1]), file=f)
                print("{} {}".format(imdates_good[ix], imdates_good[ix+1]), flush=True)

        print("\nConnected network (year, n_image):", file=f)
        print("\nConnected network (year, n_image):", flush=True)
        for list1 in connected_list:
            print("{0}-{1} ({2:.2f}, {3})".format(list1[0], list1[1], list1[2], list1[3]), file=f)
            print("{0}-{1} ({2:.2f}, {3})".format(list1[0], list1[1], list1[2], list1[3]), flush=True)

    print('\nIf you want to change the bad ifgs to be discarded, re-run with different thresholds before next step.', flush=True)


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(tsadir)))


#%%
def loop_closure_1st_wrapper(i):
    n_loop = Aloop.shape[0]

    if np.mod(i, 100) == 0:
        print("  {0:3}/{1:3}th loop...".format(i, n_loop), flush=True)

    unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = loop_lib.read_unw_loop_ph(Aloop[i, :], ifgdates, ifgdir, length, width)

    ## Calculate loop phase and check n bias (2pi*n)
    loop_ph = unw12+unw23-unw13
    loop_2pin = int(np.round(np.nanmedian(loop_ph)/(2*np.pi)))*2*np.pi
    loop_ph = loop_ph-loop_2pin #unbias 2pi x n

    if multi_prime:
        bias = np.nanmedian(loop_ph)
        loop_ph = loop_ph - bias # unbias inconsistent fraction phase

    rms = np.sqrt(np.nanmean(loop_ph**2))

    ### Output png. If exist in old, move to save time
    imd1 = ifgd12[:8]
    imd2 = ifgd23[:8]
    imd3 = ifgd23[-8:]
    png = os.path.join(loop_pngdir, imd1+'_'+imd2+'_'+imd3+'_loop.png')
    oldpng = os.path.join(loop_pngdir+'_old/', imd1+'_'+imd2+'_'+imd3+'_loop.png')
    if os.path.exists(oldpng):
        ### Just move from old png
        shutil.move(oldpng, loop_pngdir)
    else:
        ### Make png. Take time a little.
        titles4 = ['{} ({}*2pi/cycle)'.format(ifgd12, cycle),
                   '{} ({}*2pi/cycle)'.format(ifgd23, cycle),
                   '{} ({}*2pi/cycle)'.format(ifgd13, cycle),]
        if multi_prime:
            titles4.append('Loop (STD={:.2f}rad, bias={:.2f}rad)'.format(rms, bias))
        else:
            titles4.append('Loop phase (RMS={:.2f}rad)'.format(rms))

        loop_lib.make_loop_png(unw12, unw23, unw13, loop_ph, png, titles4, cycle)

    return rms


#%%
def loop_closure_2nd_wrapper(args):
    i0, i1 = args
    n_loop = Aloop.shape[0]
    ns_loop_ph1 = np.zeros((length, width), dtype=np.float32)
    ns_bad_loop1 = np.zeros((length, width), dtype=np.float32)
    loop_ph_rms_points1 = np.zeros((length, width), dtype=np.float32)

    for i in range(i0, i1):
        if np.mod(i, 100) == 0:
            print("  {0:3}/{1:3}th loop...".format(i, n_loop), flush=True)

        ### Read unw
        unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = loop_lib.read_unw_loop_ph(Aloop[i, :], ifgdates, ifgdir, length, width)

        ### Skip if bad ifg is included
        if ifgd12 in bad_ifg or ifgd23 in bad_ifg or ifgd13 in bad_ifg:
            continue

        ## Calculate loop phase and rms at points
        loop_ph = unw12+unw23-unw13
        loop_2pin = int(np.round(np.nanmedian(loop_ph)/(2*np.pi)))*2*np.pi
        loop_ph = loop_ph-loop_2pin #unbias

        if multi_prime:
            bias = np.nanmedian(loop_ph)
            loop_ph = loop_ph - bias # unbias inconsistent fraction phase

        ns_loop_ph1 = ns_loop_ph1 + ~np.isnan(loop_ph)

        loop_ph_sq = loop_ph**2
        loop_ph_sq[np.isnan(loop_ph_sq)] = 0
        loop_ph_rms_points1 = loop_ph_rms_points1 + loop_ph_sq

        ns_bad_loop1 = ns_bad_loop1+(loop_ph_sq>np.pi**2) #suspected unw error
#        ns_bad_loop = ns_bad_loop+(np.abs(loop_ph)>loop_thre)
        ## multiple nan seem to generate RuntimeWarning

    return ns_loop_ph1, ns_bad_loop1, loop_ph_rms_points1


#%%
def loop_closure_3rd_wrapper(i):
    n_loop = Aloop.shape[0]

    if np.mod(i, 100) == 0:
        print("  {0:3}/{1:3}th loop...".format(i, n_loop), flush=True)

    ### Read unw
    unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = loop_lib.read_unw_loop_ph(Aloop[i, :], ifgdates, ifgdir, length, width)

    ### Skip if bad ifg is included
    if ifgd12 in bad_ifg or ifgd23 in bad_ifg or ifgd13 in bad_ifg:
        return np.nan

    ### Skip if noref ifg is included
    if ifgd12 in noref_ifg or ifgd23 in noref_ifg or ifgd13 in noref_ifg:
        return np.nan

    ## Skip if no data in ref area in any unw. It is bad data.
    ref_unw12 = np.nanmean(unw12[refy1:refy2, refx1:refx2])
    ref_unw23 = np.nanmean(unw23[refy1:refy2, refx1:refx2])
    ref_unw13 = np.nanmean(unw13[refy1:refy2, refx1:refx2])

    ## Calculate loop phase taking into account ref phase
    loop_ph = unw12+unw23-unw13-(ref_unw12+ref_unw23-ref_unw13)
    return np.sqrt(np.nanmean((loop_ph)**2))


#%%
def loop_closure_4th_wrapper(args):
    i0, i1 = args
    n_loop = Aloop.shape[0]
    ns_loop_err1 = np.zeros((length, width), dtype=np.int16)

    for i in range(i0, i1):
        if np.mod(i, 100) == 0:
            print("  {0:3}/{1:3}th loop...".format(i, n_loop), flush=True)

        ### Read unw
        unw12, unw23, unw13, ifgd12, ifgd23, ifgd13 = loop_lib.read_unw_loop_ph(Aloop[i, :], ifgdates, ifgdir, length, width)

        ### Skip if bad ifg is included
        if ifgd12 in bad_ifg_all or ifgd23 in bad_ifg_all or ifgd13 in bad_ifg_all:
            continue

        ## Compute ref
        ref_unw12 = np.nanmean(unw12[refy1:refy2, refx1:refx2])
        ref_unw23 = np.nanmean(unw23[refy1:refy2, refx1:refx2])
        ref_unw13 = np.nanmean(unw13[refy1:refy2, refx1:refx2])

        ## Calculate loop phase taking into account ref phase
        loop_ph = unw12+unw23-unw13-(ref_unw12+ref_unw23-ref_unw13)

        ## Count number of loops with suspected unwrap error (>pi)
        loop_ph[np.isnan(loop_ph)] = 0 #to avoid warning
        ns_loop_err1 = ns_loop_err1+(np.abs(loop_ph)>np.pi) #suspected unw error

    return ns_loop_err1


#%% main
if __name__ == "__main__":
    sys.exit(main())
