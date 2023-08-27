#!/usr/bin/env python3
"""
========
Overview
========
This script:
 - plots connected components output by SNAPHU,
 - plots residual in radian divided by 2pi and rounded to the nearest integer,
 - decides whether / how to correct each ifg
         - if RMS residual already below threshold:
            - keep good ifg
         - if nearest integer doesn't not reduce RMS residual to below threshold:
            - discard bad ifg
         - else (correct unwrapping mistake by integer multiples of 2pi):
            - correct each component by the mode of nearest integer in that component (preferred)
            - correct by nearest integer (if component mode doesn't reduce RMS residual to below threshold)

===============
Input & output files
===============

Inputs in GEOCml*/ (COMP_CC_DIR):
 - baselines
 - slc.mli.par
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.conncomp
   - yyyymmdd_yyyymmdd.cc

Inputs in GEOCml*/ (UNW_DIR):
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw

Inputs in TS_GEOCml*/ :
 - 13resid*/
   - yyyymmdd_yyyymmdd.res

 - info/
   - 131resid_2pi*.txt     : RMS residuals per IFG computed in radian and as a factor of 2pi

Outputs in GEOCml*/ (CORRECT_DIR):
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd.unw

Outputs in TS_GEOCml*/ :
 - 13resid*/
   - good_ifg_no_correction/*png : residuals of uncorrected ifgs
   - bad_ifg_no_correction/*png  : residuals and nearest integers showing why ifgs can't be corrected
   - integer_correction/*png     : ifgs corrected by nearest residual integer (when mode doesn't work)
   - mode_correction/*png        : ifgs corrected by component mode (preferred)

 - info/
   - 132good_ifg_uncorrected*.txt
   - 132bad_ifg*.txt
   - 132corrected_by_nearest_integer_ifg*.txt
   - 132corrected_by_component_mode_ifg*.txt

 - network/
   - network132*.png

=====
Usage
=====
LiCSBAS132_3D_correction.py [-h] [-f FRAME_DIR] [-c COMP_CC_DIR] [-g UNW_DIR]
                                   [-r CORRECT_DIR] [-t TS_DIR] [--thresh THRESH] [--suffix SUFFIX]
"""

from scipy import stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import copy
import glob
import argparse
import sys
import time
import re
from pathlib import Path
#import cmcrameri as cm
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib
import LiCSBAS_inv_lib as inv_lib
import shutil
import multiprocessing as multi


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    '''
    Use a multiple inheritance approach to use features of both classes.
    The ArgumentDefaultsHelpFormatter class adds argument default values to the usage help message
    The RawDescriptionHelpFormatter class keeps the indentation and line breaks in the ___doc___
    '''
    pass


def init_args():
    global args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument('-f', dest='frame_dir', default="./", help="directory of LiCSBAS output of a particular frame")
    parser.add_argument('-c', dest='comp_cc_dir', default="GEOCml10GACOS", help="folder containing connected components and cc files")
    # parser.add_argument('-d', dest='unw_dir', default="GEOCml10GACOS", help="folder containing unw input to be corrected")
    parser.add_argument('-o', dest='out_dir', default="GEOCml10GACOS_corrected", help="folder for corrected/masked unw")
    parser.add_argument('-t', dest='ts_dir', default="TS_GEOCml10GACOS", help="folder containing time series and residuals")
    parser.add_argument('-s', dest='correction_thresh', type=float, help="RMS residual per ifg (in 2pi) for correction, override info/131resid_2pi.txt")
    parser.add_argument('-g', dest='target_thresh', default='thresh', choices=['mode', 'median', 'mean', 'thresh'], help="RMS residual per ifg (in 2pi) for accepting the correction, read from info/131resid_2pi.txt, or follow correction_thresh if given")
    parser.add_argument('-l', dest='ifg_list', default=None, type=str, help="text file containing a list of ifgs")
    parser.add_argument('--suffix', default="", type=str, help="suffix of the input 131resid_2pi*.txt and outputs")
    parser.add_argument('-n', dest='n_para', type=int, help="number of processes for parallel processing")
    parser.add_argument('--move_weak',  action='store_true', help="move ifgs forming weak links to subfolder of correct_dir")
    parser.add_argument('--mask_by_residual',  action='store_true', help="perform masking instead of correction")
    parser.add_argument('-m', dest='mask_thresh', type=float, default=0.2, help="RMS residual per ifg (in 2pi) for correction")
    parser.add_argument('--no_depeak', default=False, action='store_true', help="don't offset by residual mode before calculation (recommend depeak)")
    parser.add_argument('--correction_by_mode', default=False, action='store_true', help="perform correction by mode only")
    parser.add_argument('--correction_by_integer', default=False, action='store_true', help="perform correction by integer only")
    parser.add_argument('--best_network', default=False, action='store_true', help="make a connected network with the best ifgs")
    args = parser.parse_args()


def start():
    global start_time
    # intialise and print info on screen
    start_time = time.time()
    ver="1.0"; date=20221020; author="Qi Ou"
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)


def finish():
    #%% Finish
    elapsed_time = time.time() - start_time
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))
    print("\n{} {} finished!".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)
    print('Output directory: {}\n'.format(os.path.relpath(tsadir)))


def set_input_output():
    global ccdir, unwdir, tsadir, resdir, infodir, netdir, correct_dir, good_png_dir, bad_png_dir, integer_png_dir, mode_png_dir, mask_png_dir, mask_dir

    # define input directories
    ccdir = os.path.abspath(os.path.join(args.frame_dir, args.comp_cc_dir))
    if args.suffix == "1":
        unwdir = os.path.abspath(os.path.join(args.frame_dir, args.comp_cc_dir))  # to read .unw
    else:
        unwdir = os.path.abspath(os.path.join(args.frame_dir, args.comp_cc_dir + args.suffix))  # to read .unw
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    resdir = os.path.join(tsadir, '130resid{}'.format(args.suffix))
    infodir = os.path.join(tsadir, 'info')

    # define output directories
    netdir = os.path.join(tsadir, 'network')

    if args.mask_by_residual:
        mask_dir = os.path.abspath(os.path.join(args.frame_dir, args.out_dir))
        if os.path.exists(mask_dir): shutil.rmtree(mask_dir)
        Path(mask_dir).mkdir(parents=True, exist_ok=True)

        mask_png_dir = os.path.join(resdir, 'mask_png/')
        if os.path.exists(mask_png_dir): shutil.rmtree(mask_png_dir)
        Path(mask_png_dir).mkdir(parents=True, exist_ok=True)
    elif args.best_network:
        pass
    else:  # perform correction
        correct_dir = os.path.abspath(os.path.join(args.frame_dir, args.out_dir))
        if os.path.exists(correct_dir): shutil.rmtree(correct_dir)
        Path(correct_dir).mkdir(parents=True, exist_ok=True)

        good_png_dir = os.path.join(resdir, 'good_ifg_no_correction/')
        if os.path.exists(good_png_dir): shutil.rmtree(good_png_dir)
        Path(good_png_dir).mkdir(parents=True, exist_ok=True)

        bad_png_dir = os.path.join(resdir, 'bad_ifg_no_correction/')
        if os.path.exists(bad_png_dir): shutil.rmtree(bad_png_dir)
        Path(bad_png_dir).mkdir(parents=True, exist_ok=True)

        integer_png_dir = os.path.join(resdir, 'integer_correction/')
        if os.path.exists(integer_png_dir): shutil.rmtree(integer_png_dir)
        Path(integer_png_dir).mkdir(parents=True, exist_ok=True)

        mode_png_dir = os.path.join(resdir, 'mode_correction/')
        if os.path.exists(mode_png_dir): shutil.rmtree(mode_png_dir)
        Path(mode_png_dir).mkdir(parents=True, exist_ok=True)


def get_para():
    global width, length, coef_r2m, correction_thresh, target_thresh, ref_x, ref_y, n_para, res_list, resid_threshold_file

    # read ifg size and satellite frequency
    mlipar = os.path.join(ccdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    radar_frequency = float(io_lib.get_param_par(mlipar, 'radar_frequency'))  # 5405000000.0 Hz for C-band
    speed_of_light = 299792458  # m/s
    wavelength = speed_of_light/radar_frequency
    coef_r2m = -wavelength/4/np.pi*1000

    # read reference for plotting purpose
    reffile = os.path.join(infodir, '120ref.txt')
    if not os.path.exists(reffile):
        reffile = os.path.join(infodir, '12ref.txt')
    with open(reffile, "r") as f:
        refarea = f.read().split()[0]  # str, x1/x2/y1/y2
    refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]
    ref_x = int((refx1 + refx2) / 2)
    ref_y = int((refy1 + refy2) / 2)

    # multi-processing
    if not args.n_para:
        try:
            n_para = min(len(os.sched_getaffinity(0)), 8) # maximum use 8 cores
        except:
            n_para = multi.cpu_count()
    else:
        n_para = args.n_para

    # check .res files exist
    if args.ifg_list:
        ifgdates = io_lib.read_ifg_list(args.ifg_list)
        res_list = [os.path.join(resdir, '{}.res'.format(ifg)) for ifg in ifgdates]
    else:
        res_list = glob.glob(os.path.join(resdir, '*.res'))
    if len(res_list) == 0:
        sys.exit('No ifgs for correcting...\nCheck if there are *res files in the directory {}'.format(resdir))

    if args.mask_by_residual:
        pass
    elif args.correction_by_integer:
        pass
    elif args.correction_by_mode:
        pass
    else:
        # read threshold value for automatic decision
        resid_threshold_file = os.path.join(infodir, '131resid_2pi{}.txt'.format(args.suffix))
        if args.correction_thresh:
            correction_thresh = args.correction_thresh
            target_thresh = correction_thresh
        elif os.path.exists(resid_threshold_file):
            correction_thresh = float(io_lib.get_param_par(resid_threshold_file, 'RMS_thresh'))
            target_thresh = float(io_lib.get_param_par(resid_threshold_file, 'RMS_'+args.target_thresh))
        else:
            raise Exception("No input threshold or info/131resid_2pi*.txt file, quit...")

        print("Correction threshold = {:.2f}".format(correction_thresh))
        print("Target threshold = {:.2f}".format(target_thresh))


def perform_correction(ifg_list=None):
    global bad_ifg_not_corrected, ifg_corrected_by_mode, ifg_corrected_by_integer, good_ifg, res_list
    # keep bad_ifg_not_corrected empty at the start of each correction iteration
    bad_ifg_not_corrected = []

    # automatic correction
    if ifg_list:
        res_list = [os.path.join(resdir, x+'.res') for x in ifg_list]

    # parallel processing
    if n_para > 1 and len(res_list) > 100:
        pool = multi.Pool(processes=n_para)
        results = pool.map(correction_decision, even_split(res_list, n_para))
        # compile results from different parallel processes
        for bad_list, mode_list, int_list, good_list in results:
            bad_ifg_not_corrected.extend(bad_list)
            ifg_corrected_by_mode.extend(mode_list)
            ifg_corrected_by_integer.extend(int_list)
            good_ifg.extend(good_list)
    else:
        bad_list, mode_list, int_list, good_list = correction_decision(res_list)
        bad_ifg_not_corrected.extend(bad_list)
        ifg_corrected_by_mode.extend(mode_list)
        ifg_corrected_by_integer.extend(int_list)
        good_ifg.extend(good_list)


def even_split(a, n):
    """ Divide a list, a, in to n even parts"""
    n = min(n, len(a)) # to avoid empty lists
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def load_res(res_file, length, width):
    res_mm = np.fromfile(res_file, dtype=np.float32).reshape((length, width))
    res_rad = res_mm / coef_r2m
    res_num_2pi = res_rad / 2 / np.pi
    if not args.no_depeak:
        counts, bins = np.histogram(res_num_2pi, np.arange(-2.5, 2.6, 0.1))
        peak = bins[counts.argmax()] + 0.05
        res_num_2pi = res_num_2pi - peak
    res_rms = np.sqrt(np.nanmean(res_num_2pi ** 2))
    del res_mm, res_rad
    return res_num_2pi, res_rms


def correction_decision(res_list):
    # # set up
    bad_list = []
    mode_list = []
    int_list = []
    good_list = []

    for i in res_list:
        # read input res
        pair = os.path.basename(i).split('.')[0][-17:]
        print(pair)
        res_num_2pi, res_rms = load_res(i, length, width)

        # 1. good
        if res_rms < correction_thresh:
            good_list.append(pair)
            print("RMS residual = {:.2f}, good...".format(res_rms))

            # load unw
            unwfile = os.path.join(unwdir, pair, pair + '.unw')
            unw = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))

            # apply masking
            mask = abs(res_num_2pi) > args.mask_thresh
            res_num_2pi_masked = copy.copy(res_num_2pi)
            res_num_2pi_masked[mask] = np.nan
            unw_masked = copy.copy(unw)
            unw_masked[mask] = np.nan

            # plot good with masking
            png_file = good_png_dir + '{}.png'.format(pair)
            plot_masking(pair, unw, unw_masked, res_num_2pi, res_num_2pi_masked, png_file)

            # define output dir
            correct_pair_dir = os.path.join(correct_dir, pair)
            Path(correct_pair_dir).mkdir(parents=True, exist_ok=True)

            # output file
            masked_unwfile = os.path.join(correct_pair_dir, pair + '.unw')
            unw_masked.flatten().tofile(masked_unwfile)

            del res_num_2pi, res_rms, res_num_2pi_masked, mask, unw_masked

        else:
            print("RMS residual = {:2f}, not good...".format(res_rms))
            res_integer = np.round(res_num_2pi)
            rms_res_integer_corrected = np.sqrt(np.nanmean((res_num_2pi - res_integer) ** 2))

            # 2. bad
            if rms_res_integer_corrected > target_thresh:
                bad_list.append(pair)
                print("Integer reduces rms residuals to {:.2f}, still above threshold of {:.2f}, discard...".format(
                    rms_res_integer_corrected, target_thresh))
                # plot bad res
                plot_bad_res(pair, res_num_2pi, res_integer, res_rms)

                del res_num_2pi, res_rms, res_integer, rms_res_integer_corrected

            # 3. correction
            else:
                # read in unwrapped ifg and connected components
                unwfile = os.path.join(unwdir, pair, pair + '.unw')
                con_file = os.path.join(ccdir, pair, pair + '.conncomp')
                unw = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))
                con = np.fromfile(con_file, dtype=np.int8).reshape((length, width))

                # turn uncertain correction into masking
                mask1 = np.logical_and(abs(res_num_2pi) > 0.2, abs(res_num_2pi) < 0.8)
                mask2 = np.logical_and(abs(res_num_2pi) > 1.2, abs(res_num_2pi) < 1.8)
                mask = np.logical_or(mask1, mask2)
                masked_res_integer = copy.copy(res_integer)
                masked_res_integer[mask] = np.nan

                # calc masked mode correction
                res_mode, rms_res_mode_corrected = calc_component_mode(con, masked_res_integer, res_num_2pi)

                # 3.1 correct by component mode
                if rms_res_mode_corrected < target_thresh:
                    print(
                        "Component modes reduces rms residuals to {:.2f}, below target threshold of {:.2f}, correcting by component mode...".format(
                            rms_res_mode_corrected, target_thresh))
                    unw_corrected = unw - res_mode * 2 * np.pi

                    # plotting
                    mode_list.append(pair)
                    png_path = os.path.join(mode_png_dir, '{}.png'.format(pair))
                    plot_correction_by_mode(pair, unw, con, unw_corrected, res_num_2pi, res_integer, res_mode, res_rms,
                                            rms_res_integer_corrected, rms_res_mode_corrected, png_path)

                # 3.2. correct by masked nearest integer
                else:
                    print("Component modes reduces rms residuals to {:.2f}, above threshold of {:.2f}...".format(
                        rms_res_mode_corrected, target_thresh))
                    print("Integer reduces rms residuals to {:.2f}, correcting by nearest integer...".format(
                        rms_res_integer_corrected))

                    # apply masked correction
                    unmasked_unw_corrected = unw - res_integer * 2 * np.pi
                    unw_corrected = unw - masked_res_integer * 2 * np.pi
                    rms_res_mask_corrected = np.sqrt(np.nanmean((res_num_2pi - masked_res_integer) ** 2))

                    # plotting
                    int_list.append(pair)
                    png_path = os.path.join(integer_png_dir, '{}.png'.format(pair))
                    plot_correction_by_integer(pair, unw, unmasked_unw_corrected, unw_corrected, masked_res_integer, res_num_2pi,
                                                   res_integer, res_rms, rms_res_integer_corrected,
                                                   rms_res_mask_corrected, png_path)

                    del mask1, mask2, mask, masked_res_integer, unmasked_unw_corrected, rms_res_mask_corrected

                # define output dir
                correct_pair_dir = os.path.join(correct_dir, pair)
                Path(correct_pair_dir).mkdir(parents=True, exist_ok=True)

                # save the corrected unw
                unw_corrected.flatten().tofile(os.path.join(correct_pair_dir, pair + '.unw'))
                del con, unw, unw_corrected, res_num_2pi, res_integer, res_rms

    return bad_list, mode_list, int_list, good_list


def plot_good_res(pair, res_num_2pi, res_rms):
    ## plot_res
    plt.imshow(res_num_2pi, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    plt.title(pair + " RMS_res={:.2f}".format(res_rms))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(good_png_dir + '{}.png'.format(pair), dpi=300, bbox_inches='tight')
    plt.close()


def plot_bad_res(pair, res_num_2pi, res_integer, res_rms):
    fig, ax = plt.subplots(1, 2, figsize=(9, 6))
    fig.suptitle(pair)
    for x in ax:
        x.axes.xaxis.set_ticklabels([])
        x.axes.yaxis.set_ticklabels([])
    im_res = ax[0].imshow(res_num_2pi, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1].imshow(res_integer, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    ax[0].scatter(ref_x, ref_y, c='r', s=10)
    ax[0].set_title("Residual/2pi (RMS={:.2f})".format(res_rms))
    ax[1].set_title("Nearest integer")
    plt.colorbar(im_res, ax=ax, location='right', shrink=0.8)
    plt.savefig(bad_png_dir + '{}.png'.format(pair), dpi=300, bbox_inches='tight')
    plt.close()


def plot_correction_by_mode(pair, unw, con, unw_corrected, res_num_2pi, res_integer, res_mode, res_rms, rms_res_integer_corrected, rms_res_mode_corrected, png_path):
    fig, ax = plt.subplots(2, 3, figsize=(9, 5))
    fig.suptitle(pair)
    for x in ax[:, :].flatten():
        x.axes.xaxis.set_ticklabels([])
        x.axes.yaxis.set_ticklabels([])
    unw_vmin = np.nanpercentile(unw, 0.5)
    unw_vmax = np.nanpercentile(unw, 99.5)
    im_con = ax[0, 0].imshow(con, cmap=cm.tab10, interpolation='nearest')
    im_unw = ax[0, 1].imshow(unw, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu, interpolation='nearest')
    im_unw = ax[0, 2].imshow(unw_corrected, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 0].imshow(res_num_2pi, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 1].imshow(res_integer, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 2].imshow(res_mode, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    ax[1, 0].scatter(ref_x, ref_y, c='r', s=10)
    ax[0, 0].set_title("Components")
    ax[0, 1].set_title("Unw (rad)")
    ax[0, 2].set_title("Mode Corrected")
    ax[1, 0].set_title("Residual/2pi (RMS={:.2f})".format(res_rms))
    ax[1, 1].set_title("Nearest integer (to {:.2f})".format(rms_res_integer_corrected))
    ax[1, 2].set_title("Component mode (to {:.2f})".format(rms_res_mode_corrected))
    fig.colorbar(im_unw, ax=ax[0, :], location='right', shrink=0.8)
    fig.colorbar(im_res, ax=ax[1, :], location='right', shrink=0.8)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_correction_by_integer(pair, unw, unw_corrected, unw_masked, res_mask, res_num_2pi, res_integer, res_rms, rms_res_integer_corrected, rms_res_mask_corrected, png_path):
    fig, ax = plt.subplots(2, 3, figsize=(9, 5))
    fig.suptitle(pair)
    for x in ax[:, :].flatten():
        x.axes.xaxis.set_ticklabels([])
        x.axes.yaxis.set_ticklabels([])
    unw_vmin = np.nanpercentile(unw, 0.5)
    unw_vmax = np.nanpercentile(unw, 99.5)
    im_unw = ax[0, 0].imshow(unw, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu, interpolation='nearest')
    im_unw = ax[0, 1].imshow(unw_corrected, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu, interpolation='nearest')
    im_unw = ax[0, 2].imshow(unw_masked, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 0].imshow(res_num_2pi, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 1].imshow(res_integer, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 2].imshow(res_mask, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    ax[1, 0].scatter(ref_x, ref_y, c='r', s=10)
    ax[0, 0].set_title("Unw (rad)")
    ax[0, 1].set_title("Correct by Integer")
    ax[0, 2].set_title("Correct with Mask")
    ax[1, 0].set_title("Residual/2pi (RMS={:.2f})".format(res_rms))
    ax[1, 1].set_title("Nearest Integer ({:.2f})".format(rms_res_integer_corrected))
    ax[1, 2].set_title("Masked Integer ({:.2f})".format(rms_res_mask_corrected))
    fig.colorbar(im_unw, ax=ax[0, :], location='right', shrink=0.8)
    fig.colorbar(im_res, ax=ax[1, :], location='right', shrink=0.8)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_lists():

    #%% save ifg lists to text files.
    bad_ifg_file = os.path.join(infodir, '132bad_ifg{}_{:.2f}_{:.2f}.txt'.format(args.suffix, correction_thresh, target_thresh))
    if os.path.exists(bad_ifg_file): os.remove(bad_ifg_file)
    with open(bad_ifg_file, 'w') as f:
        for i in bad_ifg_not_corrected:
            print('{}'.format(i), file=f)

    mode_ifg_file = os.path.join(infodir, '132corrected_by_component_mode_ifg{}_{:.2f}_{:.2f}.txt'.format(args.suffix, correction_thresh, target_thresh))
    if os.path.exists(mode_ifg_file): os.remove(mode_ifg_file)
    with open(mode_ifg_file, 'w') as f:
        for i in ifg_corrected_by_mode:
            print('{}'.format(i), file=f)

    nearest_ifg_file = os.path.join(infodir, '132corrected_by_nearest_integer_ifg{}_{:.2f}_{:.2f}.txt'.format(args.suffix, correction_thresh, target_thresh))
    if os.path.exists(nearest_ifg_file): os.remove(nearest_ifg_file)
    with open(nearest_ifg_file, 'w') as f:
        for i in ifg_corrected_by_integer:
            print('{}'.format(i), file=f)

    good_ifg_file = os.path.join(infodir, '132good_ifg_uncorrected{}_{:.2f}_{:.2f}.txt'.format(args.suffix, correction_thresh, target_thresh))
    if os.path.exists(good_ifg_file): os.remove(good_ifg_file)
    with open(good_ifg_file, 'w') as f:
        for i in good_ifg:
            print('{}'.format(i), file=f)


def plot_networks():
    """ plot networks with and without corrected ifgs, identify weak links, return n_gap if removing weak links"""
    ### Read date, network information and size
    retained_ifgs = good_ifg + ifg_corrected_by_mode + ifg_corrected_by_integer
    corrected_ifgs = ifg_corrected_by_mode + ifg_corrected_by_integer
    retained_ifgs.sort()
    corrected_ifgs.sort()

    if len(retained_ifgs) == 0 :
        n_gap = 1
        strong_links = [] # dummy
        weak_links = [] # dummy
    else:
        # strong_links, weak_links = tools_lib.separate_strong_and_weak_links(retained_ifgs)
        component_stats_file = os.path.join(infodir, 'network132_component_stats{}_{:.2f}_{:.2f}.txt'.format(args.suffix, correction_thresh, target_thresh))
        largest_component_file = os.path.join(infodir, 'network132_largest_component_links{}_{:.2f}_{:.2f}.txt'.format(args.suffix, correction_thresh, target_thresh))
        strong_links, weak_links, edge_cuts, node_cuts = tools_lib.separate_strong_and_weak_links(retained_ifgs, component_stats_file)
        if os.path.exists(largest_component_file): os.remove(largest_component_file)
        with open(largest_component_file, 'w') as f:
            for i in strong_links:
                print('{}'.format(i), file=f)

        if len(strong_links) == 0:
            n_gap = 1
            strong_links = []  # dummy
            weak_links = []  # dummy
        else:
            imdates = tools_lib.ifgdates2imdates(retained_ifgs)
            n_im = len(imdates)

            ### Plot network
            ## Read bperp data or dummy
            bperp_file = os.path.join(ccdir, 'baselines')
            if os.path.exists(bperp_file):
                bperp = io_lib.read_bperp_file(bperp_file, imdates)
            else: #dummy
                bperp = np.random.random(n_im).tolist()

            pngfile = os.path.join(netdir, 'network132_only_good_without_correction{}_{:.2f}_{:.2f}.png'.format(args.suffix, correction_thresh, target_thresh))
            plot_lib.plot_corrected_network(retained_ifgs, bperp, corrected_ifgs, pngfile, plot_corrected=False)

            pngfile = os.path.join(netdir, 'network132_with_corrected{}_{:.2f}_{:.2f}.png'.format(args.suffix, correction_thresh, target_thresh))
            plot_lib.plot_corrected_network(retained_ifgs, bperp, corrected_ifgs, pngfile)

            pngfile = os.path.join(netdir, 'network132_all_retained{}_{:.2f}_{:.2f}.png'.format(args.suffix, correction_thresh, target_thresh))
            plot_lib.plot_network(retained_ifgs, bperp, weak_links, pngfile, plot_bad=True, label_name='Weak Links')
    return n_im, strong_links, weak_links


def correction_main():
    global bad_ifg_not_corrected, ifg_corrected_by_mode, ifg_corrected_by_integer, good_ifg  #correction_thresh, target_thresh,

    # set up empty decision lists
    ifg_corrected_by_mode = []
    ifg_corrected_by_integer = []
    good_ifg = []
    bad_ifg_not_corrected = []

    perform_correction()
    save_lists()
    plot_networks()

    # n_im, strong_links, weak_links =
    # while n_gap > 0:  # loosen correction and target thresholds until the network has no gap even after removing weak links
    #     print("n_gap=" + str(n_gap)+", increase correction_thresh and target_thresh by 0.05")
    #     correction_thresh += 0.05
    #     target_thresh += 0.05
    #
    #     print("Correction_thres = {}".format(correction_thresh))
    #     print("Target_thres = {}".format(target_thresh))
    #     print("Consider correcting {} bad ifgs".format(len(bad_ifg_not_corrected)))
    #
    #     if len(bad_ifg_not_corrected) == 0:
    #         sys.exit("No more bad ifgs for correcting...")
    #
    #     perform_correction(bad_ifg_not_corrected)
    #     save_lists()
    #     n_gap, strong_links, weak_links = plot_networks()
    #
    # if args.move_weak:
    #     # move weak ifgs to subfolder
    #     weak_ifg_dir = os.path.join(correct_dir, "weak_links")
    #     Path(weak_ifg_dir).mkdir(parents=True, exist_ok=True)
    #     for pair in weak_links:
    #         unw_folder = os.path.join(correct_dir, pair)
    #         dest_folder = os.path.join(weak_ifg_dir, pair)
    #         shutil.move(unw_folder, dest_folder)


def perform_masking():

    # multi-processing
    if n_para > 1 and len(res_list) > 20:
        pool = multi.Pool(processes=n_para)
        results = pool.map(masking, even_split(res_list, n_para))
        perc_list = np.concatenate(results)
    else:
        perc_list = masking(res_list)
    return perc_list


def masking(res_list):
    unw_perc_list = []
    for i in res_list:
        pair = os.path.basename(i).split('.')[0][-17:]
        print(pair)
        unwfile = os.path.join(unwdir, pair, pair + '.unw')
        unw = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))

        # count coherence pixels for expected total n_unw
        ccfile = os.path.join(ccdir, pair, pair + '.cc')
        coh = io_lib.read_img(ccfile, length, width, np.uint8)
        coh = coh.astype(np.float32)
        coh[coh==0] = np.nan
        cc_pixel_count = np.count_nonzero(~np.isnan(coh))

        # generate a mask using res, and make a masked_residual map (for plotting only)
        res_num_2pi, res_rms = load_res(i, length, width)
        mask = abs(res_num_2pi) > args.mask_thresh
        res_num_2pi_masked = copy.copy(res_num_2pi)
        res_num_2pi_masked[mask] = np.nan

        # mask unw and calculate percentage unwrap
        unw_masked = copy.copy(unw)
        unw_masked[mask] = np.nan
        unw_masked_pixel_count = np.count_nonzero(~np.isnan(unw_masked))
        unw_percentage = unw_masked_pixel_count/cc_pixel_count * 100
        unw_perc_list.append(unw_percentage)

        # plot masking
        png_file = os.path.join(mask_png_dir, '{}.png'.format(pair))
        plot_masking(pair, unw, unw_masked, res_num_2pi, res_num_2pi_masked, png_file)

        # output masked unw
        mask_pair_dir = os.path.join(mask_dir, pair)
        Path(mask_pair_dir).mkdir(parents=True, exist_ok=True)
        unw_masked.flatten().tofile(os.path.join(mask_pair_dir, pair + '.unw'))

        del unw, unw_masked, res_num_2pi, mask, res_num_2pi_masked
    return unw_perc_list


def plot_masking(pair, unw, unw_masked, res_num_2pi, res_num_2pi_masked, png):
    # plotting
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    fig.suptitle(pair)
    for x in ax.flatten():
        x.axes.xaxis.set_ticklabels([])
        x.axes.yaxis.set_ticklabels([])

    unw_vmin = np.nanpercentile(unw, 0.5)
    unw_vmax = np.nanpercentile(unw, 99.5)

    im_unw = ax[0, 0].imshow(unw, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu, interpolation='nearest')
    im_unw = ax[0, 1].imshow(unw_masked, vmin=unw_vmin, vmax=unw_vmax, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 0].imshow(res_num_2pi, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    im_res = ax[1, 1].imshow(res_num_2pi_masked, vmin=-2, vmax=2, cmap=cm.RdBu, interpolation='nearest')
    ax[1, 0].scatter(ref_x, ref_y, c='r', s=10)
    ax[1, 1].scatter(ref_x, ref_y, c='r', s=10)

    ax[0, 0].set_title("Unw (rad)")
    ax[0, 1].set_title("Unw_masked")
    ax[1, 0].set_title("Residual in 2pi")
    ax[1, 1].set_title("|Residual| < {:.2f}".format(args.mask_thresh))

    fig.colorbar(im_unw, ax=ax[0, :], location='right', shrink=0.8)
    fig.colorbar(im_res, ax=ax[1, :], location='right', shrink=0.8)

    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.close()


def plot_network_with_unw_perc(perc_list):
    # plot coloured networks
    ifgdates = tools_lib.get_ifgdates(args.out_dir)
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    bperp_file = os.path.join(ccdir, 'baselines')
    if os.path.exists(bperp_file):
        bperp = io_lib.read_bperp_file(bperp_file, imdates)
    else:  # dummy
        n_im = len(imdates)
        bperp = np.random.random(n_im).tolist()
    pngfile = os.path.join(netdir, 'network132_masked{}_{}.png'.format(args.suffix, args.mask_thresh))
    plot_lib.plot_coloured_network(ifgdates, bperp, perc_list, pngfile)


def mode_correction():
    # parallel processing
    if n_para > 1 and len(res_list) > 20:
        pool = multi.Pool(processes=n_para)
        pool.map(correcting_by_mode, even_split(res_list, n_para))
    else:
        correcting_by_mode(res_list)

def integer_correction():
    # parallel processing
    if n_para > 1 and len(res_list) > 20:
        pool = multi.Pool(processes=n_para)
        pool.map(correcting_by_integer, even_split(res_list, n_para))
    else:
        correcting_by_integer(res_list)


def calc_component_mode(con, res_integer, res_num_2pi):
    # calculate component modes
    uniq_components = np.unique(con.flatten())[1:]  # use [1:] to exclude the 0th component
    res_mode = copy.copy(res_integer)
    for j in uniq_components:
        component_values = res_integer[con == j]
        int_values = component_values[~np.isnan(component_values)].astype(int)
        mode = stats.mode(int_values)[0][0]
        res_mode[con == j] = mode
    rms_res_mode_corrected = np.sqrt(np.nanmean((res_num_2pi - res_mode) ** 2))
    return res_mode, rms_res_mode_corrected


def correcting_by_mode(reslist):
    for i in reslist:
        pair = os.path.basename(i).split('.')[0][-17:]
        print(pair)
        res_num_2pi, res_rms = load_res(i, length, width)
        res_integer = np.round(res_num_2pi)
        rms_res_integer_corrected = np.sqrt(np.nanmean((res_num_2pi - res_integer) ** 2))

        # calc component mode
        unwfile = os.path.join(unwdir, pair, pair + '.unw')
        con_file = os.path.join(ccdir, pair, pair + '.conncomp')
        unw = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))
        con = np.fromfile(con_file, dtype=np.int8).reshape((length, width))
        res_mode, rms_res_mode_corrected = calc_component_mode(con, res_integer, res_num_2pi)

        # correcting by component mode
        unw_corrected = unw - res_mode * 2 * np.pi

        # plotting
        png_path = os.path.join(mode_png_dir, '{}.png'.format(pair))
        plot_correction_by_mode(pair, unw, con, unw_corrected, res_num_2pi, res_integer, res_mode,
                        res_rms, rms_res_integer_corrected, rms_res_mode_corrected, png_path)

        # define output dir
        correct_pair_dir = os.path.join(correct_dir, pair)
        Path(correct_pair_dir).mkdir(parents=True, exist_ok=True)

        # save the corrected unw
        unw_corrected.flatten().tofile(os.path.join(correct_pair_dir, pair + '.unw'))
        del con, unw, unw_corrected, res_num_2pi, res_integer, res_rms


def correcting_by_integer(reslist):
    for i in reslist:
        pair = os.path.basename(i).split('.')[0][-17:]
        print(pair)
        res_num_2pi, res_rms = load_res(i, length, width)
        res_integer = np.round(res_num_2pi)
        rms_res_integer_corrected = np.sqrt(np.nanmean((res_num_2pi - res_integer) ** 2))

        # calc component mode
        unwfile = os.path.join(unwdir, pair, pair + '.unw')
        unw = np.fromfile(unwfile, dtype=np.float32).reshape((length, width))

        # correcting by component mode
        unw_corrected = unw - res_integer * 2 * np.pi

        # turn uncertain correction into masking
        mask1 = np.logical_and(abs(res_num_2pi) > 0.2, abs(res_num_2pi) < 0.8)
        mask2 = np.logical_and(abs(res_num_2pi) > 1.2, abs(res_num_2pi) < 1.8)
        mask = np.logical_or(mask1, mask2)
        res_mask = copy.copy(res_integer)
        res_mask[mask] = np.nan
        unw_masked = unw - res_mask * 2 * np.pi
        rms_res_mask_corrected = np.sqrt(np.nanmean((res_num_2pi - res_mask) ** 2))

        # plotting
        png_path = os.path.join(integer_png_dir, '{}.png'.format(pair))
        plot_correction_by_integer(pair, unw, unw_corrected, unw_masked, res_mask, res_num_2pi, res_integer, res_rms,
                                   rms_res_integer_corrected, rms_res_mask_corrected, png_path)

        # define output dir
        correct_pair_dir = os.path.join(correct_dir, pair)
        Path(correct_pair_dir).mkdir(parents=True, exist_ok=True)

        # save the corrected unw
        unw_masked.flatten().tofile(os.path.join(correct_pair_dir, pair + '.unw'))
        del mask1, mask2, mask, res_mask, unw_masked, rms_res_mask_corrected


def best_network(all_ifgs, all_resids):
    global target_thresh
    # target_thresh = 0.2
    n_gap = 1

    while n_gap > 0:  # loosen correction and target thresholds until the network has no gap even after removing weak links
        ifgs = [i for i, r in zip(all_ifgs, all_resids) if r < target_thresh]
        # strong_links, weak_links = tools_lib.separate_strong_and_weak_links(ifgs)
        # component_stats_file = os.path.join(infodir, 'network132_component_stats{}_{:.2f}_{:.2f}.txt'.format(args.suffix, correction_thresh, target_thresh))
        # strong_links, weak_links, edge_cuts, node_cuts = tools_lib.separate_strong_and_weak_links(ifgs, component_stats_file)

        print("target_thresh = {}".format(target_thresh))
        # print("{} ifgs are well-connected".format(len(strong_links)))
        # print("{} ifgs are weak links".format(len(weak_links)))
        ### Identify gaps
        G = inv_lib.make_sb_matrix(ifgs)
        ixs_inc_gap = np.where(G.sum(axis=0) == 0)[0]
        n_gap = len(ixs_inc_gap)
        print("n_gap={}".format(int(n_gap)))
        target_thresh += 0.01
    target_thresh -= 0.01

    ### Plot network
    ## Read bperp data or dummy
    imdates = tools_lib.ifgdates2imdates(ifgs)
    bperp_file = os.path.join(ccdir, 'baselines')
    if os.path.exists(bperp_file):
        bperp = io_lib.read_bperp_file(bperp_file, imdates)
    else:  # dummy
        n_im = len(imdates)
        bperp = np.random.random(n_im).tolist()

    # pngfile = os.path.join(netdir, 'network132_best_network_all{}_{:.2f}.png'.format(args.suffix, target_thresh))
    # plot_lib.plot_corrected_network(ifgs, bperp, weak_links, pngfile, plot_corrected=True, label_name='Weak Links')
    # pngfile = os.path.join(netdir, 'network132_best_network_strong{}_{:.2f}.png'.format(args.suffix, target_thresh))
    # plot_lib.plot_network(ifgs, bperp, weak_links, pngfile, plot_bad=False)
    pngfile = os.path.join(netdir, 'network132_best_network_strong{}_{:.2f}.png'.format(args.suffix, target_thresh))
    plot_lib.plot_network(ifgs, bperp, [], pngfile, plot_bad=False)

    #%% save ifg lists to text files.
    best_network_ifg_file = os.path.join(infodir, '132best_network_ifg{}_{:.2f}.txt'.format(args.suffix, target_thresh))
    if os.path.exists(best_network_ifg_file): os.remove(best_network_ifg_file)
    with open(best_network_ifg_file, 'w') as f:
        for i in ifgs:
            print('{}'.format(i), file=f)


def main():
    start()
    init_args()
    set_input_output()
    get_para()

    if args.mask_by_residual:
        perc_list = perform_masking()
        plot_network_with_unw_perc(perc_list)
    elif args.correction_by_mode:
        mode_correction()
    elif args.correction_by_integer:
        integer_correction()
    elif args.best_network:
        all_ifgs, all_resids = io_lib.read_residual_file(resid_threshold_file)
        best_network(all_ifgs, all_resids)
    else:
        correction_main()

    finish()


if __name__ == "__main__":
    main()