#!/usr/bin/env python3
"""
========
Overview
========
This script
(1) calculates a histogram of each residual map converted into factors of 2pi radian,
(2) offsets the residual map by the histogram peak to remove any bias from referencing effect
(3) calculate and saves the RMS of the de-peaked residual as multiples of 2pi [.txt]
(4) plots and saves a histogram[.png] of all RMS of all ifgs
(5) sets a threshold [.txt] for automatic correction with 132_3D_correction.py

===============
Input & output files
===============

Inputs in GEOCml*/ :
 - slc.mli.par

Inputs in TS_GEOCml*/ :
 - 13resid/
   - yyyymmdd_yyyymmdd.res

Outputs in TS_GEOCml*/ :
 - info/
   - 131resid_2pi{suffix}.txt        : RMS of the de-peaked residuals as factors of 2pi radian
   - 131RMS_ifg_res_hist{suffix}.png : plot of histogram with a vertical bar indicating threshold

=====
Usage
=====
LiCSBAS131_residual_threshold.py [-h] [-f FRAME_DIR] [-g UNW_DIR] [-t TS_DIR]
                                        [-p PERCENTILE] [--suffix SUFFIX]
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
import sys
import time
import LiCSBAS_io_lib as io_lib


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
    parser.add_argument('-c', dest='cc_dir', default="GEOCml10GACOS", help="folder containing slc.mli.par")
    parser.add_argument('-t', dest='ts_dir', default="TS_GEOCml10GACOS", help="folder containing time series")
    parser.add_argument('-r', dest='thresh', type=float, help="user specified threshold value, otherwise auto-detected")
    parser.add_argument('-p', dest='percentile', type=float, help="optional percentile RMS for thresholding")
    parser.add_argument('--suffix', default="", type=str, help="suffix of both input and output")
    parser.add_argument('--no_depeak', default=False, action='store_true', help="don't offset by mode (recommend depeak)")
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
    print("\n{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)
    print('Output directory: {}\n'.format(os.path.relpath(tsadir)))


def set_input_output():
    global unwdir, tsadir, resdir, infodir, hist_png, restxtfile

    # define input directories
    unwdir = os.path.abspath(os.path.join(args.frame_dir, args.cc_dir))
    tsadir = os.path.abspath(os.path.join(args.frame_dir, args.ts_dir))
    resdir = os.path.join(tsadir, '130resid'+args.suffix)

    # define output directory and files
    infodir = os.path.join(tsadir, 'info')
    hist_png = os.path.join(infodir, "131RMS_ifg_res_hist{}.png".format(args.suffix))
    restxtfile = os.path.join(infodir, '131resid_2pi{}.txt'.format(args.suffix))
    if os.path.exists(restxtfile): os.remove(restxtfile)


def get_para():
    global width, length, coef_r2m

    # read ifg size and satellite frequency
    mlipar = os.path.join(unwdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    radar_frequency = float(io_lib.get_param_par(mlipar, 'radar_frequency'))  # 5405000000.0 Hz for C-band
    speed_of_light = 299792458  # m/s
    wavelength = speed_of_light/radar_frequency
    coef_r2m = -wavelength/4/np.pi*1000


def plot_histogram_of_rms_of_depeaked_residuals():
    print('Reading residual maps from {}'.format(resdir))
    with open(restxtfile, "w") as f:
        print('# RMS of residual (in number of 2pi)', file=f)

        # calc rms of de-peaked residuals
        res_rms_list = []
        for i in glob.glob(os.path.join(resdir, '*.res')):
            pair = os.path.basename(i).split('.')[0][-17:]
            print(pair)
            res_mm = np.fromfile(i, dtype=np.float32)
            res_rad = res_mm / coef_r2m
            res_num_2pi = res_rad / 2 / np.pi
            if not args.no_depeak:
                counts, bins = np.histogram(res_num_2pi, np.arange(-2.5, 2.6, 0.1))
                peak = bins[counts.argmax()] + 0.05
                res_num_2pi = res_num_2pi - peak
            res_rms = np.sqrt(np.nanmean(res_num_2pi ** 2))
            res_rms_list.append(res_rms)

            print('{} {:5.2f}'.format(pair, res_rms), file=f)

        # plotting histogram and peak and threshold vertical lines
        count_ifg_res_rms, bin_edges, patches = plt.hist(res_rms_list, np.arange(0, 3, 0.1))
        peak_ifg_res_rms = bin_edges[count_ifg_res_rms.argmax()] + 0.05  # nanmode
        # plotting median and mean to illustrate skewedness of the distribution. mode<median<mean = right skewed
        median = np.nanpercentile(res_rms_list, 50)
        mean = np.nanmean(res_rms_list)
        plt.axvline(x=peak_ifg_res_rms, color='r', linestyle=':', label="mode = {:.2f}".format(peak_ifg_res_rms))
        plt.axvline(x=median, color='r', linestyle='--', label="median = {:.2f}".format(median))
        plt.axvline(x=mean, color='r', linestyle='-', label="mean = {:.2f}".format(mean))

        # auto selection of threshold based on mode value if neither -r (args.thresh) nor -p (args.percentile) is specified by the user
        if args.thresh:
            threshold = args.thresh
        elif args.percentile:
            threshold = np.nanpercentile(res_rms_list, args.percentile)
        elif peak_ifg_res_rms < 0.1:
            threshold = 0.2
        else:
            threshold = 2 * peak_ifg_res_rms

        plt.axvline(x=threshold, linestyle='-.', color='cyan', label="thresh = {:.2f}".format(threshold))
        plt.legend()
        plt.title("RMS Residual, threshold = {:.2f}".format(threshold))
        plt.savefig(hist_png, dpi=300)
        plt.close()

        # print to file
        print('RMS_mode: {:5.2f}'.format(peak_ifg_res_rms), file=f)
        print('RMS_median: {:5.2f}'.format(median), file=f)
        print('RMS_mean: {:5.2f}'.format(mean), file=f)
        if args.percentile:
            print('RMS_percentile: {}'.format(int(args.percentile), ), file=f)
            print('IFG RMS res, mode = {:.2f}, median = {:.2f}, mean = {:.2f}, {}% = {:.2f}'.format(peak_ifg_res_rms, median, mean, int(args.percentile), threshold))
        print('RMS_thresh: {:5.2f}'.format(threshold), file=f)
        print('IFG RMS res, mode = {:.2f}, median = {:.2f}, mean = {:.2f}, thresh = {:.2f}'.format(peak_ifg_res_rms, median, mean, threshold))


def main():
    start()
    init_args()
    set_input_output()
    get_para()
    plot_histogram_of_rms_of_depeaked_residuals()
    finish()


if __name__ == "__main__":
    main()
