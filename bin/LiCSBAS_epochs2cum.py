#!/usr/bin/env python3
"""
========
Overview
========
This script loads epoch-wide tif files into an h5 cube referenced to the first epoch, and copy over other metadata from an existing cum.h5
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
import sys
from matplotlib import cm
import SCM
import glob
from osgeo import gdal
import h5py as h5


global ver, date, author
ver = "1.0"; date = 20230605; author = "Qi Ou, ULeeds"  # generalise to any epoch


class OpenTif:
    """ a Class that stores the band array and metadata of a Gtiff file."""
    def __init__(self, filename, sigfile=None, incidence=None, heading=None, N=None, E=None, U=None):
        self.ds = gdal.Open(filename)
        self.basename = os.path.splitext(os.path.basename(filename))[0]
        self.band = self.ds.GetRasterBand(1)
        self.data = self.band.ReadAsArray()
        self.xsize = self.ds.RasterXSize
        self.ysize = self.ds.RasterYSize
        self.left = self.ds.GetGeoTransform()[0]
        self.top = self.ds.GetGeoTransform()[3]
        self.xres = self.ds.GetGeoTransform()[1]
        self.yres = self.ds.GetGeoTransform()[5]
        self.right = self.left + self.xsize * self.xres
        self.bottom = self.top + self.ysize * self.yres
        self.projection = self.ds.GetProjection()
        pix_lin, pix_col = np.indices((self.ds.RasterYSize, self.ds.RasterXSize))
        self.lat, self.lon = self.top + self.yres*pix_lin, self.left+self.xres*pix_col

        # convert 0 and 255 to NaN
        self.data[self.data==0.] = np.nan


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
    parser.add_argument('-i', dest='input_dir', default="./GACOSml10/", type=str, help="input directory containing gacos epochs")
    parser.add_argument('-s', dest='input_suffix', default=".sltd.geo.tif", type=str, help="suffix of gacos epochs")
    parser.add_argument('-o', dest='outfile', default="gacos_cum.h5", type=str, help="output cumulative displacement from gacos epochs")
    parser.add_argument('-c', dest='existing_cumh5file', default='TS_GEOCml10GACOS/cum.h5', type=str, help="cumulative displacement from LiCSBAS inversion to copy over meta data only")
    parser.add_argument('-e', dest='same_epochs_as_existing_h5', default=False, action='store_true', help="only add the same epochs as in the existing h5")
    parser.add_argument('--radian2mm', default=False, action='store_true', help="convert from radian to mm")
    args = parser.parse_args()


def start():
    global start_time
    # intialise and print info on screen
    start_time = time.time()
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
    print('Output: {}\n'.format(os.path.relpath(args.outfile)))


if __name__ == "__main__":
    init_args()
    start()

    # Open existing cum.h5 for reading metadata
    cumh5 = h5.File(args.existing_cumh5file, 'a')

    # if adding all available epochs into cube
    if not args.same_epochs_as_existing_h5:
        # add epochs into cube referenced to the first epoch
        tifList = sorted(glob.glob(os.path.join(args.input_dir, '*'+args.input_suffix)))
        ref_tif = OpenTif(tifList[0])
        cube = np.ones([len(tifList), ref_tif.ysize, ref_tif.xsize])
        for i, tif in enumerate(tifList):
            print(tif)
            slice = OpenTif(tif)
            cube[i, :, :] = slice.data - ref_tif.data
    else:  # keep the epochs the same as in existing h5
        ref_tif = OpenTif(glob.glob(os.path.join(args.input_dir, str(cumh5['imdates'][0])+"*"+args.input_suffix))[0])
        cube = np.ones([len(cumh5['imdates']), ref_tif.ysize, ref_tif.xsize])
        for i, tif in enumerate(cumh5['imdates']):
            print(tif)
            slice = OpenTif(glob.glob(os.path.join(args.input_dir, str(cumh5['imdates'][i])+"*"+args.input_suffix))[0])
            cube[i, :, :] = slice.data - ref_tif.data

    ### Get scaling factor
    if args.radian2mm:
        speed_of_light = 299792458 #m/s
        radar_frequency = 5405000000.0 #Hz
        wavelength = speed_of_light/radar_frequency #meter
        coef_r2m = -wavelength/4/np.pi*1000 #rad -> mm, positive is -LOS
    else:
        coef_r2m = 1

    # write into new cum.h5
    gacosh5 = h5.File(args.outfile, 'w')
    compress = 'gzip'
    gacosh5.create_dataset('cum', data=cube*coef_r2m, compression=compress)
    gacosh5.create_dataset('refarea', data=cumh5['refarea'] )
    gacosh5.create_dataset('imdates', data=cumh5['imdates'] )
    gacosh5.create_dataset('corner_lat', data=cumh5['refarea'])
    gacosh5.create_dataset('corner_lon', data=cumh5['corner_lon'])
    gacosh5.create_dataset('post_lat', data=cumh5['post_lat'])
    gacosh5.create_dataset('post_lon', data=cumh5['post_lon'])

    # close new cum.h5
    gacosh5.close()

    # close existing cum.h5
    cumh5.close()

    finish()