#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import glob
import os
import sys
import argparse
from osgeo import gdal
from scipy import stats
from pathlib import Path


class OpenTif:
    """ a Class that stores the band array and metadata of a Gtiff file."""
    def __init__(self, filename):
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


if __name__ == "__main__":

    # parse frame name as argument
    parser = argparse.ArgumentParser(description="Detect coregistration error")
    parser.add_argument("frame", help="frame name")
    args = parser.parse_args()
    frame = args.frame

    track = str(int(frame[0:3]))
    input_dir = "/gws/nopw/j04/nceo_geohazards_vol1/public/LiCSAR_products/{}/{}/interferograms/".format(track, frame)
    output_dir = frame
    Path(frame).mkdir(parents=True, exist_ok=True)
    Path(frame+"/png").mkdir(parents=True, exist_ok=True)
    stats_file = "{}/{}_middle_column_stats.txt".format(frame, frame)
    if os.path.exists(stats_file):
        os.remove(stats_file)

    with open(stats_file, "a") as f:
        f.write("Frame Pair Slope R_squares Abs_slope\n")
        for unw in glob.glob(input_dir+"*/*.geo.unw.tif"):
            tif=OpenTif(unw)
            pair = unw.split('/')[-1][:17]
            print(pair, frame)

            fig, ax=plt.subplots(1,2, figsize=(6, 3))
            ax[0].imshow(tif.data, vmin = np.nanpercentile(tif.data, 0.5), vmax = np.nanpercentile(tif.data, 99.5))
            ax[0].plot([tif.xsize // 2,tif.xsize // 2], [0, tif.ysize+1])
            ax[0].set_title(pair)

            middle_column = tif.data[:, tif.xsize // 2]
            middle_column_latitudes = tif.lat[:, tif.xsize // 2]
            ax[1].plot(middle_column_latitudes, middle_column)
            non_nan_mask = ~np.isnan(middle_column)
            slope, intercept, r_value, p_value, std_err = stats.linregress(middle_column_latitudes[non_nan_mask], middle_column[non_nan_mask])

            ax[1].annotate("Phase = {:.2f} latitude + {:.2f} \n$R^2$ = {:.3f}".format(slope, intercept, r_value**2), xy=(0.05, 0.965),
                            xycoords='axes fraction', ha='left', va='top', zorder=10)
            ax[1].plot(middle_column_latitudes, middle_column_latitudes*slope+intercept)
            ax[1].set_xlabel("Latitude")
            ax[1].set_ylabel("Phase / mm")
            ax[1].set_title("Middle Column")
            plt.tight_layout()
            plt.savefig("{}/png/{}_middle_column_{}.png".format(frame, frame, pair), dpi=500)
            plt.close()

            del tif, middle_column, middle_column_latitudes, non_nan_mask

            f.write("{} {} {} {} {}\n".format(frame, pair, slope, r_value**2, abs(slope)))
    f.close()

    # read text file into pandas DataFrame
    df = pd.read_csv("{}/{}_middle_column_stats.txt".format(frame, frame), sep=" ")

    plt.hist2d(df["Abs_slope"], df["R_squares"], bins=20, cmin=1)
    plt.title(frame)
    plt.xlabel("|slope|")
    plt.ylabel("R^2")
    plt.xlim(0,100)
    plt.ylim(0,1)
    plt.savefig("{}/{}_middle_column_hist2d.png".format(frame, frame))
    plt.close()

    df['Epoch1'] = df['Pair'].str[:8]
    df['date1'] = pd.to_datetime(df['Epoch1'], format='%Y%m%d')
    fig, axes = plt.subplots(2)
    df.plot.scatter(x='date1', y='Abs_slope', c='R_squares', colormap='viridis', ax=axes[0])
    df.plot.scatter(x='date1', y='R_squares', c='Abs_slope', colormap='viridis', ax=axes[1])
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, 1)
    axes[0].set_xlabel("")
    plt.suptitle(frame)
    plt.savefig("{}/{}_scatter.png".format(frame, frame))
    plt.close()