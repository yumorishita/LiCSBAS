#!/usr/bin/env python3
"""
v1.11.0 20210205 Yu Morishita, GSI

This script displays an image file.

=====
Usage
=====
LiCSBAS_disp_img.py -i image_file -p par_file [-c cmap] [--cmin float]
  [--cmax float] [--auto_crange float] [--n_color int] [--cycle float]
  [--nodata float] [--bigendian] [--png pngname] [--kmz kmzname] [--title str]

 -i  Input image file in float32, uint8, GeoTIFF, or NetCDF
 -p  Parameter file containing width and length (e.g., EQA.dem_par or mli.par)
     (Not required if input is GeoTIFF or NetCDF)
 -c  Colormap name (Default: SCM.roma_r, reverse of SCM.roma)
     Available colormaps (all cmap can be reversed with "_r"):
     - Matplotlib predefined name (e.g. viridis)
       https://matplotlib.org/tutorials/colors/colormaps.html
     - Scientific colour maps (e.g. SCM.roma)
       http://www.fabiocrameri.ch/colourmaps.php
     - Generic Mapping Tools (e.g. GMT.polar)
       https://docs.generic-mapping-tools.org/dev/cookbook/cpts.html
     - cmocean (e.g. cmocean.phase)
       https://matplotlib.org/cmocean/
     - colorcet (e.g. colorcet.CET_C1)
       https://colorcet.holoviz.org/
     - cm_insar (GAMMA standard rainbow color for wrapped phase)
     - cm_isce (ISCE standard rainbow color for wrapped phase)
 --cmin|cmax    Min|max values of color (Default: None (auto))
 --auto_crange  % of color range used for automatic determination (Default: 99)
 --n_color    Number of rgb quantization levels (Default: 256)
 --cycle        Value*2pi/cycle only if cyclic cmap (i.e., insar or SCM.*O*)
                (Default: 3 (6pi/cycle))
 --nodata       Nodata value (only for float32) (Default: 0)
 --bigendian    If input file is in big endian
 --png          Save png (pdf etc also available) instead of displaying
 --kmz          Save kmz (need EQA.dem_par for -p option)
 --title        Title to be displayed on the image
"""


#%% Change log
'''
v1.12 20221025 Qi Ou,Uni of Leeds
 - Add a title option
v1.11 20210205 Yu Morishita, GSI
 - More cmap available
 - Show colorbar for cyclic cmaps
v1.10 20210120 Yu Morishita, GSI
 - Add --n_color option
 - Bug fix in creating kmz with standard cmap
v1.9.1 20210108 Yu Morishita, GSI
 - Simultaneously usable png and kmz
v1.9 20201111 Yu Morishita, GSI
 - Data GeoTIFF or NetCDF available
v1.8.1 20200916 Yu Morishita, GSI
 - Small bug fix to display uint8
v1.8 20200902 Yu Morishita, GSI
 - Always use nearest interpolation to avoid expanded nan
v1.7 20200828 Yu Morishita, GSI
 - Update for matplotlib >= 3.3
 - Use nearest interpolation for cyclic cmap to avoid aliasing
v1.6 20200814 Yu Morishita, GSI
 - Set 0 as nodata by default
v1.5 20200317 Yu Morishita, Uni of Leeds and GSI
 - Add offscreen when kmz or png for working with bsub on Jasmin
 - Add name and description (cbar) tag in kmz
v1.4 20200225 Yu Morishita, Uni of Leeds and GSI
 - Use SCM instead of SCM5
 - Support uint8
v1.3 20200212 Yu Morishita, Uni of Leeds and GSI
 - Not display image with --kmz option
v1.2 20191025 Yu Morishita, Uni of Leeds and GSI
 - Add --kmz option
v1.1 20190828 Yu Morishita, Uni of Leeds and GSI
 - Add --png option
v1.0 20190729 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''


#%% Import
import getopt
import sys
import os
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import subprocess as subp
import zipfile
from osgeo import gdal

import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_io_lib as io_lib

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


#%% make_kmz
def make_kmz(lat1, lat2, lon1, lon2, pngfile, kmzfile, pngcfile, description):
    kmlfile = kmzfile.replace('.kmz', '.kml')
    name = os.path.basename(kmzfile).replace('.kmz', '')

    with open(kmlfile, "w") as f:
        print('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document><GroundOverlay>\n<name>{}</name>\n<description>{}</description>\n<Icon><href>{}</href></Icon>\n<altitude>0</altitude>\n<tessellate>0</tessellate>\n<altitudeMode>clampToGround</altitudeMode>\n<LatLonBox><south>{}</south><north>{}</north><west>{}</west><east>{}</east></LatLonBox>\n</GroundOverlay></Document></kml>'.format(name, description, pngfile, lat1, lat2, lon1, lon2), file=f)

    with zipfile.ZipFile(kmzfile, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(kmlfile)
        f.write(pngfile)
        if pngcfile: f.write(pngcfile)

    os.remove(kmlfile)

    return


#%% Main
## Not use def main to use global valuables
if __name__ == "__main__":
    argv = sys.argv

    ver="1.11.0"; date=20210205; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    infile = []
    parfile = []
    cmap_name = "SCM.roma_r"
    cmin = None
    cmax = None
    auto_crange = 99.0
    n_color = 256
    cycle = 3.0
    nodata = 0
    endian = 'little'
    pngname = []
    kmzname = []
    interp = 'nearest' #'antialiased'


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:p:c:", ["help", "cmin=", "cmax=", "auto_crange=", "n_color=", "cycle=", "nodata=", "bigendian", "png=", "kmz=", "title="])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                sys.exit(0)
            elif o == '-i':
                infile = a
            elif o == '-p':
                parfile = a
            elif o == '-c':
                cmap_name = a
            elif o == '--cmin':
                cmin = float(a)
            elif o == '--cmax':
                cmax = float(a)
            elif o == '--auto_crange':
                auto_crange = float(a)
            elif o == '--n_color':
                n_color = int(a)
            elif o == '--cycle':
                cycle = float(a)
            elif o == '--nodata':
                nodata = float(a)
            elif o == '--bigendian':
                endian = 'big'
            elif o == '--png':
                pngname = a
            elif o == '--kmz':
                kmzname = a
            elif o == '--title':
                title = a

        if not infile:
            raise Usage('No image file given, -i is not optional!')
        elif not os.path.exists(infile):
            raise Usage('No {} exists!'.format(infile))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        sys.exit(2)


    #%% Get info and Read data
    if gdal.IdentifyDriver(infile): ## If Geotiff or grd
        geotiff = gdal.Open(infile)
        data = geotiff.ReadAsArray()
        if data.ndim > 2:
            print('\nERROR: {} has multiple bands and cannot be displayed.\n'.format(infile), file=sys.stderr)
            sys.exit(2)

        length, width = data.shape
        lon_w_p, dlon, _, lat_n_p, _, dlat = geotiff.GetGeoTransform()
        lat_s_p = lat_n_p+dlat*length
        lon_e_p = lon_w_p+dlon*width
        if data.dtype == np.float32:
            data[data==nodata] = np.nan

    else: ## Not GeoTIFF
        if not parfile:
            print('\nERROR: No par file given, -p is not optional!\n', file=sys.stderr)
            sys.exit(2)
        elif not os.path.exists(parfile):
            print('\nERROR: No {} exists!\n'.format(parfile), file=sys.stderr)
            sys.exit(2)

        try:
            try:
                ### EQA.dem_par
                width = int(subp.check_output(['grep', 'width', parfile]).decode().split()[1].strip())
                length = int(subp.check_output(['grep', 'nlines', parfile]).decode().split()[1].strip())
            except:
                ### slc.mli.par
                width = int(subp.check_output(['grep', 'range_samples', parfile]).decode().split()[1].strip())
                length = int(subp.check_output(['grep', 'azimuth_lines', parfile]).decode().split()[1].strip())
        except:
            print('No fields about width/length found in {}!'.format(parfile), file=sys.stderr)
            sys.exit(2)

        if kmzname:
            try:
                ### EQA.dem_par
                dlat = float(io_lib.get_param_par(parfile, 'post_lat'))
                dlon = float(io_lib.get_param_par(parfile, 'post_lon'))

                lat_n_g = float(io_lib.get_param_par(parfile, 'corner_lat')) #grid reg
                lon_w_g = float(io_lib.get_param_par(parfile, 'corner_lon')) #grid reg
                ## Grid registration to pixel registration by shifing half pixel
                lat_n_p = lat_n_g - dlat/2
                lon_w_p = lon_w_g - dlon/2
                lat_s_p = lat_n_p+dlat*length
                lon_e_p = lon_w_p+dlon*width
            except:
                print('No fields about geo for kmz found in {}!'.format(parfile), file=sys.stderr)
                sys.exit(2)

        ### Read data
        if os.path.getsize(infile) == length*width:
            print('File format: uint8')
            data = io_lib.read_img(infile, length, width, np.uint8, endian=endian)
        else:
            data = io_lib.read_img(infile, length, width, endian=endian)
            data[data==nodata] = np.nan


    #%% Get cmap, set cyclic and color range
    cmap = tools_lib.get_cmap(cmap_name, n_color)

    if cmap_name == 'cm_insar' or \
            cmap_name == 'cm_isce' or \
            (cmap_name.startswith('SCM') and 'O' in cmap_name) or \
            'GMT.cyclic' in cmap_name or \
            'cmocean.phase' in cmap_name or \
            (cmap_name.startswith('colorcet') and
             re.search(r'C[1-5]', cmap_name) is not None):
        print('\n{} is a cyclic cmap.'.format(cmap_name))
        cyclic= True
        data = np.angle(np.exp(1j*(data/cycle))*cycle)
        cmin = -np.pi
        cmax = np.pi
        interp = 'nearest'
    else:
        cyclic= False

    if cmin is None and cmax is None: ## auto
        climauto = True
        cmin = np.nanpercentile(data, 100-auto_crange)
        cmax = np.nanpercentile(data, auto_crange)
    else:
        climauto = False
        if cmin is None: cmin = np.nanpercentile(data, 100-auto_crange)
        if cmax is None: cmax = np.nanpercentile(data, auto_crange)


    #%% Output kmz
    if kmzname:
        ### Make png
        os.environ['QT_QPA_PLATFORM']='offscreen'
        dpi = 100
        figsize2 = (width/dpi, length/dpi)
        plt.figure(figsize=figsize2, dpi=dpi)
        plt.imshow(data, clim=[cmin, cmax], cmap=cmap, interpolation=interp)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        pngnametmp = kmzname.replace('.kmz', '_tmp.png')
        plt.savefig(pngnametmp, dpi=dpi, transparent=True)
        plt.close()

        ### Make cbar png
        fig, ax = plt.subplots(figsize=(3, 1))
        norm = mpl.colors.Normalize(cmin, cmax)
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
        pngcfile = kmzname.replace('.kmz', '_ctmp.png')
        plt.tight_layout()
        if cyclic:
            cbar.set_ticks([])
            descr_cycle = '{}*2pi/cycle'.format(cycle)
        else:
            descr_cycle = ''
        plt.savefig(pngcfile, transparent=True)
        plt.close()

        description = '<![CDATA[{}<img style="max-width:200px;" src="{}">]]>'.format(descr_cycle, os.path.basename(pngcfile))

        make_kmz(lat_s_p, lat_n_p, lon_w_p, lon_e_p, pngnametmp, kmzname, pngcfile, description)

        os.remove(pngnametmp)
        if pngcfile:
            os.remove(pngcfile)
        print('\nOutput: {}\n'.format(kmzname))

        if not pngname: sys.exit(0)


    #%% Plot figure
    if pngname: os.environ['QT_QPA_PLATFORM']='offscreen'

    figsize_x = 6 if length > width else 8
    figsize = (figsize_x, ((figsize_x-2)*length/width))
    plt.figure('{}'.format(infile), figsize)
    plt.imshow(data, clim=[cmin, cmax], cmap=cmap, interpolation=interp)
    if title:
        plt.title(title)
    if cyclic:
        title = title+' {}*2pi/cycle'.format(cycle)
        plt.title(title)
        cbar = plt.colorbar()
        cbar.set_ticks([])
    else:
        plt.colorbar()
    plt.tight_layout()

    if pngname:
        plt.savefig(pngname)
        plt.close()
        print('\nOutput: {}\n'.format(pngname))
        sys.exit(0)
    else:
        plt.show()
