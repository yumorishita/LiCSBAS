#!/usr/bin/env python3
"""
v1.4 20200225 Yu Morishita, Uni of Leeds and GSI

========
Overview
========
This script displays an image file.

=====
Usage
=====
LiCSBAS_disp_img.py -i image_file -p par_file [-c cmap] [--cmin float] [--cmax float] [--auto_crange float]  [--cycle float] [--bigendian] [--png pngname] [--kmz kmzname]

 -i  Input image file in float32 or uint8
 -p  Parameter file containing width and length (e.g., EQA.dem_par or mli.par)
 -c  Colormap name (see below for available colormap)
     - https://matplotlib.org/tutorials/colors/colormaps.html
     - http://www.fabiocrameri.ch/colourmaps.php
     - insar
     (Default: SCM.roma_r, reverse of SCM.roma)
 --cmin|cmax    Min|max values of color (Default: None (auto))
 --auto_crange  % of color range used for automatic determinatin (Default: 99)
 --cycle        Value*2pi/cycle only if cyclic cmap (i.e., insar or SCM.*O*)
                (Default: 3 (6pi/cycle))
 --bigendian    If input file is in big endian
 --png          Save png (pdf etc also available) instead of displaying
 --kmz          Save kmz (need EQA.dem_par for -p option)

"""

#%% Change log
'''
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
import matplotlib.pyplot as plt
import numpy as np
import subprocess as subp
import SCM
import zipfile

import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_io_lib as io_lib

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg

#%%
def make_kmz(lat1, lat2, lon1, lon2, pngfile, kmzfile):
    kmlfile = kmzfile.replace('.kmz', '.kml')
        
    with open(kmlfile, "w") as f:
        print('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">/n<Document><GroundOverlay><Icon>\n<href>{}</href>\n</Icon>\n<altitude>0</altitude>\n<tessellate>0</tessellate>\n<altitudeMode>clampToGround</altitudeMode>\n<LatLonBox><south>{}</south><north>{}</north><west>{}</west><east>{}</east></LatLonBox>\n</GroundOverlay></Document></kml>'.format(pngfile, lat1, lat2, lon1, lon2), file=f)
        
    with zipfile.ZipFile(kmzfile, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(kmlfile)
        f.write(pngfile)

    os.remove(kmlfile)

    return


#%% Main
## Not use def main to use global valuables
if __name__ == "__main__":
    argv = sys.argv

    ver=1.4; date=20200224; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    #%% Set default
    infile = []
    parfile = []
    cmap_name = "SCM.roma_r"
    cmin = None
    cmax = None
    auto_crange = 99.0
    cycle = 3.0
    endian = 'little'
    pngname = []
    kmzname = []
    
    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:p:c:", ["help", "cmin=", "cmax=", "auto_crange=", "cycle=", "bigendian", "png=", "kmz="])
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
            elif o == '--cycle':
                cycle = float(a)
            elif o == '--bigendian':
                endian = 'big'
            elif o == '--png':
                pngname = a
            elif o == '--kmz':
                kmzname = a

        if not infile:
            raise Usage('No image file given, -i is not optional!')
        elif not os.path.exists(infile):
            raise Usage('No {} exists!'.format(infile))
        if not parfile:
            raise Usage('No par file given, -p is not optional!')
        elif not os.path.exists(parfile):
            raise Usage('No {} exists!'.format(parfile))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        sys.exit(2)


    #%% Set cmap if SCM
    if cmap_name.startswith('SCM'):
        if cmap_name.endswith('_r'):
            exec("cmap = {}.reversed()".format(cmap_name[:-2]))
        else:
            exec("cmap = {}".format(cmap_name))
    elif cmap_name == 'insar':
        cdict = tools_lib.cmap_insar()
        plt.register_cmap(name='insar', data=cdict)
        cmap='insar'
    else:
        cmap = cmap_name
        

    #%% Get info
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


    #%% Read data
    if os.path.getsize(infile) == length*width:
        print('File format: uint8')
        data = io_lib.read_img(infile, length, width, np.uint8, endian=endian)
    else:
        data = io_lib.read_img(infile, length, width, endian=endian)
    
    if cmap_name == 'insar' or (cmap_name.startswith('SCM') and 'O' in cmap_name):
        data = np.angle(np.exp(1j*(data/cycle))*cycle)
        cmin = -np.pi
        cmax = np.pi


    #%% Set color range for displacement and vel
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
        dpi = 100
        figsize2 = (width/dpi, length/dpi)
        plt.figure(figsize=figsize2, dpi=dpi)
        plt.imshow(data, clim=[cmin, cmax], cmap=cmap)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        pngnametmp = kmzname.replace('.kmz', '_tmp.png')
        plt.savefig(pngnametmp, dpi=dpi, transparent=True)
        plt.close()
          
        make_kmz(lat_s_p, lat_n_p, lon_w_p, lon_e_p, pngnametmp, kmzname)
        
        os.remove(pngnametmp)
        print('\nOutput: {}\n'.format(kmzname))
        
        sys.exit(0)


    #%% Plot figure
    figsize_x = 6 if length > width else 8
    figsize = (figsize_x, ((figsize_x-2)*length/width))
    plt.figure('{}'.format(infile), figsize)
    plt.imshow(data, clim=[cmin, cmax], cmap=cmap)
    if cmap == 'insar' or (cmap_name.startswith('SCM') and 'O' in cmap_name):
        plt.title('{}*2pi/cycle'.format(cycle))
    else:  ### Not cyclic
        plt.colorbar()
    plt.tight_layout()
    
    if pngname:
        plt.savefig(pngname)
        plt.close()
        print('\nOutput: {}\n'.format(pngname))
    else:
        plt.show()

