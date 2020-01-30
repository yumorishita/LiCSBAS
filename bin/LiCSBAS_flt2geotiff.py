#!/usr/bin/env python3
"""
========
Overview
========
This script makes a GeoTIFF file from an image file (only in float format). The geotiff file can be read by a GIS software (e.g., QGIS) and used to make a figure.

=========
Changelog
=========
v1.2 20200130 Yu Morishita, Uni of Leeds and GSI
 - Add compress option with DEFLATE for gdal
v1.0 20190729 Yu Morishita, Uni of Leeds and GSI
 - Original implementationf

=====
Usage
=====
LiCSBAS_flt2geotiff.py -i infile -p dempar [-o outfile] [--bigendian]

 -i  Path to input file (float, little endian)
 -p  Path to dem parameter file (EQA.dem_par)
 -o  Output geotiff file (Default: infile[.geo].tif)
 --bigendian  If input file is in big endian

"""
# Caution: gdal or GAMMA must be installed.


#%% Import
import getopt
import shutil
import os
import sys
import time
import subprocess as subp
import numpy as np
import LiCSBAS_io_lib as io_lib

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

    #%% Set default
    infile = []
    dempar = []
    outfile = []
    endian = 'little'

    compress_option = ['COMPRESS=DEFLATE', 'PREDICTOR=3']
    ## ['COMPRESS=LZW', 'PREDICTOR=3'], ['COMPRESS=PACKBITS']

    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:p:o:", ["help", "bigendian"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                infile = a
            elif o == '-p':
                dempar = a
            elif o == '-o':
                outfile = a
            elif o == '--bigendian':
                endian = 'big'

        if not infile:
            raise Usage('No input file given, -i is not optional!')
        elif not os.path.exists(infile):
            raise Usage('No {} exists!'.format(infile))
        elif not dempar:
            raise Usage('No dempar file given, -p is not optional!')
        elif not os.path.exists(dempar):
            raise Usage('No {} exists!'.format(dempar))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Read info
    if not outfile:
        outfile = infile+'.geo.tif'

    width = int(io_lib.get_param_par(dempar, 'width'))
    length = int(io_lib.get_param_par(dempar, 'nlines'))


    #%% If data2geotiff (GAMMA) is available
    gamma_comm = 'data2geotiff'
    if shutil.which(gamma_comm):
        dtype = '2' # float
        print('Use data2geotiff in GAMMA')

        if endian == 'little':
            ### Make temporary big endian file
            data = io_lib.read_img(infile, length, width, endian=endian)
            data.byteswap().tofile(infile+'.bigendian')
            infile = infile+'.bigendian'

       
        print('{} {} {} {} {}'.format(gamma_comm, dempar, infile, dtype, outfile))
        call=[gamma_comm, dempar, infile, dtype, outfile]
        subp.run(call)
        
        if endian == 'little':
            ### Remove temporary file
            os.remove(infile)
        
        
    #%% if gdal and osr available
    else:
        print('Use gdal module')
        try:
            import gdal, osr
        except:
            print("\nERROR: gdal must be installed.", file=sys.stderr)
            return 1

        width = int(io_lib.get_param_par(dempar, 'width'))
        length = int(io_lib.get_param_par(dempar, 'nlines'))
        
        dlat = float(io_lib.get_param_par(dempar, 'post_lat'))
        dlon = float(io_lib.get_param_par(dempar, 'post_lon'))
        
        lat_n_g = float(io_lib.get_param_par(dempar, 'corner_lat')) #grid reg
        lon_w_g = float(io_lib.get_param_par(dempar, 'corner_lon')) #grid reg
        
        ## Grid registration to pixel registration by shifing half pixel
        lat_n_p = lat_n_g - dlat/2
        lon_w_p = lon_w_g - dlon/2
                
        data = io_lib.read_img(infile, length, width, endian=endian)
    
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(outfile, width, length, 1, gdal.GDT_Float32, options=compress_option)
        outRaster.SetGeoTransform((lon_w_p, dlon, 0, lat_n_p, 0, dlat))
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(data)
        outband.SetNoDataValue(0)
        outRaster.SetMetadataItem('AREA_OR_POINT', 'Point')
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output: {}\n'.format(outfile), flush=True)


#%% main
if __name__ == "__main__":
    sys.exit(main())
