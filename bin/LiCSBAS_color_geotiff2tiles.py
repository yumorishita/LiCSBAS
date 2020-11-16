#!/usr/bin/env python3
"""
v1.0.3 20201116 Yu Morishita, GSI

========
Overview
========
This script generates directory with TMS tiles using gdal2tiles.py.
https://gdal.org/programs/gdal2tiles.html

=====
Usage
=====
LiCSBAS_color_geotiff2tiles.py -i infile [-o outdir] [--zmin int] [--zmax int]
 [--xyz] [--n_para int] 

 -i  Input color GeoTIFF file
 -o  Output directory containing XYZ tiles
     (Default: tiles_[infile%.tif], '.' will be replaced with '_')
 --zmin  Minimum zoom level to render (Default: 5)
 --zmax  Maximum zoom level to render (Default: auto, see below)
         17 (pixel spacing <=   5m)
         16 (pixel spacing <=  10m)
         15 (pixel spacing <=  20m)
         14 (pixel spacing <=  40m)
         13 (pixel spacing <=  80m)
         12 (pixel spacing <= 160m)
         11 (pixel spacing >  160m)
 --xyz     Output XYZ tiles instead of TMS (opposite Y)
 --n_para  Number of parallel processing (Default: # of usable CPU)
           Available only in gdal>=2.3

"""
#%% Change log
'''
v1.0.3 20201116 Yu Morishita, GSI
 - Bug fix of multiprocessing in Mac python>=3.8
v1.0.2 20201028 Yu Morishita, GSI
 - Update how to get n_para
v1.0.1 20201016 Yu Morishita, GSI
 - Change default output name
v1.0 20200924 Yu Morishita, GSI
 - Original implementation
'''


#%% Import
import getopt
import os
import sys
import time
import shutil
import glob
import gdal
import numpy as np
import subprocess as subp
import multiprocessing as multi
multi.set_start_method('fork') # for python >=3.8 in Mac

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
    ver='1.0.3'; date=20201116; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    ### For paralell processing
    global files


    #%% Set default
    infile = []
    outdir = []
    zmin = 5
    zmax = []
    tms_flag = True
    try:
        n_para = len(os.sched_getaffinity(0))
    except:
        n_para = multi.cpu_count()


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:", ["help", "zmin=", "zmax=", "xyz", "n_para="])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                infile = a
            elif o == '-o':
                outdir = a
            elif o == '--zmin':
                zmin = int(a)
            elif o == '--zmax':
                zmax = int(a)
            elif o == '--xyz':
                tms_flag = False
            elif o == '--n_para':
                n_para = int(a)

        if not infile:
            raise Usage('No input file given, -i is not optional!')
        elif not os.path.exists(infile):
            raise Usage('No {} exists!'.format(infile))
        elif gdal.Open(infile) is None:
            raise Usage('{} is not GeoTIFF!'.format(infile))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Parameter setting
    if not outdir:
        outdir = 'tiles_'+infile.replace('.tif', '').replace('.', '_')
    if os.path.exists(outdir):
        print('\n{} already exists. Remove and overwrite.'.format(outdir), flush=True)
        shutil.rmtree(outdir)

    print('\nOutput dir: {}'.format(outdir), flush=True)

    if not zmax:
        geotiff = gdal.Open(infile)
        lon_w, dlon, _, lat_n, _, dlat = geotiff.GetGeoTransform()

        ### Approx pixel spacing in meter
        dlat_m = abs(dlat*40000000/360)  ## deg -> meter
        dlon_m = abs(dlon*40000000/360*np.cos(np.deg2rad(lat_n)))  ## deg -> meter

        pixsp = dlat_m if dlat_m < dlon_m else dlon_m  ## Use smaller one

        if pixsp <= 5: zmax = 17
        elif pixsp <= 10: zmax = 16
        elif pixsp <= 20: zmax = 15
        elif pixsp <= 40: zmax = 14
        elif pixsp <= 80: zmax = 13
        elif pixsp <= 160: zmax = 12
        else: zmax = 11

    print('\nZoom levels: {} - {}'.format(zmin, zmax), flush=True)

    gdalver = gdal.VersionInfo() ## e.g., 3.1.1 -> 3010100, str


    #%% gdal2ties
    call = ["gdal2tiles.py", "-z", "{}-{}".format(zmin, zmax), 
            "--no-kml", "-w", "leaflet",
            infile, outdir]

    if int(gdalver[0]) >= 3:
        ## -x option (Exclude transparent tiles) available in ver>=3
        call.insert(1, "-x")

    if int(gdalver[0:3]) >= 203:
        ## --processes option available in ver>=2.3
        call.insert(1, "--processes={}".format(n_para))

    if not tms_flag and int(gdalver[0:3]) >= 301:
        ## --xyz option available ver>=3.1
        call.insert(1, "--xyz")
    
    print('', flush=True)
    print(' '.join(call), flush=True)

    p = subp.Popen(call, stdout = subp.PIPE, stderr = subp.STDOUT)
    for line in iter(p.stdout.readline, b''):
        print(line.rstrip().decode("utf8"), flush=True)


    #%% Remove transparent tiles if gdal<3
    if int(gdalver[0]) <3:
        print('\nRemove transparent tiles...', flush=True)
        call = ["find", outdir, "-size", "334c", "|",  "xargs", "rm", "-f"]
        subp.run(' '.join(call), shell=True)
        

    #%% Invert y if XYZ tiles and gdal<3.1
    if not tms_flag and int(gdalver[0:3]) < 301:
        print('\nInvert Y with {} parallel processing...'.format(n_para), flush=True)
        files = glob.glob('{}/*/*/*.png'.format(outdir))
        p = multi.Pool(n_para)
        p.map(invert_y_wrapper, range(len(files)))
        p.close()

    
    #%% Edit leaflet.html
    with open(os.path.join(outdir, 'leaflet.html'), 'r') as f:
        lines = f.readlines()

    ### Add GSImaps
    gsi = '        //  .. GSIMaps\n        var gsi = L.tileLayer(\'https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png\', {attribution: \'<a href="https://maps.gsi.go.jp/development/ichiran.html" target="_blank">地理院タイル</a>\'});\n'
    gsi2 = '"GSIMaps": gsi, '
    gsi_photo = '        //  .. GSIMaps photo\n        var gsi_photo = L.tileLayer(\'https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg\', {attribution: \'<a href="https://maps.gsi.go.jp/development/ichiran.html" target="_blank">地理院タイル</a>\'});\n'
    gsi_photo2 = '"GSIMaps Photo": gsi_photo, '

    ### XYZ or TMS
    tms = 'true' if tms_flag else 'false'

    ### Replace
    lines2 = [s+gsi+gsi_photo if '// Base layers\n' in s else 
              s.replace('= {', '= {'+gsi2+gsi_photo2) if 'var basemaps =' in s else
              s.replace('true', tms) if 'tms: true' in s else
              s for s in lines]
    
    with open(os.path.join(outdir, 'leaflet2.html'), 'w') as f:
         f.writelines(lines2)

    
    #%% Create layers.txt for GSIMaps
#    url = 'file://' + os.path.join(os.path.abspath(outdir), '{z}', '{x}', '{y}.png')
    url = os.path.join('http://', 'XXX', outdir, '{z}', '{x}', '{y}.png')
    with open(os.path.join(outdir, 'layers.txt'), 'w') as f:
         f.writelines(layers_txt(outdir, url, zmin, zmax, tms))

    
    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output: {}'.format(outdir), flush=True)
    print('')


#%%
def invert_y_wrapper(i):
    file = files[i]
    if np.mod(i, 1000) == 0:
        print("  {0:5}/{1:5}th file...".format(i, len(files)), flush=True)

    d, z, x, ypng = file.split('/')
    y = os.path.splitext(ypng)[0]
    y_new = str(2**int(z)-int(y)-1)
    file_new = os.path.join(d, z, x, y_new+'.png')
    os.rename(file, file_new)


#%%
def layers_txt(outdir, url, zmin, zmax, tms):
    layerstxt = \
'{{\n\
  "layers": [\n\
    {{\n\
      "type": "LayerGroup",\n\
      "title": "",\n\
      "entries": [\n\
        {{\n\
          "type": "Layer",\n\
          "id": "",\n\
          "title": "{}",\n\
          "iconUrl": "",\n\
          "url": "{}",\n\
          "subdomains": "",\n\
          "attribution": "",\n\
          "errorTileUrl": "",\n\
          "cocotile": false,\n\
          "minZoom": {},\n\
          "maxZoom": 18,\n\
          "maxNativeZoom": {},\n\
          "tms": {},\n\
          "legendUrl": "",\n\
          "html": ""\n\
        }}\n\
      ]\n\
    }}\n\
  ]\n\
}}\n'.format(outdir, url, zmin, zmax, tms)

    return layerstxt


#%% main
if __name__ == "__main__":
    sys.exit(main())

