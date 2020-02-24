#!/usr/bin/env python3
"""
v1.0 20190730 Yu Morishita, Uni of Leeds and GSI

========
Overview
========
This script clips a specified rectangular area of interest from unw and cc data. The clipping can make the data size smaller and processing faster, and improve the result of step 12 (loop closure).

===============
Input & output files
===============
Inputs in GEOCml* directory:
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.unw
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.cc
 - slc.mli[.par|.png]
 - EQA.dem.par
 
Outputs in output directory:
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.unw[.png] (clipped)
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.cc (clipped)
 - slc.mli[.par|.png] (clipped)
 - EQA.dem.par (clipped)
 - Other files in input directory if exist (copy or clipped)
 - cliparea.txt

=====
Usage
=====
LiCSBAS05op_clip_unw.py -i in_dir -o out_dir [-r x1:x2/y1:y2] [-g lon1/lon2/lat1/lat2]

 -i  Path to the GEOCml* dir containing stack of unw data.
 -o  Path to the output dir.
 -r  Range to be clipped. Index starts from 0.
     0 for x2/y2 means all. (i.e., 0:0/0:0 means whole area).
 -g  Range to be clipped in geographical coordinates (deg).

"""
#%% Change log
'''
v1.0 20190730 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''

#%% Import
import getopt
import os
import re
import sys
import glob
import shutil
import time
import numpy as np
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg


#%%
def main(argv=None):
   
    #%% Check argv
    if argv == None:
        argv = sys.argv
        
    start = time.time()
    ver=1.0; date=20190730; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    in_dir = []
    out_dir = []
    range_str = []
    range_geo_str = []


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:o:r:g:", ["help"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                in_dir = a
            elif o == '-o':
                out_dir = a
            elif o == '-r':
                range_str = a
            elif o == '-g':
                range_geo_str = a

        if not in_dir:
            raise Usage('No input directory given, -i is not optional!')
        if not out_dir:
            raise Usage('No output directory given, -o is not optional!')
        if not range_str and not range_geo_str:
            raise Usage('No clip area given, use either -r or -g!')
        if range_str and range_geo_str:
            raise Usage('Both -r and -g given, use either -r or -g not both!')
        elif not os.path.isdir(in_dir):
            raise Usage('No {} dir exists!'.format(in_dir))
        elif not os.path.exists(os.path.join(in_dir, 'slc.mli.par')):
            raise Usage('No slc.mli.par file exists in {}!'.format(in_dir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Read info and make dir
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    ifgdates = tools_lib.get_ifgdates(in_dir)
    n_ifg = len(ifgdates)

    mlipar = os.path.join(in_dir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))

    speed_of_light = 299792458 #m/s
    radar_frequency = float(io_lib.get_param_par(mlipar, 'radar_frequency')) #Hz
    wavelength = speed_of_light/radar_frequency #meter
    if wavelength > 0.2: ## L-band
        cycle = 1.5  # 2pi/cycle for png
    else: ## C-band
        cycle = 3  # 2pi*3/cycle for png

    dempar = os.path.join(in_dir, 'EQA.dem_par')
    lat1 = float(io_lib.get_param_par(dempar, 'corner_lat')) # north
    lon1 = float(io_lib.get_param_par(dempar, 'corner_lon')) # west
    postlat = float(io_lib.get_param_par(dempar, 'post_lat')) # negative
    postlon = float(io_lib.get_param_par(dempar, 'post_lon')) # positive
    lat2 = lat1+postlat*(length-1) # south
    lon2 = lon1+postlon*(width-1) # east

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    #%% Check and set range to be clipped
    ### Read -r or -g option
    if range_str: ## -r
        if not tools_lib.read_range(range_str, width, length):
            print('\nERROR in {}\n'.format(range_str), file=sys.stderr)
            return 1
        else:
            x1, x2, y1, y2 = tools_lib.read_range(range_str, width, length)
    else: ## -g
        if not tools_lib.read_range_geo(range_geo_str, width, length, lat1, postlat, lon1, postlon):
            print('\nERROR in {}\n'.format(range_geo_str), file=sys.stderr)
            return 1
        else:
            x1, x2, y1, y2 = tools_lib.read_range_geo(range_geo_str, width, length, lat1, postlat, lon1, postlon)
            range_str = '{}:{}/{}:{}'.format(x1, x2, y1, y2)
        
    ### Calc clipped  info
    width_c = x2-x1
    length_c = y2-y1
    lat1_c = lat1+postlat*y1 # north
    lon1_c = lon1+postlon*x1 # west
    lat2_c = lat1_c+postlat*(length_c-1) # south
    lon2_c = lon1_c+postlon*(width_c-1) # east

    print("\nArea to be clipped:", flush=True)
    print("  0:{}/0:{} -> {}:{}/{}:{}".format(width, length, x1, x2, y1, y2))
    print("  {:.7f}/{:.7f}/{:.7f}/{:.7f} ->".format(lon1, lon2, lat2, lat1))
    print("  {:.7f}/{:.7f}/{:.7f}/{:.7f}".format(lon1_c, lon2_c, lat2_c, lat1_c))
    print("  Width/Length: {}/{} -> {}/{}".format(width, length, width_c, length_c))
    print("", flush=True)

    clipareafile = os.path.join(out_dir, 'cliparea.txt')
    with open(clipareafile, 'w') as f: f.write(range_str)


    #%% Make clipped par files
    mlipar_c = os.path.join(out_dir, 'slc.mli.par')
    dempar_c = os.path.join(out_dir, 'EQA.dem_par')
    
    ### slc.mli.par
    with open(mlipar, 'r') as f: file = f.read()
    file = re.sub(r'range_samples:\s*{}'.format(width), 'range_samples: {}'.format(width_c), file)
    file = re.sub(r'azimuth_lines:\s*{}'.format(length), 'azimuth_lines: {}'.format(length_c), file)
    with open(mlipar_c, 'w') as f: f.write(file)

    ### EQA.dem_par
    with open(dempar, 'r') as f: file = f.read()
    file = re.sub(r'width:\s*{}'.format(width), 'width: {}'.format(width_c), file)
    file = re.sub(r'nlines:\s*{}'.format(length), 'nlines: {}'.format(length_c), file)
    file = re.sub(r'corner_lat:\s*{}'.format(lat1), 'corner_lat: {}'.format(lat1_c), file)
    file = re.sub(r'corner_lon:\s*{}'.format(lon1), 'corner_lon: {}'.format(lon1_c), file)
    with open(dempar_c, 'w') as f: f.write(file)


    #%% Clip or copy other files than unw and cc
    files = glob.glob(os.path.join(in_dir, '*'))
    for file in files:
        if os.path.isdir(file):
            continue  #not copy directory
        elif file==mlipar or file==dempar:
            continue  #not copy 
        elif os.path.getsize(file) == width*length*4: ##float file
            print('Clip {}'.format(os.path.basename(file)), flush=True)
            data = io_lib.read_img(file, length, width)
            data = data[y1:y2, x1:x2]
            filename = os.path.basename(file)
            outfile = os.path.join(out_dir, filename)
            data.tofile(outfile)
        elif file==os.path.join(in_dir, 'slc.mli.png'):
            print('Recreate slc.mli.png', flush=True)
            mli = io_lib.read_img(os.path.join(out_dir, 'slc.mli'), length_c, width_c)
            pngfile = os.path.join(out_dir, 'slc.mli.png')
            plot_lib.make_im_png(mli, pngfile, 'gray', 'MLI', cbar=False)
        else:
            print('Copy {}'.format(os.path.basename(file)), flush=True)
            shutil.copy(file, out_dir)


    #%% Clip unw and cc
    print('\nClip unw and cc', flush=True)
    ### First, check if already exist
    ifgdates2 = []
    for ifgix, ifgd in enumerate(ifgdates): 
        out_dir1 = os.path.join(out_dir, ifgd)
        unwfile_c = os.path.join(out_dir1, ifgd+'.unw')
        ccfile_c = os.path.join(out_dir1, ifgd+'.cc')
        if not (os.path.exists(unwfile_c) and os.path.exists(ccfile_c)):
            ifgdates2.append(ifgd)

    n_ifg2 = len(ifgdates2)
    if n_ifg-n_ifg2 > 0:
        print("  {0:3}/{1:3} clipped unw and cc already exist. Skip".format(n_ifg-n_ifg2, n_ifg), flush=True)

    ### Clip    
    for ifgix, ifgd in enumerate(ifgdates2): 
        if np.mod(ifgix,100) == 0:
            print("  {0:3}/{1:3}th unw...".format(ifgix, n_ifg2), flush=True)
    
        unwfile = os.path.join(in_dir, ifgd, ifgd+'.unw')
        ccfile = os.path.join(in_dir, ifgd, ifgd+'.cc')
        
        unw = io_lib.read_img(unwfile, length, width)
        coh = io_lib.read_img(ccfile, length, width)

        ### Clip
        unw = unw[y1:y2, x1:x2]
        coh = coh[y1:y2, x1:x2]

        ### Output
        out_dir1 = os.path.join(out_dir, ifgd)
        if not os.path.exists(out_dir1): os.mkdir(out_dir1)
        
        unw.tofile(os.path.join(out_dir1, ifgd+'.unw'))
        coh.tofile(os.path.join(out_dir1, ifgd+'.cc'))

        ## Output png for corrected unw
        pngfile = os.path.join(out_dir1, ifgd+'.unw.png')
        title = '{} ({}pi/cycle)'.format(ifgd, cycle*2)
        plot_lib.make_im_png(np.angle(np.exp(1j*unw/cycle)*cycle), pngfile, 'insar', title, -np.pi, np.pi, cbar=False)


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(out_dir)))


#%% main
if __name__ == "__main__":
    sys.exit(main())
