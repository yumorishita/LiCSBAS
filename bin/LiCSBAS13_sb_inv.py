#!/usr/bin/env python3
"""
v1.2 20200225 Yu Morishita, Uni of Leeds and GSI

========
Overview
========
This script inverts the small baseline network of ifgs to get the time series of displacement and velocity using the NSBAS (López-Quiroz et al., 2009; Doin et al., 2011) approach.
A stable reference point is determined after the inversion. RMS of the time series wrt median among all points as tentative reference is calculated for each point. Then the point with minimum RMS and minumum n_gap is selected as new stable reference point.

===============
Input & output files
===============
Inputs in GEOCml* :
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.unw
 - yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd.cc
 - EQA.dem_par
 - slc.mli.par
 - baselines (can be dummy)

Inputs in TS_GEOCml*/info :
 - ref[_prelim].txt
 - 11bad_ifg.txt
 - 12bad_ifg.txt
 
Outputs in TS_GEOCml* directory
 - cum.h5             : Cumulative displacement (time-seires) in mm
 - results/
   - vel[.png]        : Velocity in mm/yr (positive means LOS decrease; uplift)
   - vintercept[.png] : Constant part of linear velocity (c for vt+c) in mm
   - resid_rms[.png]  : RMS of residual in mm
   - n_gap[.png]      : Number of gaps in SB network
   - n_ifg_noloop[.png] :  Number of ifgs with no loop
   - maxTlen[.png]    : Max length of continous SB network in year
 - info/
   - 13parameters.txt   : Used parameters
   - 13used_image.txt : List of used images
   - 13resid.txt      : List of RMS of residual for each ifg
   - ref_stable[.txt|.kml]   : Auto-determined stable ref point
 - 13increment/yyyymmdd_yyyymmdd.inc_comp.png
                      : Estimated incremental displacement and daisy chain IFG
 - 13resid/yyyymmdd_yyyymmdd.res.png : Residual for each ifg
 - network/network13*.png : Figures of the network

=====
Usage
=====
LiCSBAS13_sb_inv.py -d ifgdir [-t tsadir] [--inv_alg LS|WLS] [--mem_size float] [--gamma float] [--n_core int] [--n_unw_r_thre float] [--keep_incfile] 

 -d  Path to the GEOCml* dir containing stack of unw data
 -t  Path to the output TS_GEOCml* dir.

 --inv_alg    Inversion algolism (Default: LS)
   LS :       NSBAS Least Square with no weight
   WLS:       NSBAS Weighted Least Square (not well tested)
              Weight (variance) is calculated by (1-coh**2)/(2*coh**2)
 --mem_size   Max memory size for each patch in MB. (Default: 4000)
 --gamma      Gamma value for NSBAS inversion (Default: 0.0001)
 --n_core     Number of cores for parallel processing (Default: 1)
 --n_unw_r_thre
              Threshold of n_unw (number of used unwrap data)
              (Note this value is ratio to the number of images; i.e., 1.5*n_im)
              Larger number (e.g. 2.5) makes processing faster but result sparser.
              (Default: 1 and 0.5 for C- and L-band, respectively)
 --keep_incfile
              Not remove inc and resid files (Default: remove them)

"""
#%% Change log
'''
v1.2 20200225 Yu Morishita, Uni of Leeds and GSI
 - Not output network pdf
 - Change color of png
 - Change name of parameters.txt to 13parameters.txt
 - Deal with cc file in uint8 format
 - Automatically find stable reference point
v1.1 20190829 Yu Morishita, Uni of Leeds and GSI
 - Remove cum.h5 if exists before creation
v1.0 20190730 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''

#%% Import
import getopt
import os
import sys
import re
import time
import h5py as h5
import numpy as np
import datetime as dt
import SCM
import LiCSBAS_io_lib as io_lib
import LiCSBAS_inv_lib as inv_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_loop_lib as loop_lib
import LiCSBAS_plot_lib as plot_lib

class Usage(Exception):
    """Usage context manager"""
    def __init__(self, msg):
        self.msg = msg

#%%
def make_point_kml(lat, lon, kmlfile):
    with open(kmlfile, "w") as f:
        print('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document><Placemark><Point>\n<coordinates>{},{}</coordinates>\n</Point></Placemark></Document>\n</kml>'.format(lon, lat), file=f)


#%% Main
def main(argv=None):
   
    #%% Check argv
    if argv == None:
        argv = sys.argv
        
    start = time.time()
    ver=1.2; date=20200225; author="Y. Morishita"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)


    #%% Set default
    ifgdir = []
    tsadir = []
    inv_alg = 'LS'
    n_core = 1

    memory_size = 4000
    gamma = 0.0001
    n_unw_r_thre = []
    keep_incfile = False

    cmap_vel = SCM.roma.reversed()
    cmap_noise = SCM.turku
    cmap_noise_r = SCM.turku.reversed()

    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hd:t:", ["help",  "mem_size=", "gamma=", "n_unw_r_thre=", "keep_incfile", "inv_alg=", "n_core="])
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
            elif o == '--mem_size':
                memory_size = float(a)
            elif o == '--gamma':
                gamma = float(a)
            elif o == '--n_unw_r_thre':
                n_unw_r_thre = float(a)
            elif o == '--keep_incfile':
                keep_incfile = True
            elif o == '--inv_alg':
                inv_alg = a
            elif o == '--n_core':
                n_core = int(a)

        if not ifgdir:
            raise Usage('No data directory given, -d is not optional!')
        elif not os.path.isdir(ifgdir):
            raise Usage('No {} dir exists!'.format(ifgdir))
        elif not os.path.exists(os.path.join(ifgdir, 'slc.mli.par')):
            raise Usage('No slc.mli.par file exists in {}!'.format(ifgdir))

    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Directory settings
    ifgdir = os.path.abspath(ifgdir)

    if not tsadir:
        tsadir = os.path.join(os.path.dirname(ifgdir), 'TS_'+os.path.basename(ifgdir))
    
    if not os.path.isdir(tsadir):
        print('\nNo {} exists!'.format(tsadir), file=sys.stderr)
        return 1

    tsadir = os.path.abspath(tsadir)
    resultsdir = os.path.join(tsadir, 'results')
    infodir = os.path.join(tsadir, 'info')
    netdir = os.path.join(tsadir, 'network')

    bad_ifg11file = os.path.join(infodir, '11bad_ifg.txt')
    bad_ifg12file = os.path.join(infodir, '12bad_ifg.txt')
    reffile = os.path.join(infodir, 'ref_prelim.txt')
    if not os.path.exists(reffile): ## for old LiCSBAS12 < v1.1
        reffile = os.path.join(infodir, 'ref.txt')

    incdir = os.path.join(tsadir,'13increment')
    if not os.path.exists(incdir): os.mkdir(incdir)

    resdir = os.path.join(tsadir,'13resid')
    if not os.path.exists(resdir): os.mkdir(resdir)
    restxtfile = os.path.join(infodir,'13resid.txt')

    cumh5file = os.path.join(tsadir,'cum.h5')
    

    #%% Check files
    try:
        if not os.path.exists(bad_ifg11file):
            raise Usage('No 11bad_ifg.txt file exists in {}!'.format(infodir))
        if not os.path.exists(bad_ifg12file):
            raise Usage('No 12bad_ifg.txt file exists in {}!'.format(infodir))
        if not os.path.exists(reffile):
            raise Usage('No ref[_prelim].txt file exists in {}!'.format(infodir))
    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2

      
    #%% Set preliminaly reference
    with open(reffile, "r") as f:
        refarea = f.read().split()[0]  #str, x1/x2/y1/y2
    refx1, refx2, refy1, refy2 = [int(s) for s in re.split('[:/]', refarea)]


    #%% Read data information
    ### Get size
    mlipar = os.path.join(ifgdir, 'slc.mli.par')
    width = int(io_lib.get_param_par(mlipar, 'range_samples'))
    length = int(io_lib.get_param_par(mlipar, 'azimuth_lines'))
    speed_of_light = 299792458 #m/s
    radar_frequency = float(io_lib.get_param_par(mlipar, 'radar_frequency')) #Hz
    wavelength = speed_of_light/radar_frequency #meter
    coef_r2m = -wavelength/4/np.pi*1000 #rad -> mm, positive is -LOS

    ### Calc pixel spacing depending on IFG or GEOC, used in later spatial filter
    dempar = os.path.join(ifgdir, 'EQA.dem_par')
    width_geo = int(io_lib.get_param_par(dempar, 'width'))
    length_geo = int(io_lib.get_param_par(dempar, 'nlines'))
    dlat = float(io_lib.get_param_par(dempar, 'post_lat')) #negative
    dlon = float(io_lib.get_param_par(dempar, 'post_lon')) #positive
    lat1 = float(io_lib.get_param_par(dempar, 'corner_lat'))
    lon1 = float(io_lib.get_param_par(dempar, 'corner_lon'))
    if width == width_geo and length == length_geo: ## Geocoded
        print('In geographical coordinates', flush=True)
        centerlat = lat1+dlat*(length/2)
        ra = float(io_lib.get_param_par(dempar, 'ellipsoid_ra'))
        recip_f = float(io_lib.get_param_par(dempar, 'ellipsoid_reciprocal_flattening'))
        rb = ra*(1-1/recip_f) ## polar radius
        pixsp_a = 2*np.pi*rb/360*abs(dlat)
        pixsp_r = 2*np.pi*ra/360*dlon*np.cos(np.deg2rad(centerlat))
    else:
        print('In radar coordinates', flush=True)
        pixsp_r_org = float(io_lib.get_param_par(mlipar, 'range_pixel_spacing'))
        pixsp_a = float(io_lib.get_param_par(mlipar, 'azimuth_pixel_spacing'))
        inc_agl = float(io_lib.get_param_par(mlipar, 'incidence_angle'))
        pixsp_r = pixsp_r_org/np.sin(np.deg2rad(inc_agl))
     

    ### Set n_unw_r_thre and cycle depending on L- or C-band
    if wavelength > 0.2: ## L-band
        if not n_unw_r_thre: n_unw_r_thre = 0.5
        cycle = 1.5 # 2pi/cycle for comparison png
    elif wavelength <= 0.2: ## C-band
        if not n_unw_r_thre: n_unw_r_thre = 1.0
        cycle = 3 # 3*2pi/cycle for comparison png


    #%% Read date and network information
    ### Get all ifgdates in ifgdir
    ifgdates_all = tools_lib.get_ifgdates(ifgdir)
    imdates_all = tools_lib.ifgdates2imdates(ifgdates_all)
    n_im_all = len(imdates_all)
    n_ifg_all = len(ifgdates_all)
    
    ### Read bad_ifg11 and 12
    bad_ifg11 = io_lib.read_ifg_list(bad_ifg11file)
    bad_ifg12 = io_lib.read_ifg_list(bad_ifg12file)
    bad_ifg_all = list(set(bad_ifg11+bad_ifg12))
    bad_ifg_all.sort()

    ### Remove bad ifgs and images from list
    ifgdates = list(set(ifgdates_all)-set(bad_ifg_all))
    ifgdates.sort()

    imdates = tools_lib.ifgdates2imdates(ifgdates)

    n_ifg = len(ifgdates)
    n_ifg_bad = len(set(bad_ifg11+bad_ifg12))
    n_im = len(imdates)
    n_unw_thre = int(n_unw_r_thre*n_im)

    ### Make 13used_image.txt
    imfile = os.path.join(infodir, '13used_image.txt')
    with open(imfile, 'w') as f:
        for i in imdates:
            print('{}'.format(i), file=f)

    ### Calc dt in year
    imdates_dt = ([dt.datetime.strptime(imd, '%Y%m%d').toordinal() for imd in imdates])
    dt_cum = np.float32((np.array(imdates_dt)-imdates_dt[0])/365.25)

    ### Construct G and Aloop matrix for increment and n_gap
    G = inv_lib.make_sb_matrix(ifgdates)
    Aloop = loop_lib.make_loop_matrix(ifgdates)
    n_loop = Aloop.shape[0] # (n_loop,n_ifg)


    #%% Plot network
    ## Read bperp data or dummy
    bperp_file = os.path.join(ifgdir, 'baselines')
    if os.path.exists(bperp_file):
        bperp_all = io_lib.read_bperp_file(bperp_file, imdates_all)
        bperp = io_lib.read_bperp_file(bperp_file, imdates)
    else: #dummy
        bperp_all = np.random.random(len(imdates_all)).tolist()
        bperp = np.random.random(n_im).tolist()
        
    pngfile = os.path.join(netdir, 'network13_all.png')
    plot_lib.plot_network(ifgdates_all, bperp_all, [], pngfile)

    pngfile = os.path.join(netdir, 'network13.png')
    plot_lib.plot_network(ifgdates_all, bperp_all, bad_ifg_all, pngfile)

    pngfile = os.path.join(netdir, 'network13_nobad.png')
    plot_lib.plot_network(ifgdates_all, bperp_all, bad_ifg_all, pngfile, plot_bad=False)


    #%% Get patch row number
    if inv_alg == 'WLS':
        n_store_data = n_ifg*3+n_im*2+n_im*0.3 #
    else:
        n_store_data = n_ifg*2+n_im*2+n_im*0.3 #not sure
        

    n_patch, patchrow = tools_lib.get_patchrow(width, length, n_store_data, memory_size)


    #%% Display and output settings & parameters
    print('')
    print('Size of image (w,l)    : {}, {}'.format(width, length))
    print('# of all images        : {}'.format(n_im_all))
    print('# of images to be used : {}'.format(n_im))
    print('# of all ifgs          : {}'.format(n_ifg_all))
    print('# of ifgs to be used   : {}'.format(n_ifg))
    print('# of removed ifgs      : {}'.format(n_ifg_bad))
    print('Threshold of used unw  : {}'.format(n_unw_thre))
    print('')
    print('Reference area (X/Y)   : {}:{}/{}:{}'.format(refx1, refx2, refy1, refy2))
    print('Allowed memory size    : {} MB'.format(memory_size))
    print('Number of patches      : {}'.format(n_patch))
    print('Inversion algorism     : {}'.format(inv_alg))
    print('Gamma value            : {}'.format(gamma), flush=True)

    with open(os.path.join(infodir, '13parameters.txt'), "w") as f:
        print('range_samples:  {}'.format(width), file=f)
        print('azimuth_lines:  {}'.format(length), file=f)
        print('wavelength:     {}'.format(wavelength), file=f)
        print('n_im_all:       {}'.format(n_im_all), file=f)
        print('n_im:           {}'.format(n_im), file=f)
        print('n_ifg_all:      {}'.format(n_ifg_all), file=f)
        print('n_ifg:          {}'.format(n_ifg), file=f)
        print('n_ifg_bad:      {}'.format(n_ifg_bad), file=f)
        print('n_unw_thre:     {}'.format(n_unw_thre), file=f)
        print('ref_area:       {}:{}/{}:{}'.format(refx1, refx2, refy1, refy2), file=f)
        print('memory_size:    {} MB'.format(memory_size), file=f)
        print('n_patch:        {}'.format(n_patch), file=f)
        print('inv_alg:        {}'.format(inv_alg), file=f)
        print('gamma:          {}'.format(gamma), file=f)
        print('pixel_spacing_r: {:.2f} m'.format(pixsp_r), file=f)
        print('pixel_spacing_a: {:.2f} m'.format(pixsp_a), file=f)


#%% Ref phase for inversion
    lengththis = refy2-refy1
    countf = width*refy1
    countl = width*lengththis # Number to be read
    ref_unw = []
    for i, ifgd in enumerate(ifgdates):
        unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
        f = open(unwfile, 'rb')
        f.seek(countf*4, os.SEEK_SET) #Seek for >=2nd path, 4 means byte
        
        ### Read unw data (mm) at ref area
        unw = np.fromfile(f, dtype=np.float32, count=countl).reshape((lengththis, width))[:, refx1:refx2]*coef_r2m
        
        unw[unw == 0] = np.nan
        if np.all(np.isnan(unw)):
            print('All nan in ref area in {}.'.format(ifgd))
            print('Rerun LiCSBAS12.')
            return 1
        
        ref_unw.append(np.nanmean(unw))
        
        f.close()


    #%% Open cum.h5 for output
    if os.path.exists(cumh5file): os.remove(cumh5file)
    cumh5 = h5.File(cumh5file, 'w')
    cumh5.create_dataset('imdates', data=[np.int32(imd) for imd in imdates])
    if not np.all(np.abs(np.array(bperp))<=1):# if not dummy
        cumh5.create_dataset('bperp', data=bperp)
    cum = cumh5.require_dataset('cum', (n_im, length, width), dtype=np.float32)
    vel = cumh5.require_dataset('vel', (length, width), dtype=np.float32)
    vconst = cumh5.require_dataset('vintercept', (length, width), dtype=np.float32)
    gap = cumh5.require_dataset('gap', (n_im-1, length, width), dtype=np.int8)

    if width == width_geo and length == length_geo: ## if geocoded
        cumh5.create_dataset('corner_lat', data=lat1)
        cumh5.create_dataset('corner_lon', data=lon1)
        cumh5.create_dataset('post_lat', data=dlat)
        cumh5.create_dataset('post_lon', data=dlon)


    #%% For each patch
    i_patch = 1
    for rows in patchrow:
        print('\nProcess {0}/{1}th line ({2}/{3}th patch)...'.format(rows[1], patchrow[-1][-1], i_patch, n_patch), flush=True)
        start2 = time.time()
        
        #%% Read data
        ### Allocate memory
        lengththis = rows[1] - rows[0]
        n_pt_all = lengththis*width
        unwpatch = np.zeros((n_ifg, lengththis, width), dtype=np.float32)
        
        if inv_alg == 'WLS':
            cohpatch = np.zeros((n_ifg, lengththis, width), dtype=np.float32)
        
        ### For each ifg
        print("  Reading {0} ifg's unw data...".format(n_ifg), flush=True)
        countf = width*rows[0]
        countl = width*lengththis
        for i, ifgd in enumerate(ifgdates):
            unwfile = os.path.join(ifgdir, ifgd, ifgd+'.unw')
            f = open(unwfile, 'rb')
            f.seek(countf*4, os.SEEK_SET) #Seek for >=2nd patch, 4 means byte
            
            ### Read unw data (mm) at patch area
            unw = np.fromfile(f, dtype=np.float32, count=countl).reshape((lengththis, width))*coef_r2m
            unw[unw == 0] = np.nan # Fill 0 with nan
            unw = unw - ref_unw[i]
            unwpatch[i] = unw
            f.close()

            ### Read coh file at patch area for WLS
            if inv_alg == 'WLS':
                cohfile = os.path.join(ifgdir, ifgd, ifgd+'.cc')
                f = open(cohfile, 'rb')
                
                if os.path.getsize(cohfile) == length*width: ## uint8 format
                    f.seek(countf, os.SEEK_SET) #Seek for >=2nd patch
                    cohpatch[i, :, :] = (np.fromfile(f, dtype=np.uint8, count=countl).reshape((lengththis, width))).astype(np.float32)/255
                else: ## old float32 format
                    f.seek(countf*4, os.SEEK_SET) #Seek for >=2nd patch, 4 means byte
                    cohpatch[i, :, :] = np.fromfile(f, dtype=np.float32, count=countl).reshape((lengththis, width))
                cohpatch[cohpatch==0] = np.nan

        unwpatch = unwpatch.reshape((n_ifg, n_pt_all)).transpose() #(n_pt_all, n_ifg)

        ### Calc variance from coherence for WLS
        if inv_alg == 'WLS':
            cohpatch = cohpatch.reshape((n_ifg, n_pt_all)).transpose() #(n_pt_all, n_ifg)
            cohpatch[cohpatch<0.01] = 0.01 ## because negative value possible due to geocode
            cohpatch[cohpatch>0.99] = 0.99 ## because >1 possible due to geocode
            varpatch = (1-cohpatch**2)/(2*cohpatch**2)
            del cohpatch


        #%% Remove points with less valid data than n_unw_thre
        ix_unnan_pt = np.where(np.sum(~np.isnan(unwpatch), axis=1) > n_unw_thre)[0]
        n_pt_unnan = len(ix_unnan_pt)

        unwpatch = unwpatch[ix_unnan_pt,:] ## keep only unnan data
        if inv_alg == 'WLS':
            varpatch = varpatch[ix_unnan_pt,:] ## keep only unnan data

        print('  {}/{} points removed due to not enough ifg data...'.format(n_pt_all-n_pt_unnan, n_pt_all), flush=True)


        #%% Compute number of gaps, ifg_noloop, maxTlen point-by-point
        ns_gap_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
        gap_patch = np.zeros((n_im-1, n_pt_all), dtype=np.int8)
        ns_ifg_noloop_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
        maxTlen_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan

        ### n_gap
        print('\n  Identifing gaps and counting n_gap...', flush=True)
#        ns_unw_unnan4inc = (np.matmul(np.int8(G[:, :, None]), (~np.isnan(unwpatch.T))[:, None, :])).sum(axis=0, dtype=np.int16) #n_ifg, n_im-1, n_pt -> n_im-1, n_pt
        ns_unw_unnan4inc = np.array([(G[:, i]*(~np.isnan(unwpatch))).sum(axis=1, dtype=np.int16) for i in range(n_im-1)]) #n_ifg*(n_pt,n_ifg) -> (n_im-1,n_pt)
        ns_gap_patch[ix_unnan_pt] = (ns_unw_unnan4inc==0).sum(axis=0) #n_pt
        gap_patch[:, ix_unnan_pt] = ns_unw_unnan4inc==0

        del ns_unw_unnan4inc

        ### n_ifg_noloop
        print('  Counting n_ifg_noloop...', flush=True)
        # n_ifg*(n_pt,n_ifg)->(n_loop,n_pt) 
        # Number of ifgs for each loop at eath point.
        # 3 means complete loop, 1 or 2 means broken loop.
        ns_ifg4loop = np.array([
                (np.abs(Aloop[i, :])*(~np.isnan(unwpatch))).sum(axis=1)
                for i in range(n_loop)])
        bool_loop = (ns_ifg4loop==3) #(n_loop,n_pt) identify complete loop only
        
        # n_loop*(n_loop,n_pt)*n_pt->(n_ifg,n_pt)
        # Number of loops for each ifg at eath point.
        ns_loop4ifg = np.array([(
                (np.abs(Aloop[:, i])*bool_loop.T).T*
                (~np.isnan(unwpatch[:, i]))
                ).sum(axis=0) for i in range(n_ifg)]) #

        ns_ifg_noloop_tmp = (ns_loop4ifg==0).sum(axis=0) #n_pt
        ns_nan_ifg = np.isnan(unwpatch).sum(axis=1) #n_pt, nan ifg count
        ns_ifg_noloop_patch[ix_unnan_pt] = ns_ifg_noloop_tmp - ns_nan_ifg

        del bool_loop, ns_ifg4loop, ns_loop4ifg

        ### maxTlen
        _maxTlen = np.zeros((n_pt_unnan), dtype=np.float32) #temporaly
        _Tlen = np.zeros((n_pt_unnan), dtype=np.float32) #temporaly
        for imx in range(n_im-1):
            _Tlen = _Tlen + (dt_cum[imx+1]-dt_cum[imx]) ## Adding dt
            _Tlen[gap_patch[imx, ix_unnan_pt]==1] = 0 ## reset to 0 if gap
            _maxTlen[_maxTlen<_Tlen] = _Tlen[_maxTlen<_Tlen] ## Set Tlen to maxTlen
        maxTlen_patch[ix_unnan_pt] = _maxTlen


        #%% Time series inversion
        print('\n  Small Baseline inversion by {}...\n'.format(inv_alg), flush=True)
        if inv_alg == 'WLS':
            inc_tmp, vel_tmp, vconst_tmp = inv_lib.invert_nsbas_wls(unwpatch, varpatch, G, dt_cum, gamma, n_core)
        else:
            inc_tmp, vel_tmp, vconst_tmp = inv_lib.invert_nsbas(unwpatch, G, dt_cum, gamma, n_core)

        ### Set to valuables
        inc_patch = np.zeros((n_im-1, n_pt_all), dtype=np.float32)*np.nan
        vel_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan
        vconst_patch = np.zeros((n_pt_all), dtype=np.float32)*np.nan

        inc_patch[:, ix_unnan_pt] = inc_tmp
        vel_patch[ix_unnan_pt] = vel_tmp
        vconst_patch[ix_unnan_pt] = vconst_tmp

        ### Calculate residuals
        res_patch = np.zeros((n_ifg, n_pt_all), dtype=np.float32)*np.nan
        res_patch[:, ix_unnan_pt] = unwpatch.T-np.dot(G, inc_tmp)

        res_sumsq = np.nansum(res_patch**2, axis=0)
        res_n = np.float32((~np.isnan(res_patch)).sum(axis=0))
        res_n[res_n==0] = np.nan # To avoid 0 division
        res_rms_patch = np.sqrt(res_sumsq/res_n)

        ### Cumulative displacememt
        cum_patch = np.zeros((n_im, n_pt_all), dtype=np.float32)*np.nan
        cum_patch[1:, :] = np.cumsum(inc_patch, axis=0)

        ## Fill 1st image with 0 at unnan points from 2nd images
        bool_unnan_pt = ~np.isnan(cum_patch[1, :])
        cum_patch[0, bool_unnan_pt] = 0

        ## Drop (fill with nan) interpolated cum by 2 continuous gaps
        for i in range(n_im-2): ## from 1->n_im-1
            gap2 = gap_patch[i, :]+gap_patch[i+1, :] 
            bool_gap2 = (gap2==2) ## true if 2 continuous gaps for each point
            cum_patch[i+1, :][bool_gap2] = np.nan

        ## Last (n_im th) image. 1 gap means interpolated
        cum_patch[-1, :][gap_patch[-1, :]==1] = np.nan

        
        #%% Output data and image
        ### cum.h5 file
        cum[:, rows[0]:rows[1], :] = cum_patch.reshape((n_im, lengththis, width))
        vel[rows[0]:rows[1], :] = vel_patch.reshape((lengththis, width))
        vconst[rows[0]:rows[1], :] = vconst_patch.reshape((lengththis, width))
        gap[:, rows[0]:rows[1], :] = gap_patch.reshape((n_im-1, lengththis, width))
        
        ### Others
        openmode = 'w' if rows[0] == 0 else 'a' #w only 1st patch

        ## For each imd. cum and inc
        for imx, imd in enumerate(imdates):
            ## Incremental displacement
            if imd == imdates[-1]: continue #skip last
            incfile = os.path.join(incdir, '{0}_{1}.inc'.format(imd, imdates[imx+1]))
            with open(incfile, openmode) as f:
                inc_patch[imx, :].tofile(f)

        ## For each ifgd. resid
        for i, ifgd in enumerate(ifgdates):
            resfile = os.path.join(resdir, '{0}.res'.format(ifgd))
            with open(resfile, openmode) as f:
                res_patch[i, :].tofile(f)

        ## velocity and noise indecies in results dir
        names = ['vel', 'vintercept', 'resid_rms', 'n_gap', 'n_ifg_noloop', 'maxTlen']
        data = [vel_patch, vconst_patch, res_rms_patch, ns_gap_patch, ns_ifg_noloop_patch, maxTlen_patch]
        for i in range(len(names)):
            file = os.path.join(resultsdir, names[i])
            with open(file, openmode) as f:
                data[i].tofile(f)
            

        #%% Finish patch
        elapsed_time2 = int(time.time()-start2)
        hour2 = int(elapsed_time2/3600)
        minite2 = int(np.mod((elapsed_time2/60),60))
        sec2 = int(np.mod(elapsed_time2,60))
        print("  Elapsed time for {0}th patch: {1:02}h {2:02}m {3:02}s".format(i_patch, hour2, minite2, sec2), flush=True)

        i_patch += 1 #Next patch count


    #%% Find stable ref point
    print('\nFind stable reference point...', flush=True)
    ### Compute RMS of time series with reference to all points
    sumsq_cum_wrt_med = np.zeros((length, width), dtype=np.float32)
    for i in range(n_im):
        sumsq_cum_wrt_med = sumsq_cum_wrt_med + (cum[i, :, :]-np.nanmedian(cum[i, :, :]))**2
    rms_cum_wrt_med = np.sqrt(sumsq_cum_wrt_med/n_im)

    ### Mask by minimum n_gap
    n_gap = io_lib.read_img(os.path.join(resultsdir, 'n_gap'), length, width)
    min_n_gap = np.nanmin(n_gap)
    mask_n_gap = np.float32(n_gap==min_n_gap)
    mask_n_gap[mask_n_gap==0] = np.nan
    rms_cum_wrt_med = rms_cum_wrt_med*mask_n_gap
    
    ### Find stable reference
    min_rms = np.nanmin(rms_cum_wrt_med)
    refy1s, refx1s = np.where(rms_cum_wrt_med==min_rms)
    refy1s, refx1s = refy1s[0], refx1s[0] ## Only first index
    refy2s, refx2s = refy1s+1, refx1s+1
    print('Selected ref: {}:{}/{}:{}'.format(refx1s, refx2s, refy1s, refy2s), flush=True)

    ### Rerferencing cumulative displacement  and vel to new stable ref
    for i in range(n_im):
        cum[i, :, :] = cum[i, :, :] - cum[i, refy1s, refx1s]
    vel = vel - vel[refy1s, refx1s]
    vconst = vconst - vconst[refy1s, refx1s]

    ### Save image
    rms_cum_wrt_med_file = os.path.join(resultsdir, 'rms_cum_wrt_med')
    with open(rms_cum_wrt_med_file, 'w') as f:
        rms_cum_wrt_med.tofile(f)

    pngfile = os.path.join(resultsdir, 'rms_cum_wrt_med.png')
    plot_lib.make_im_png(rms_cum_wrt_med, pngfile, cmap_noise_r, 'RMS of cum wrt median (mm)', np.nanpercentile(rms_cum_wrt_med, 1), np.nanpercentile(rms_cum_wrt_med, 99))

    ### Save ref
    cumh5.create_dataset('refarea', data='{}:{}/{}:{}'.format(refx1s, refx2s, refy1s, refy2s))
    refsfile = os.path.join(infodir, 'ref_stable.txt')
    with open(refsfile, 'w') as f:
        print('{}:{}/{}:{}'.format(refx1s, refx2s, refy1s, refy2s), file=f)

    if width == width_geo and length == length_geo: ## Geocoded
        ### Make ref_stable.kml
        reflat = lat1+dlat*refy1
        reflon = lon1+dlon*refx1
        make_point_kml(reflat, reflon, os.path.join(infodir, 'ref_stable.kml'))


    #%% Close h5 file
    cumh5.close()


    #%% Output png images
    print('\nOutput png images...', flush=True)
    ### Incremental displacement
    for imx, imd in enumerate(imdates):
        if imd == imdates[-1]: continue #skip last for increment
        ## Comparison of increment and daisy chain pair
        ifgd = '{}_{}'.format(imd, imdates[imx+1])
        incfile = os.path.join(incdir, '{}.inc'.format(ifgd))
        unwfile = os.path.join(ifgdir, ifgd, '{}.unw'.format(ifgd))
        pngfile = os.path.join(incdir, '{}.inc_comp.png'.format(ifgd))
        
        inc = io_lib.read_img(incfile, length, width)
        
        try:
            unw = io_lib.read_img(unwfile, length, width)*coef_r2m
            ix_ifg = ifgdates.index(ifgd)
            unw = unw - ref_unw[ix_ifg]
        except:
            unw = np.zeros((length, width), dtype=np.float32)*np.nan

        ### Output png for comparison
        data3 = [np.angle(np.exp(1j*(data/coef_r2m/cycle))*cycle) for data in [unw, inc, inc-unw]]
        title3 = ['Daisy-chain IFG ({}pi/cycle)'.format(cycle*2), 'Inverted ({}pi/cycle)'.format(cycle*2), 'Difference ({}pi/cycle)'.format(cycle*2)]
        pngfile = os.path.join(incdir, '{}.increment.png'.format(ifgd))
        plot_lib.make_3im_png(data3, pngfile, 'insar', title3, vmin=-np.pi, vmax=np.pi, cbar=False)

        if not keep_incfile:
            os.remove(incfile)

    ### Residual for each ifg. png and txt.
    with open(restxtfile, "w") as f:
        print('# RMS of residual (mm)', file=f)
    for ifgd in ifgdates:
        infile = os.path.join(resdir, '{}.res'.format(ifgd))
        resid = io_lib.read_img(infile, length, width)
        resid_rms = np.sqrt(np.nanmean(resid**2))
        with open(restxtfile, "a") as f:
            print('{} {:5.2f}'.format(ifgd, resid_rms), file=f)
        
        pngfile = infile+'.png'
        title = 'Residual (mm) of {} (RMS:{:.2f}mm)'.format(ifgd, resid_rms)
        plot_lib.make_im_png(resid, pngfile, cmap_vel, title, -wavelength/2*1000, wavelength/2*1000)
        

        if not keep_incfile:
            os.remove(infile)
    
    ### Velocity and noise indices
    cmins = [None, None, None, None, None, None]
    cmaxs = [None, None, None, None, None, None]
    cmap_noise = SCM.turku
    cmaps = [cmap_vel, cmap_vel, cmap_noise_r, cmap_noise_r, cmap_noise_r, cmap_noise]
    titles = ['Velocity (mm/yr)', 'Intercept of velocity (mm)', 'RMS of residual (mm)', 'Number of gaps in SB network', 'Number of ifgs with no loops', 'Max length of connected SB network (yr)']

    for i in range(len(names)):
        file = os.path.join(resultsdir, names[i])
        data = io_lib.read_img(file, length, width)

        pngfile = file+'.png'
        
        ## Get color range if None
        if cmins[i] is None:
            cmins[i] = np.nanpercentile(data, 1)
        if cmaxs[i] is None:
            cmaxs[i] = np.nanpercentile(data, 99)
        if cmins[i] == cmaxs[i]: cmins[i] = cmaxs[i]-1

        plot_lib.make_im_png(data, pngfile, cmaps[i], titles[i], cmins[i], cmaxs[i])


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output directory: {}\n'.format(os.path.relpath(tsadir)))


#%% main
if __name__ == "__main__":
    sys.exit(main())
