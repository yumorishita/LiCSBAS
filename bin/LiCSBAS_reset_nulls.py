#!/usr/bin/env python3
"""
========
Overview
========
This script will reset any nulled data to a pre-defined state

===============
Input & output files
===============

Inputs in GEOCml*/ :
 - yyyymmdd_yyyymmdd/
   - yyyymmdd_yyyymmdd_orig*.unw
   - yyyymmdd_yyyymmdd.unw

=====
Usage
=====
LiCSBAS_reset_nulls.py [-h] [-d GEOC_DIR] [--reset_all] [--reset_NoLoop] [--reset_LoopErr]
"""
#%% Change log
'''
v1.0 20230803 Jack McGrath
 - Original implementation
 
'''

import os
import re
import sys
import glob
import time
import shutil
import argparse
import numpy as np
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
    parser.add_argument('-f', dest='frame_dir', default="./", help="Frame directory")
    parser.add_argument('-d', dest='geoc_dir', default="GEOCml10GACOS", help="GEOCdir containing IFGs to be reset")
    parser.add_argument('-t', dest='ts_dir', default="TS_$GEOCdir", help="TS_GEOCdir containing info/11bad_ifg.txt")
    parser.add_argument('--reset_all', dest='reset_all', default=False, action='store_true', help='Reset all IFGs to original, unnulled state')
    parser.add_argument('--reset_NoLoop', dest='reset_NoLoop', default=False, action='store_true', help='Add noLoops back into the IFGs (LiCSBAS130_remove_noloops.py must have been the last nullification)')
    parser.add_argument('--reset_LoopErr', dest='reset_LoopErr', default=False, action='store_true', help='Add Loop Errors back into the IFGs (LiCSBAS12_loop_closure.py must have been the last nullification)')
    args = parser.parse_args()

def start():
    global start_time
    # intialise and print info on screen
    start_time = time.time()
    ver="1.0"; date=20230706; author="Jack McGrath"
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
    print('IFG directory: {}\n'.format(os.path.relpath(ifgdir)))

def set_input_output():
    global ifgdir, ifglist, noLoopList, bad_ifg_list

    ifgdir = os.path.abspath(os.path.join(args.frame_dir, args.geoc_dir))
    ifglist = glob.glob(os.path.join(ifgdir, '20*'))
    if '$' in args.ts_dir:
        bad_ifg11file = os.path.join(args.frame_dir, 'TS_' + args.geoc_dir, 'info', '11bad_ifg.txt')
    else:
        bad_ifg11file = os.path.join(args.frame_dir, args.ts_dir, 'info', '11bad_ifg.txt')

    ### Read bad_ifg11 and 12
    if os.path.exists(bad_ifg11file):
        bad_ifg_list = io_lib.read_ifg_list(bad_ifg11file)
    else:
        bad_ifg_list = []
    
    # Set reset flags
    if args.reset_NoLoop and args.reset_LoopErr:
        print('--reset_NoLoop (LiCSBAS13_remove_noloops) and --reset_LoopErr (LiCSBAS12_loop_closure) both set')
        print('Setting --reset_all instead')
        args.reset_all = True

    if args.reset_all:
        args.reset_NoLoop = False
        args.reset_LoopErr = False

    if not args.reset_all and not args.reset_NoLoop and not args.reset_LoopErr:
        raise Exception('No reset options selected......')
    
    if args.reset_all or args.reset_NoLoop:
        noLoopDir = os.path.join(ifgdir, 'no_loop_ifg')
        if os.path.exists(noLoopDir):
            noLoopList = glob.glob(os.path.join(noLoopDir, '20*'))
        else:
            noLoopList = []

def reset_all():
    # Return No Loop IFGs to the main folder
    for ifg in noLoopList:
        ifgd = re.split('/', ifg)[-1]
        shutil.move(ifg, os.path.join(ifgdir, ifgd))

    ifglist = glob.glob(os.path.join(ifgdir, '20*'))

    for ifg in ifglist:
        ifgd = re.split('/', ifg)[-1]
        if os.path.exists(os.path.join(ifg, ifgd + '_orig.unw')):
            shutil.move(os.path.join(ifg, ifgd + '_orig.unw'), os.path.join(ifg, ifgd + '.unw'))
            for backup in glob.glob(os.path.join(ifg, '*orig*')):
                if os.path.islink(backup):
                    os.unlink(backup)
                else:
                    os.remove(backup)
        elif os.path.exists(os.path.join(ifg, ifgd + '.unw')):
            if ifgd not in bad_ifg_list:
                print('CAUTION: NO {}_orig.unw exists to backup from!'.format(ifgd))
            else:
                print('CAUTION: {0} identified as a bad by step 11. No nulling occurred, so no {0}_orig.unw exists to backup from!'.format(ifgd))
        else:
            print('WARNING: NO {0}.unw OR {0}_orig.unw EXISTS in {1}!'.format(ifgd, os.path.dirname(ifgdir)))

def reset_null():
    if args.reset_LoopErr:
        resetcode=12
    elif args.reset_NoLoop:
        resetcode=13

        # Return No Loop IFGs to the main folder
        for ifg in noLoopList:
            ifgd = re.split('/', ifg)[-1]
            shutil.move(ifg, os.path.join(ifgdir, ifgd))
    else:
        resetcode = 999 # undefined

    ifglist = glob.glob(os.path.join(ifgdir, '20*'))

    for ifg in ifglist:
        ifgd = re.split('/', ifg)[-1]        
        backups = glob.glob(os.path.join(ifg, ifgd + '_orig*{}.unw'.format(resetcode)))
        if len(backups) == 0:
            print('WARNING: No backup \'_orig*{}.unw\' files found in {}! Skipping.....'.format(resetcode, ifgd))
        elif len(backups) > 1:
            print('WARNING: Ambiguous as to which file is the last backup in {}! Skipping.....'.format(ifgd))
        else:
            backup = backups[0]
            # Check if backup is a symlink
            if os.path.islink(backup):
                # Remove symlink, reset with _orig.unw
                os.unlink(backup)
                shutil.move(os.path.join(ifg, ifgd + '_orig.unw'), os.path.join(ifg, ifgd + '.unw'))
            else:
                shutil.move(backup, os.path.join(ifg, ifgd + '.unw'))

def main():

    start()
    init_args()
    set_input_output()

    if args.reset_all:
        reset_all()
    else:
        reset_null()

    finish()

if __name__ == "__main__":
    main()