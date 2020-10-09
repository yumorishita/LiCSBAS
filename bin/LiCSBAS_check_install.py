#!/usr/bin/env python3
"""
v1.0 20201009 Yu Morishita, GSI

This script checks if LiCSBAS install is OK or not.

"""
#%% Change log
'''
v1.0 20190729 Yu Morishita, Uni of Leeds and GSI
 - Original implementation
'''

#%% Import
from importlib import import_module
import platform
import shutil

#%% Main
if __name__ == "__main__":
    flag = True
    modules = ['astropy',
                'bs4',
                'gdal',
                'h5py',
                'matplotlib',
                'numpy',
                'requests',
                'statsmodels',
               ]
    
    print('\nPython version: {}'.format(platform.python_version()))
    pyver = platform.python_version_tuple()
    if int(pyver[0]) < 3 or int(pyver[1]) < 6:
        print('  ERROR: must be >= 3.6'.format())
        flag = False
    else:
        print('  OK')

    print('\nCheck required modues')
    for module in modules:
        try:
            imported = import_module(module)
            
        except Exception as err:
            print('  ERROR: {}'.format(err))
            flag = False
        else:
            if module == 'gdal':
                ver = imported.VersionInfo()       
            else:
                ver = imported.__version__
            
            print('  {}({}) OK'.format(module, ver))

    print('\nCheck LiCSBAS commands')
    rc = shutil.which('LiCSBAS01_get_geotiff.py')
    if rc is None:
        print('  ERROR: PATH is not set to LiCSBAS commands')
        flag = False
    else:
        print('  OK')

 
    print('\nCheck LiCSBAS library')
    try:
         imported = import_module('LiCSBAS_io_lib')
    except Exception:
         print('  ERROR: PYTHONPATH is not set to LiCSBAS library')
         flag = False
    else:
         print('  OK')

    if flag:
        print('\nLiCSBAS install is OK\n')
    else:
        print('\nERROR: LiCSBAS install is NOT OK\n')
        
