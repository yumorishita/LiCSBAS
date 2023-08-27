#!/usr/bin/env python3
"""
v1.1 20220130 Qi Ou, Uni of Leeds
v1.0 20200225 Yu Morishita, Uni of Leeds and GSI

========
Overview
========
This script creates a png file (or in other formats) of SB network. A Gap of the network are denoted by a black vertical line if a gap exist. Bad ifgs can be denoted by red lines.

=====
Usage
=====
LiCSBAS_plot_network.py -i ifg_list -b bperp_list [-o outpngfile] [-r bad_ifg_list] [--not_plot_bad] [-s]

 -i  Text file of ifg list (format: yyymmdd_yyyymmdd)
 -b  Text file of bperp list (format: yyyymmdd yyyymmdd bperp dt)
 -o  Output image file (Default: netowrk.png)
     Available file formats: png, ps, pdf, or svg
     (see manual for matplotlib.pyplot.savefig)
 -r  Text file of bad ifg list to be plotted with red lines (format: yyymmdd_yyyymmdd)
 --not_plot_bad  Not plot bad ifgs with red lines
 -s Separate strongly connected component from weak connections
 -m Allowed months (eg. 3.4.5.12, months separated by period, order doesn't matter)
 -t lower bound of temporal baseline
 -e keep edge cuts (option when -s is used)
 -n keep node cuts (option when -s is used)
 -l label for "red lines" when -r is used
 --skip_node_cuts do not check for node cuts (option when -s is used)
 --not_strict keep ifg as long as one epoch is in an allowed month (option when -m is used)


"""

#%% Change log
'''
v1.2 20230207 Qi Ou, Uni of Leeds
 - keep only summer epochs
 - remove short temporal baseline ifgs
v1.1 20230130 Qi Ou, Uni of Leeds
 - Separate strongly connected component from weak links
v1.0 20200225 Yu Morishita, Uni of Leeds and GSI
 - Original implementationf
'''

#%% Import
import getopt
import os
import sys
import time
import numpy as np
import LiCSBAS_io_lib as io_lib
import LiCSBAS_tools_lib as tools_lib
import LiCSBAS_plot_lib as plot_lib

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
    ver=1.1; date=20220209; author="Q. Ou"
    print("\n{} ver{} {} {}".format(os.path.basename(argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(argv[0]), ' '.join(argv[1:])), flush=True)

    #%% Set default
    ifgfile = []
    bperpfile = []
    pngfile = 'network.png'
    bad_ifgfile = []
    plot_bad_flag = True
    strong_connected = False
    suffix = ""
    strict = True
    months = False
    thresh = False
    remove_edge_cuts = True
    remove_node_cuts = True
    skip_node_cuts = False
    label_name = None


    #%% Read options
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "hi:b:o:r:sm:t:enl:", ["help", "not_plot_bad", "not_strict", "skip_node_cuts"])
        except getopt.error as msg:
            raise Usage(msg)
        for o, a in opts:
            if o == '-h' or o == '--help':
                print(__doc__)
                return 0
            elif o == '-i':
                ifgfile = a
            elif o == '-b':
                bperpfile = a
            elif o == '-o':
                pngfile = a
            elif o == '-r':
                bad_ifgfile = a
            elif o == '--not_plot_bad':
                plot_bad_flag = False
            elif o == '-s':
                strong_connected = True
            elif o == '-m':
                months = a
            elif o == '-t':
                thresh = float(a)
            elif o == '--not_strict':
                strict = False
            elif o == '-e':
                remove_edge_cuts = False
            elif o == '-n':
                remove_node_cuts = False
            elif o == '--skip_node_cuts':
                skip_node_cuts = True
            elif o == '-l':
                label_name = a


        if not ifgfile:
            raise Usage('No ifg list given, -i is not optional!')
        elif not os.path.exists(ifgfile):
            raise Usage('No {} exists!'.format(ifgfile))
        elif not bperpfile:
            raise Usage('No bperp list given, -b is not optional!')
        elif not os.path.exists(bperpfile):
            raise Usage('No {} exists!'.format(bperpfile))


    except Usage as err:
        print("\nERROR:", file=sys.stderr, end='')
        print("  "+str(err.msg), file=sys.stderr)
        print("\nFor help, use -h or --help.\n", file=sys.stderr)
        return 2


    #%% Read info
    ifgdates = io_lib.read_ifg_list(ifgfile)
    basename = os.path.basename(ifgfile).split('.')[0]

    if bad_ifgfile:
        bad_ifgdates = io_lib.read_ifg_list(bad_ifgfile)
        # ifgdates = list(set(ifgdates)-set(bad_ifgdates))
    else:
        bad_ifgdates = []

    if thresh:
        dt = tools_lib.calc_temporal_baseline(ifgdates)
        shortifg = [ifg for ifg, t in zip(ifgdates, dt) if t <= thresh]
        ifgdates = list(set(ifgdates)-set(shortifg))
        suffix = suffix + "_dt_gt_{}".format(int(thresh))
        # export list
        with open("{}_dt_le_{}.txt".format(basename, int(thresh)), 'w') as f:
            for i in shortifg:
                print('{}'.format(i), file=f)
        with open("{}_dt_gt_{}.txt".format(basename, int(thresh)), 'w') as f:
            for i in ifgdates:
                print('{}'.format(i), file=f)

    if months:
        ifgdates_allowed_months = tools_lib.select_ifgs_by_months(ifgdates, allowed_month=months, strict=strict)
        ifgdates_other_months = list(set(ifgdates) - set(ifgdates_allowed_months))
        ifgdates = ifgdates_allowed_months
        suffix = suffix + "_months{}".format(months)
        # export list
        if strict:
            with open("{}{}_either_out.txt".format(basename, suffix), 'w') as f:
                for i in ifgdates_other_months:
                    print('{}'.format(i), file=f)
            with open("{}{}_both_in.txt".format(basename, suffix), 'w') as f:
                for i in ifgdates:
                    print('{}'.format(i), file=f)
        else:
            with open("{}{}_both_out.txt".format(basename, suffix), 'w') as f:
                for i in ifgdates_other_months:
                    print('{}'.format(i), file=f)
            with open("{}{}_either_in.txt".format(basename, suffix), 'w') as f:
                for i in ifgdates:
                    print('{}'.format(i), file=f)

    # extract bperp after modifying ifgdates
    imdates = tools_lib.ifgdates2imdates(ifgdates)
    bperp = io_lib.read_bperp_file(bperpfile, imdates)

    if strong_connected:
        strong_links, weak_links, edge_cuts, node_cuts = tools_lib.separate_strong_and_weak_links(
            ifgdates, "{}_stats.txt".format(basename), remove_edge_cuts=remove_edge_cuts, remove_node_cuts=remove_node_cuts, skip_node_cuts=skip_node_cuts)
        pngfile = "{}{}_strongly_connected_network.png".format(basename, suffix)
        plot_lib.plot_strong_weak_cuts_network(ifgdates, bperp, weak_links, edge_cuts, node_cuts, pngfile, plot_weak=True)
        # export weak links
        with open("{}{}_weak_links.txt".format(basename, suffix), 'w') as f:
            for i in weak_links:
                print('{}'.format(i), file=f)

        # export strong links
        with open("{}{}_strong_links.txt".format(basename, suffix), 'w') as f:
            for i in strong_links:
                print('{}'.format(i), file=f)

        # export edge cuts
        print("{} ifgs are edge cuts".format(len(edge_cuts)))
        with open("{}_edge_cuts.txt".format(basename), 'w') as f:
            for i in edge_cuts:
                print('{}'.format(i), file=f)
                print('{}'.format(i))

        # export edge cuts
        print("{} epochs are node cuts".format(len(node_cuts)))
        with open("{}_node_cuts.txt".format(basename), 'w') as f:
            for i in node_cuts:
                print('{}'.format(i), file=f)
                print('{}'.format(i))

    else:    #%% Plot image
        pngfile = "{}{}_network.png".format(basename, suffix)
        plot_lib.plot_network(ifgdates, bperp, bad_ifgdates, pngfile, plot_bad_flag, label_name=label_name)


    #%% Finish
    elapsed_time = time.time()-start
    hour = int(elapsed_time/3600)
    minite = int(np.mod((elapsed_time/60),60))
    sec = int(np.mod(elapsed_time,60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour,minite,sec))

    print('\n{} Successfully finished!!\n'.format(os.path.basename(argv[0])))
    print('Output: {}\n'.format(pngfile), flush=True)


#%% main
if __name__ == "__main__":
    sys.exit(main())
