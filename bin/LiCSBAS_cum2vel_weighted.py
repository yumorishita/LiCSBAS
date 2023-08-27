#!/usr/bin/env python3
"""
========
Overview
========
This script takes in a cumulative displacement file in .h5 format and
    - estimates ramp coefficients per epoch and model the coef time series with a linear + seasonal model
    - calculates temporal residuals from modelling ramp coef time series
    - calculates std of spatial residuals from deramped displacements
    - weight the inversion for linear + seasonal components from the time series
    - calculate standard error of model parameters from reduced chi-squares and covariance matrix
    - optionally export time series with seasonal component removed

Input:
    - [cum.h5] any .h5 file with a 3D array and a imdate vector

Outputs:
    - xx.h5.png [--plot_cum]
    - xx.h5_vel [-l] and .png [--plot_png]
    - xx.h5_vstd [-l] and .png [--plot_png]
    - xx.h5_amp [-s] and .png [--plot_png]
    - xx.h5_dt [-s] and .png [--plot_png]
    - xx.h5_ramp_coefs_resid_flat_std.png [-p]
    - xx.de_seasoned.h5 [--de_season]

"""

import numpy as np
import h5py as h5
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import SCM
import time
import os
import sys
import statsmodels.api as sm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s -- %(levelname)s -- %(message)s')
logger = logging.getLogger('pre_co_post_seismic.log')

# changelog
ver = "1.0"; date = 20230815; author = "Qi Ou, ULeeds"
"""
- estimate ramp coefficients per epoch and model the coef time series with a linear + seasonal model
- calculate the weight using both temporal residuals from ramp coef time series and spatial residuals from deramped displacements
- weight the inversion for linear + seasonal components from the time series
- calculate standard error of model parameters from reduced chi-squares and covariance matrix
- optionally export time series with seasonal component removed
"""



class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """
    Use a multiple inheritance approach to use features of both classes.
    The ArgumentDefaultsHelpFormatter class adds argument default values to the usage help message
    The RawDescriptionHelpFormatter class keeps the indentation and line breaks in the ___doc___
    """
    pass


def init_args():
    global args
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
    parser.add_argument('-i', dest='cumfile', default="cum.h5", type=str, help="input .h5 file")
    parser.add_argument('-d', dest='downsample', default=10, type=int, help="downsample cumfile for ramp estimation and time series plotting")
    parser.add_argument('-p', dest='ramp', default=False, action='store_true', help="model planar ramp and use std of flattened time series to weight vel inversion")
    parser.add_argument('-s', dest='season', default=False, action='store_true', help="model seasonal trend")
    parser.add_argument('-l', dest='linear', default=False, action='store_true', help="model linear trend")
    parser.add_argument('-r', dest='ref', default=False, action='store_true', help="reference to the center of the image, useful for forward model displacements")
    parser.add_argument('--heading', type=float, default=0, choices=[-10, -170, 0], help="heading azimuth, -10 for asc, -170 for dsc, 0 if in radar coordinates, required if using deramp")
    parser.add_argument('--plot_cum', default=False, action='store_true', help="plot 3D time series")
    parser.add_argument('--plot_vel', default=False, action='store_true', help="plot vel components and uncertainties")
    parser.add_argument('--de_season', default=False, action='store_true', help="save the time series without the seasonal component, requires -s")
    parser.add_argument('-n', dest='count_nans', default=False, action='store_true', help="produce a map of number of nan epochs in the time series")
    # parser.add_argument('-t', dest='delta_t', default="xx.dt", type=str, help="this is an output of LiCSBAS_cum2vel.py")
    # parser.add_argument('-a', dest='amp', default="xx.amp", type=str, help="this is an output of LiCSBAS_cum2vel.py")
    args = parser.parse_args()


def start():
    global start_time
    start_time = time.time()
    print("\n{} ver{} {} {}".format(os.path.basename(sys.argv[0]), ver, date, author), flush=True)
    print("{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)


def finish():
    elapsed_time = time.time() - start_time
    hour = int(elapsed_time / 3600)
    minite = int(np.mod((elapsed_time / 60), 60))
    sec = int(np.mod(elapsed_time, 60))
    print("\nElapsed time: {0:02}h {1:02}m {2:02}s".format(hour, minite, sec))
    print("\n{} {}".format(os.path.basename(sys.argv[0]), ' '.join(sys.argv[1:])), flush=True)
    # print('Output: {}\n'.format(os.path.relpath(args.outfile)))


def plot_cum_grid(cum, imdates, suptitle, png):
    print("Plotting {}".format(png))
    # decide dimension of plotting grid
    n_im, length, width = cum.shape
    n_row = int(np.sqrt(n_im))
    n_col = int(np.ceil(n_im / n_row))

    vmin_list = []
    vmax_list = []
    for i in np.arange(n_im):
        vmin_list.append(np.nanpercentile(cum[i, :, :], 1))
        vmax_list.append(np.nanpercentile(cum[i, :, :], 99))
    vmin = min(vmin_list)
    vmax = max(vmax_list)

    fig, ax = plt.subplots(n_row, n_col, sharex='all', sharey='all', figsize=(n_col * width / length, n_row))
    for i in np.arange(n_im):
        row = i // n_col
        col = i % n_col
        # print(i, row, col)
        im = ax[row, col].imshow(cum[i, :, :], vmin=vmin, vmax=vmax, cmap=SCM.roma.reversed())
        ax[row, col].set_title(imdates[i])
    plt.suptitle(suptitle, fontsize='xx-large')
    plt.tight_layout()
    fig.colorbar(im, ax=ax, label="Displacement, mm")
    plt.savefig(png, bbox_inches='tight')
    plt.close()


def fit_plane(z, theta=0):
    """Fit a plane to data.
    Parameters
    ----------
    z : `numpy.ndarray`
        2D array of z values
    theta : heading angle in radian
    """
    yy, xx = np.indices(z.shape)
    yy = -yy  # minus to make y axis positive upward, otherwise indices increases down the rows
    ones = np.ones(z.shape)

    pts = np.isfinite(z)
    coefs = np.linalg.lstsq(np.stack([xx[pts], yy[pts], ones[pts]]).T, z[pts].flatten(), rcond=None)[0]

    plane_fit = coefs[0] * xx + coefs[1] * yy + coefs[2]

    # rotate axis
    range_coef = coefs[0] * np.cos(theta) + coefs[1] * np.sin(theta)
    azi_coef = coefs[0] * -np.sin(theta) + coefs[1] * np.cos(theta)
    return plane_fit, range_coef, azi_coef


def plot_ramp_coef_time_series(epochs, range_coefs, azi_coefs, flat_std, detrended_flat_std, weights, coef_resid, sig, wlsfit):
    """
    Plot 3 panels.
    Top: time series of ramp coefs weighted least square modeled.
    Middle: Residual time series of ramp coefs
    Bottom: Time series of the standard deviation of flattened cum
    """
    # plot time series of ramp parameters
    fig, [ax1, ax2, ax3] = plt.subplots(3, figsize=(8, 8), sharex='all')
    size_scale = 5 / np.std(weights)
    ax1.scatter(epochs, range_coefs, s=size_scale * weights, label="range_coef_weights")
    ax1.scatter(epochs, azi_coefs, s=size_scale * weights, label="azi_coef_weights")
    ax1.plot(epochs, range_coefs)
    ax1.plot(epochs, azi_coefs)
    ax1.plot(epochs, wlsfit.fittedvalues, label='wls.model')
    ax2.plot(epochs, wlsfit.resid, label='wls.resid')
    ax3.plot(epochs, flat_std, label="flat_std", color='C2')
    ax3.plot(epochs, detrended_flat_std, label="detrended_flat_std", color='C2', linestyle='dashed')
    ax3.plot(epochs, coef_resid, label="scaled_coef_resid", color='C3')
    ax3.plot(epochs, sig, label="sig", color='C4')

    ax1.set_ylabel("Ramp rate, unit/pixel")
    ax2.set_ylabel("Residual ramp rate, unit/pixel")
    ax3.set_ylabel("Std of flattened displacement")
    ax3.set_xlabel("Epoch")
    ax1.set_title(args.cumfile)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.tight_layout()
    plt.savefig(args.cumfile + "_ramp_coefs_resid_flat_std.png")
    plt.close()


def wls_batch(d, G, sig):
    """
    Perform weighted least squares inversion to all pixels without nans through matrix manipulation
    @param d: n_im x n_pt
    @param G: n_im x n_para
    @param sig: n_im
    @return:
    inverted_params : n_para x n_pt
    standard_errors : n_para x n_pt
    """
    # weighted least squares inversion
    wlsfit = sm.WLS(d, G, weights=1 / sig ** 2).fit()
    # calculate standard error of the model parameters: (wlsfit.bse) = [sqrt(reduced_chi_square * Cov_m)]
    chi_square = np.sum(np.square(wlsfit.resid / sig[:, np.newaxis]), axis=0)
    reduced_chi_square = chi_square / wlsfit.df_resid
    cov_d = np.diag(np.square(sig).transpose())
    cov_m = np.linalg.inv(np.dot(np.dot(G.transpose(), np.linalg.inv(cov_d)), G))
    param_errors = np.sqrt(np.outer(np.diag(cov_m), reduced_chi_square))
    return wlsfit.params, param_errors, wlsfit.resid


def wls_pixel_wise(d, G, sig):
    """
    Perform weighted least squares inversion to pixels with nans point by point by dropping nans
    @param d: n_im x n_pt
    @param G: n_im x n_para
    @param sig: n_im
    @return:
    inverted_params : n_para x n_pt
    standard_errors : n_para x n_pt
    residual_cube : n_im x n_pt

    """
    params = np.zeros((G.shape[1], d.shape[1]))
    errors = np.zeros((G.shape[1], d.shape[1]))
    res = np.zeros(d.shape)

    for i in np.arange(d.shape[1]):
        if d.shape[1] > 1000:
            if i % 100 == 0:
                print("  Solving {} / {} pixels".format(i, d.shape[1]), end="\r")
        # weighted least squares inversion
        mask = ~np.isnan(d[:, i])
        masked_d = d[:, i][mask]
        masked_G = G[mask]
        masked_sig = sig[mask]
        wlsfit = sm.WLS(masked_d, masked_G, weights=1 / masked_sig ** 2).fit()
        params[:, i] = wlsfit.params
        errors[:, i] = wlsfit.bse
        res[mask, i] = wlsfit.resid

    return params, errors, res


def parallel_wls_pixel_wise(d, G, sig):
    from functools import partial
    import multiprocessing as multi

    try:
        threads = min(len(os.sched_getaffinity(0)), 8)  # maximum use 8 cores
    except:
        threads = multi.cpu_count()

    # slicing cubes for multi-processing
    if threads > 1:
        d_slices = np.array_split(d, threads, axis=1)
        pool = multi.Pool(processes=threads)
        run_wls_pixel_wise = partial(wls_pixel_wise, G=G, sig=sig)
        result_slices = pool.map(run_wls_pixel_wise, d_slices)
        params = np.concatenate(list(zip(*result_slices))[0], axis=1)
        errors = np.concatenate(list(zip(*result_slices))[1], axis=1)
        res = np.concatenate(list(zip(*result_slices))[2], axis=1)
        logger.info("Result concatenation done...")
    else:
        params, errors, res = wls_pixel_wise(d, G, sig)
    return params, errors, res


def calc_vel_and_err(cum, G, sig):
    """
    Calculate model and standard errors using weighted least squares inversion, or ordinary least squares inversion if sig = ones.
    All data have values
    @param cum: n_im x length x width
    @param G: n_im x n_para
    @param sig: n_im
    @return:
    inverted_params : n_para x length x width
    standard_errors : n_para x length x width
    """
    logger.info('Estimating velocity from cumulative displacement...')

    # initialise
    result_cube = np.zeros((G.shape[1], cum.shape[1], cum.shape[2]), dtype=np.float32) * np.nan
    stderr_cube = np.zeros((G.shape[1], cum.shape[1], cum.shape[2]), dtype=np.float32) * np.nan
    resid_cube = np.zeros(cum.shape, dtype=np.float32) * np.nan

    # identify pixels with data to solve
    has_data = np.any(~np.isnan(cum), axis=0)
    data = cum[()].reshape(n_im, cum[0].size)[:, has_data.ravel()]  # [()] to expose array under HDF5 dataset "cum", use ravel() because fancy indexing is only allowed on 1D arrays
    result = np.zeros((G.shape[1], data.shape[1]), dtype=np.float32) * np.nan
    stderr = np.zeros((G.shape[1], data.shape[1]), dtype=np.float32) * np.nan
    resid = np.zeros(data.shape, dtype=np.float32) * np.nan

    # solve pixels with full data and partial data separately
    full = np.all(~np.isnan(data), axis=0)
    n_pt_full = full.sum()
    if n_pt_full != 0:
        logger.info('  Solving {}/{} points with full data together...'.format(n_pt_full, data.shape[1]))
        d = data[:, full]
        result[:, full], stderr[:, full], resid[:, full] = wls_batch(d, G, sig)

    d = data[:, ~full]
    if sum(~full) > 500:
        logger.info('  Solve {} points with nans point-by-point in parallel...'.format(sum(~full)))
        result[:, ~full], stderr[:, ~full], resid[:, ~full] = parallel_wls_pixel_wise(d, G, sig)
    else:
        logger.info('  Solve {} points with nans point-by-point...'.format(sum(~full)))
        result[:, ~full], stderr[:, ~full], resid[:, ~full] = wls_pixel_wise(d, G, sig)

    # place model and errors into cube
    result_cube[:, has_data] = result
    stderr_cube[:, has_data] = stderr
    resid_cube[:, has_data] = resid

    return result_cube, stderr_cube, resid_cube


def make_G(dt_cum):
    if args.season:
        sin = np.sin(2 * np.pi * dt_cum)
        cos = np.cos(2 * np.pi * dt_cum)
        G = np.vstack([np.ones_like(dt_cum), dt_cum, sin, cos]).transpose()
    else:
        G = np.vstack([np.ones_like(dt_cum), dt_cum]).transpose()
    return G


def count_nans(cum):
    # identify pixels with data to solve
    nan_epochs = np.sum(np.isnan(cum), axis=0).astype(float)
    nan_epochs[nan_epochs == nan_epochs.max()] = np.nan

    # identify locations of pixels with data but also with nans in the time series
    plt.imshow(nan_epochs, interpolation='nearest', cmap=cm.viridis.reversed())
    plt.colorbar()
    plt.title("Number of nan epochs")
    plt.savefig('{}_data_completeness.png'.format(args.cumfile))
    plt.close()


if __name__ == "__main__":
    start()
    init_args()

    # read input cum.h5
    cumh5 = h5.File(args.cumfile, 'r')
    imdates = cumh5['imdates'][()].astype(str).tolist()
    cum = cumh5['cum']
    if args.plot_cum:
        plot_cum_grid(cum[:, ::args.downsample, ::args.downsample], imdates, args.cumfile, args.cumfile + ".png")

    if args.count_nans:
        count_nans(cum)

    ### Calc dt in year
    imdates_dt = ([dt.datetime.strptime(imd, '%Y%m%d').toordinal() for imd in imdates])
    dt_cum = np.float32((np.array(imdates_dt) - imdates_dt[0]) / 365.25)
    epochs = ([dt.datetime.strptime(imd, '%Y%m%d') for imd in imdates])
    n_im = len(dt_cum)

    if args.ref:
        cum_ref = np.ones(cum.shape) * np.nan
        for i in np.arange(n_im):
            cum_ref[i, :, :] = cum[i, :, :] - cum[i, cum.shape[1] // 2, cum.shape[2] // 2]
        if args.plot_cum:
            plot_cum_grid(cum_ref[:, ::args.downsample, ::args.downsample], imdates, args.cumfile + "_ref2center", args.cumfile + "_ref2center.png")
        cum = cum_ref

    if args.ramp:
        logger.info("Estimating a planar ramp per epoch...")
        # downsample
        small_cum = cum[:, ::args.downsample, ::args.downsample]

        # initialise
        ramp_cum = np.zeros(small_cum.shape) * np.nan
        range_coefs = np.zeros(n_im) * np.nan
        azi_coefs = np.zeros(n_im) * np.nan

        # calc a best fit ramp for each epoch and register range and azimuth ramp parameters
        for i in np.arange(n_im):
            ramp_cum[i, :, :], range_coefs[i], azi_coefs[i] = fit_plane(small_cum[i, :, :], np.deg2rad(args.heading))

        # calculate std of flattened cum displacement to use as weights
        flat_cum = small_cum - ramp_cum
        flat_std = np.array([np.nanstd(flat_cum[i, :, :]) for i in np.arange(n_im)])
        flat_std[0] = flat_std[1]  # to avoid 0 weight for the first epoch

        # remove a linear trend from flat_std because std increases over time due to real signal
        G = np.vstack([np.ones_like(dt_cum), dt_cum]).transpose()
        olsfit = sm.OLS(flat_std.transpose(), G).fit()
        detrended_flat_std = flat_std - olsfit.fittedvalues + olsfit.fittedvalues[0]
        if np.nanmin(detrended_flat_std) < 0:  # to make sure all weights are safely above 0
            detrended_flat_std = detrended_flat_std - np.nanmin(detrended_flat_std) + np.nanstd(detrended_flat_std)

        weights = 1 / detrended_flat_std ** 2

        # model and plot ramp coefs time series
        G = make_G(dt_cum)
        d = np.vstack([range_coefs, azi_coefs]).transpose()
        wlsfit = sm.WLS(d, G, weights=weights).fit()

        vector_ramp_coef_resid = np.sqrt(np.sum(np.square(wlsfit.resid), axis=1))
        vector_ramp_coef_resid_scaled = np.std(flat_std) / np.std(vector_ramp_coef_resid) * vector_ramp_coef_resid
        sig = np.sqrt(detrended_flat_std**2 + vector_ramp_coef_resid_scaled**2)
        plot_ramp_coef_time_series(epochs, range_coefs, azi_coefs, flat_std, detrended_flat_std, weights, vector_ramp_coef_resid_scaled, sig, wlsfit)

        # plot 3D time series
        if args.plot_cum:
            ramp_cum[np.isnan(small_cum)] = np.nan
            plot_cum_grid(ramp_cum, imdates, "Best-fit ramps {}".format(args.cumfile), args.cumfile + "_ramps.png")
            plot_cum_grid(flat_cum, imdates, "Flattened {}".format(args.cumfile), args.cumfile + "_flattened.png")

    if args.linear:
        ### Weighted Least Squares Inversion
        G = make_G(dt_cum)
        if args.ramp:  # use residuals of ramp coefs seasonal models and deramped std for weighting the inversion
            result_cube, stderr_cube, resid_cube = calc_vel_and_err(cum, G, sig)
        else:  # unweighted inversion
            result_cube, stderr_cube, resid_cube = calc_vel_and_err(cum, G, np.ones_like(dt_cum))

        # save linear velocity
        vel = result_cube[1]
        vstd = stderr_cube[1]
        vel.tofile('{}_vel'.format(args.cumfile))
        vstd.tofile('{}_vstd'.format(args.cumfile))
        if args.plot_vel:
            import LiCSBAS_plot_lib as plot_lib
            plot_lib.make_im_png(vel, '{}_vel.png'.format(args.cumfile), SCM.roma.reversed(), 'vel {}'.format(args.cumfile))
            plot_lib.make_im_png(vstd, '{}_vstd.png'.format(args.cumfile), 'viridis', 'vstd {}'.format(args.cumfile))

        if args.season:
            coef_s = result_cube[2]
            coef_c = result_cube[3]
            coef_s_sigma = stderr_cube[2]
            coef_c_sigma = stderr_cube[3]

            doy0 = (dt.datetime.strptime(imdates[0], '%Y%m%d') - dt.datetime.strptime(imdates[0][0:4] + '0101','%Y%m%d')).days
            amp = np.sqrt(coef_s ** 2 + coef_c ** 2)
            delta_t = np.arctan2(-coef_c, coef_s) / 2 / np.pi * 365.25  ## wrt 1st img
            delta_t = delta_t + doy0  ## wrt Jan 1
            delta_t[delta_t < 0] = delta_t[delta_t < 0] + 365.25  # 0-365.25
            delta_t[delta_t > 365.25] = delta_t[delta_t > 365.25] - 365.25

            amp.tofile('{}_amp'.format(args.cumfile))
            delta_t.tofile('{}_delta_t'.format(args.cumfile))
            amp_max = np.nanpercentile(amp, 99)

            if args.plot_vel:
                plot_lib.make_im_png(amp, '{}_amp.png'.format(args.cumfile), 'viridis', 'amp {}'.format(args.cumfile), vmin=0, vmax=amp_max)
                plot_lib.make_im_png(delta_t, '{}_delta_t.png'.format(args.cumfile), SCM.romaO.reversed(), 'delta_t {}'.format(args.cumfile))

            if args.de_season:
                # add linear and residual component as an easier way to remove the seasonal component
                linear_cube = np.dot(G[:, 1], vel)
                de_seasoned_cube = linear_cube + resid_cube
                print('\nWriting to HDF5 file...')
                de_seasoned_h5 = h5.File(args.cumfile[:-2]+'de_seasoned.h5', 'w')
                de_seasoned_h5.create_dataset('imdates', data=[np.int32(imd) for imd in imdates])
                de_seasoned_h5.create_dataset('amp', data=amp, compression='gzip')
                de_seasoned_h5.create_dataset('delta_t', data=delta_t, compression='gzip')
                de_seasoned_h5.create_dataset('de_seasoned_cube', data=de_seasoned_cube, compression='gzip')
                de_seasoned_h5.close()

        if args.plot_cum:
            plot_cum_grid(resid_cube[:, ::args.downsample, ::args.downsample], imdates, "Resid {} (linear={}, season={})".format(args.cumfile, str(args.linear), str(args.season)), args.cumfile + "resid.png")

    cumh5.close()
    finish()


    # if args.de_season:
    #     n_im, length, width = cum.shape
    #     # read dt and amp from inputs and downsample
    #     delta_t = np.fromfile(args.delta_t, dtype=np.float32).reshape(length, width)
    #     amp = np.fromfile(args.amp, dtype=np.float32).reshape(length, width)
    #     delta_t = delta_t[::args.downsample, ::args.downsample]
    #     amp = amp[::args.downsample, ::args.downsample]
    #
    #     # remove seasonal_cum from cum to get remaining cum
    #     print("Removing seasonal component...")
    #     seasonal_cum = np.zeros(cum.shape) * np.nan
    #     remain_cum = np.zeros(cum.shape) * np.nan
    #     print("New cubes created...")
    #     for x in np.arange(cum.shape[2]):
    #         if x % (cum.shape[2] // 10) == 0:
    #             print("Processing {}0%".format(x // (cum.shape[2] // 10)))
    #         for y in np.arange(cum.shape[1]):
    #             seasonal_cum[:, y, x] = amp[y, x] * np.cos(2 * np.pi * (dt_cum - delta_t[y, x] / 365.26))
    #             remain_cum[:, y, x] = cum[:, y, x] - seasonal_cum[:, y, x]
    #     # plot cumulative displacement grids
    #     if args.plot_cum:
    #         plot_cum_grid(seasonal_cum, imdates, "Seasonal {}".format(args.cumfile), args.cumfile + ".seasonal.png")
    #         plot_cum_grid(remain_cum, imdates, "De-seasoned {}".format(args.cumfile), args.cumfile + ".de-seasoned.png")

