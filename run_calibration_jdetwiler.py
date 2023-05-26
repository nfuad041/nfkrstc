#!/usr/bin/env python3
import sys, h5py, json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pygama import lh5
import pygama.utils as pgu
import pygama.analysis.histograms as pgh
import pygama.analysis.calibration as pgc
import pygama.analysis.peak_fitting as pgf
if len(sys.argv) < 2:
    print("Usage: run33_Ecal [lh5 dsp file(s)]")
    sys.exit()
# set up input streams
filenames = sys.argv[1:]
store = lh5.Store()
# load detector IDs
with open('/global/project/projectdirs/legend/data/legend-metadata/hardware/channelmaps/channel-map-run0029.json') as f:
    cmap = json.load(f)
detids = {}
for key, data in cmap.items(): detids[f"g0{data['adc']}"] = data['detid']
# get detector list. assumes all files have same channel map!
with h5py.File(filenames[0], 'r') as f:
    det_list = list(f.keys())
det_list = list(filter(lambda x: x.startswith('g'), det_list))
# prepare storage for calibration constants
cal_consts = {}
# loop over detectors
for det in det_list:
    if det in ['g029', 'g030', 'g031', 'g034', 'g035']: continue
    # pull out E array
    #energy_name = det+'/dsp/trapEftp'
    energy_name = det+'/dsp/trapE'
    e_lh5, n_E = store.read_object(energy_name, filenames)
    EE = e_lh5.nda
    if n_E != len(EE):
        print(f'got n_E ({n_E}) != len(EE) ({len(EE)})')
        continue
    # eliminate crazy values
    trap_ul = 1e5
    EE = EE[np.where(EE < trap_ul)]
    print('')
    print(f'calibrating {det} ({len(EE)} events)...')
    # rough hist with automatic binning
    hist, bins, var = pgh.get_hist(EE, bins=3000)
    #if True:
    if False:
        pgh.plot_hist(hist, bins, var)
        plt.yscale('log')
        plt.ylim(0.1, 1.1*np.amax(hist))
        plt.show()
    # check extrema finding
    #if True:
    if False:
        imaxes, imins = pgc.get_i_local_extrema(np.sqrt(hist), 5)
        pgh.plot_hist(hist, bins, var)
        bin_ctrs = pgh.get_bin_centers(bins)
        plt.plot(bin_ctrs[imaxes], hist[imaxes], lw=0, marker='v')
        plt.plot(bin_ctrs[imins], hist[imins], lw=0, marker='^')
        plt.yscale('log')
        plt.ylim(0.1, 1.1*np.amax(hist))
        plt.show()
    # rough cal: match big peaks
    big_peaks_keV = [ 583.187, 727.330, 785.37, 860.557, 1620.50, 2614.511 ]
    
    found_peaks, pars = pgc.hpge_find_E_peaks(hist, bins, var, big_peaks_keV, deg=0, verbosity=1)
    E2uc = 1/pars[0] # rough conversion from E in keV to uncalibrated energy
    print('found peaks:')
    for E_peak in found_peaks: print(f'{E_peak/E2uc} --> {E_peak}')
    #if True:
    if False:
        pgh.plot_hist(hist, bins, var)
        mxi = [ hist[pgh.find_bin(Euci, bins)] for Euci in found_peaks ]
        plt.plot(found_peaks, mxi, lw=0, marker='v')
        plt.yscale('log')
        plt.ylim(0.1, 1.1*np.amax(hist))
        plt.show()
    # rebin in 0.2 keV bins, but keep in uncalibrated E
    Euc_hi = 3000*E2uc
    hist, bins, var = pgh.get_hist(EE, bins=15000, range=(0,Euc_hi))
    #if True:
    if False:
        pgh.plot_hist(hist, bins, var)
        plt.yscale('log')
        plt.ylim(0.1, 1.1*np.amax(hist))
        plt.show()
    # need to re-match big peaks in new binning
    found_peaks, pars = pgc.hpge_find_E_peaks(hist, bins, var, big_peaks_keV, deg=0, verbosity=1)
    print('found peaks 2:')
    for E_peak in found_peaks: print(f'{E_peak/E2uc} --> {E_peak}')
    #if True:
    if False:
        pgh.plot_hist(hist, bins, var)
        mxi = [ hist[pgh.find_bin(Euci, bins)] for Euci in found_peaks ]
        plt.plot(found_peaks, mxi, lw=0, marker='v')
        plt.yscale('log')
        plt.ylim(0.1, 1.1*np.amax(hist))
        plt.show()
    # refined cal: fit to peak tops twice
    n_to_fit = 9
    pt_pars, pt_covs = pgc.hpge_fit_E_peak_tops(hist, bins, var, found_peaks, n_to_fit=n_to_fit)
    mu1s = pt_pars[:,0]
    print('peak top fits 1:', mu1s)
    n_to_fit = 11
    pt_pars, pt_covs = pgc.hpge_fit_E_peak_tops(hist, bins, var, mu1s, n_to_fit=n_to_fit, inflate_errors=True, gof_method='Pearson')
    mus = pt_pars[:,0]
    mu_vars = pt_covs[:,0,0]
    print('peak top fits 2:', mus, np.sqrt(mu_vars))
    #if True:
    if False:
        pgh.plot_hist(hist, bins, var)
        plt.xlim(3300, 3400)
        gpars = pt_pars[1]
        pgu.plot_func(pgf.gauss_basic, gpars)
        plt.yscale('log')
        plt.ylim(0.1, 1.1*np.amax(hist))
        plt.show()
    # Fit the E scale to a full linear function
    pars, cov = pgc.hpge_fit_E_scale(mus, mu_vars, big_peaks_keV, deg=1)
    print('E scale:', pars, cov.flatten())
    # Invert the E scale to get the calibration function
    pars, cov = pgc.hpge_fit_E_cal_func(mus, mu_vars, big_peaks_keV, pars, deg=1)
    print('cal func:', pars, cov.flatten())
    # store calibration constants
    cal_consts[det] = {}
    cal_consts[det]['trapE'] = {}
    cal_consts[det]['trapE']['calibration'] = {}
    cal_consts[det]['trapE']['calibration']['func'] = 'poly'
    cal_consts[det]['trapE']['calibration']['pars'] = pars.tolist()
    cal_consts[det]['trapE']['calibration']['cov'] = cov.tolist()
    if True:
    #if False:
        plt.style.use('seaborn-talk')
        fig = plt.figure(figsize=(10,8))
        fig.suptitle(f'PGT run 33, {detids[det]} ({det})', fontsize=16)
        grid = plt.GridSpec(4, 1, hspace=0.4)
        spectrum = fig.add_subplot(grid[:2,0])
        residuals = fig.add_subplot(grid[2,0], sharex=spectrum)
        fwhm_fit = fig.add_subplot(grid[3,0], sharex=spectrum)
        # add vertical lines at real peak locations
        for peak_E in big_peaks_keV:
            spectrum.axvline(x=peak_E, color='b', ls='--')
        # plot calibrated energy spectrum
        hist, bins, var = pgh.get_hist(pgf.poly(EE, pars), bins=15000, range=(0,3000))
        spectrum.step(np.concatenate(([bins[0]], bins)), 
                      np.concatenate(([0], hist, [0])), 
                      where="post", color='black')
        spectrum.set_ylabel('counts / 0.2 keV')
        spectrum.set_yscale('log')
        spectrum.set_ylim(0.1, 2*np.amax(hist))
        # add markers where peaks were found
        found_peaks_cal = pgf.poly(found_peaks, pars)
        found_peak_heights = [ hist[pgh.find_bin(Ei, bins)] for Ei in found_peaks_cal ]
        spectrum.plot(found_peaks_cal, found_peak_heights, lw=0, marker='v')
        # draw fits to peak tops
        for i in range(len(pt_pars)):
            Ei = pgf.poly(mu1s[i], pars)
            ww = n_to_fit*0.2/2
            xxi = np.linspace(Ei-ww, Ei+ww, 20)
            mu = pgf.poly(pt_pars[i][0], pars)
            sig = pars[0]*pt_pars[i][1]
            amp = pt_pars[i][2] # should still be 0.2 keV bins --> same height
            yyi = pgf.gauss_basic(xxi, mu, sig, amp)
            spectrum.plot(xxi, yyi, color='red')
        # draw fit residuals
        residuals.axhline(y=0, color='black', lw=1)
        residuals.errorbar(big_peaks_keV, pgf.poly(mus, pars)-big_peaks_keV, 
                           pars[0]*np.sqrt(mu_vars), c='b', lw=0, elinewidth=2, marker='.')
        residuals.set_ylabel('E$_{f}$ - E$_{p}$ [keV]')
        residuals.set_ylim(-0.5, 0.5)
        # get FWHMs
        ww = 10 # will go +/- ww on either side of each peak
        bgwings_keV = 5
        fwhm = np.zeros(len(mus))
        dfwhm = np.zeros(len(mus))
        for i, Ei in enumerate(pgf.poly(mus, pars)):
            # first estimate the background
            i_0 = pgh.find_bin(Ei-ww, bins)
            i_f = pgh.find_bin(Ei-ww+bgwings_keV, bins)
            bg1 = np.sum(hist[i_0:i_f])
            n1 = i_f-i_0
            i_0 = pgh.find_bin(Ei+ww-bgwings_keV, bins)
            i_f = pgh.find_bin(Ei+ww, bins)
            bg2 = np.sum(hist[i_0:i_f])
            n2 = i_f-i_0
            bg = (bg1+bg2)/(n1+n2)
            dbg = np.sqrt(bg1+bg2)/(n1+n2)
            # get the peak max estimate from the earlier fit
            mx = pt_pars[i][2]
            dmx = np.sqrt(pt_covs[i][2,2])
            i_0 = pgh.find_bin(Ei-ww, bins)
            i_f = pgh.find_bin(Ei+ww, bins)
            hh = hist[i_0:i_f]
            bb = bins[i_0:i_f+1]
            vv = var[i_0:i_f]
            meth = 'fit_slopes'
            fwhm[i], dfwhm[i] = pgh.get_fwhm(hh, bb, vv, mx=mx, dmx=dmx, bl=bg, dbl=dbg, method=meth)
        # fit and draw fwhm trend
        fwhm2 = fwhm**2
        dfwhm2 = 2 * fwhm * dfwhm
        f2pars, f2cov = np.polyfit(big_peaks_keV, fwhm2, 2, w=1/dfwhm2, cov=True)
        fwhm_trend = np.sqrt(pgf.poly(bins, f2pars))
        fwhm_fit.plot(bins, fwhm_trend)
        fwhm_fit.errorbar(big_peaks_keV, fwhm, dfwhm, c='b', lw=0, elinewidth=2, marker='.')
        fwhm_fit.set_xlabel('E [keV]')
        fwhm_fit.set_xlim(0, 3000)
        fwhm_fit.set_ylabel('FWHM [keV]')
        fwhm_fit.set_ylim(1, 3)
        #plt.show()
        plt.savefig(det+'cal.png')
        plt.close()
        mpl.rcParams.update(mpl.rcParamsDefault)
print(json.dumps(cal_consts, indent=4))