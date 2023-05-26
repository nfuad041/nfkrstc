import os, h5py, json, copy
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('/global/homes/f/fnafis/utils_nf/nfuad.mpl')

from pygama.flow import DataGroup
from pygama.lgdo.lh5_store import load_nda, LH5Store
import pygama.math.histogram as pgh
from pygama.lgdo import ls, load_dfs
from pygama.dsp import build_dsp
from pygama.vis import WaveformBrowser

import h5py

import pint
ureg = pint.UnitRegistry()

import pygama.lgdo.lh5_store as lh5
import pygama.math.histogram as pgh

from matplotlib import pyplot as plt
plt.style.use('/global/u2/f/fnafis/nfuad.mpl')



krtscDB = DataGroup('/global/cfs/cdirs/legend/software/KrSTC/data/krstc.json', load=True)
runDB = krtscDB.runDB
lh5_dir = krtscDB.lh5_dir
dsp_list = lh5_dir + krtscDB.fileDB['dsp_path'] + '/' + krtscDB.fileDB['dsp_file']

BEGINNING_CYCLE = 2019

def get_dsp_params_list(run):
    start_cycle = int(runDB[str(run)][0].split('-')[0])
    f = h5py.File('/global/cfs/cdirs/m2676/data/krstc/LH5/dsp/krstc_run'+str(run)+'_cyc'+str(start_cycle)+'_dsp.lh5','r')
    return f['ORSIS3302DecoderForEnergy/dsp'].keys()

def load_dsp(run, 
             params=[['trapEmax', 'tp_0', 'stp_20']], 
             verbose=True,
             all_columns=False,
             calibration_consts = [0.433,0] #[0.4309, 0.208],  #[0.431, 0.132]
             skip_cycles = None):
    """
    Load dsp data for a given run
    """
    start_cycle = int(runDB[str(run)][0].split('-')[0])
    if runDB[str(run)][0].__contains__('-'):
        end_cycle = int(runDB[str(run)][0].split('-')[1])
    else:
        end_cycle = start_cycle
    run_type = runDB[str(run)][1]
    run_description = runDB[str(run)][2]
    # hit_list = dsp_list[start_cycle-BEGINNING_CYCLE:end_cycle-BEGINNING_CYCLE+1]
    hit_list = []
    for dsp_file in dsp_list:
        if dsp_file.__contains__('run'+str(run)):
            # strip 4 letters after cyc in each dsp file
            cyc = int(dsp_file.split('cyc')[1].split('_dsp')[0])
            if (skip_cycles is not None):
                if (cyc in skip_cycles):
                    continue
            hit_list.append(dsp_file)
    
    #get data
    dat = pd.DataFrame()
    for hit in hit_list:
        
        cyc = int(hit.split('cyc')[1].split('_dsp')[0])

        if all_columns:
            ff = h5py.File(hit, 'r')
            print(ff['ORSIS3302DecoderForEnergy/dsp'].keys())
            all_columns = False
        
        if verbose:
            print(hit)
        # load the data
        d = lh5.load_dfs(hit, *params, 'ORSIS3302DecoderForEnergy/dsp')
        d['trapEmax_cal_keV'] = d['trapEmax']*calibration_consts[0] + calibration_consts[1]
        if params[0].__contains__('tp_0'):
            d['pulse_rise_time_ns'] = d['stp_20']-d['tp_0']
        d['run'] = run
        d['cycle'] = cyc
        f = krtscDB.fileDB
        d['runtime_s'] = float(f[f.cycle==cyc]['runtime'])*60 #convert to seconds

        

        # add d to dat
        dat = pd.concat([dat,d])
    if verbose:
        print(len(hit_list), 'files found with run '+str(run)+', expected:'+str(end_cycle-start_cycle+1))
        runtime = dat.runtime_s.unique().sum()/60 #minutes
        print(str(run)+'--'+run_type+'--{} hrs'.format(runtime/60)+'--'+run_description)
    # if all_columns:
    #     print(dat.columns)
    return dat, run_type, run_description


def get_raw_wfs(df, 
                nwfs=3,
                random = False,
                display_params=['trapEmax_cal_keV', 'tp_0'], 
                verbose=True, 
                plot=True, 
                show_legend=False, 
                xlim = (36000,42000),
                blsub = True,
                align_at = 'tp_50',
                **kwargs):
    if random:
        df = df.sample(n=nwfs)

    time = np.arange(0, 8192*10, 10) # in ns

    wfs = []


    # iter rows
    for idx, row in df.iterrows():
        filename = '/global/cfs/cdirs/m2676/data/krstc/LH5/raw/krstc_run'+str(row.run.astype(int))+'_cyc'+str(row.cycle.astype(int))+'_raw.lh5'
        with h5py.File(filename, 'r') as f:
            wf = f['ORSIS3302DecoderForEnergy']['raw']['waveform']['values'][idx]
            if blsub:
                bl = np.mean(wf[1000:2000])
                wf = wf - bl
            if align_at is not None:
                wf = np.roll(wf, ((28000 - row[align_at])/10).astype(int))
                xlim = (24000,32000)
            wfs.append(wf)
            
            # shifted_wf = np.roll(wf, ((superpulse_shift_to-row[superpulse_shift_from])/10).astype(int))
            # shifted_wfs.append(shifted_wf)

        
        label_text = ''
        # label with 2 decimal points
        for param in display_params:
            label_text += param + ': ' + f'{row[param]:.2f}' + ', '

        if plot:
            if show_legend:
                plt.plot(time, wf, label=label_text, **kwargs)
                plt.legend()
            else:
                plt.plot(time, wf, **kwargs)
            plt.xlabel('Time (ns)')
            plt.xlim(xlim[0],xlim[1])
            plt.ylabel('ADC')


    if verbose:
        display_params.append('run')
        display_params.append('cycle')
        print(df[display_params])
        print('\n')

    wfs = np.array(wfs)
    return wfs

def normalize_wf(wf, btn=(0,1)):
    max = np.max(wf)
    min = np.min(wf)

    wf = (btn[1] - btn[0]) * (wf - min) / (max - min) + btn[0]
    return wf

    
    # superpulse = np.mean(np.array(shifted_wfs), axis=0)
    # if plot_superpulse:

    #     plt.figure()
        
    #     if superpulse_label is None:
    #         superpulse_label = 'Run: '+str(df.run.unique())
    #     plt.plot(time, superpulse, label='Run '+str(df.iloc[0]['run']))
    #     if plot_shifted_wfs:
    #         for shifted_wf in shifted_wfs:
    #             plt.plot(time, shifted_wf)
    #     plt.legend()
    #     plt.xlim(xlim[0], xlim[1])
    #     plt.title('Superpulses')
    #     plt.xlabel('Time (ns)')

    # return superpulse

def get_superpulse(df_cut,
                   shift_to = 38000,
                   shift_from = 'tp_0',
                   plot_shifted = False,
                   plot_superpulse = True,
                   superpulse_label = 'superpulse',
                   **kwargs):
    """
    df_cut: all the events from df_cut will be retrieved to make the superpulse. So do any cuts you want before passing df_cut to this function. Keep the size of df_cut small to get faster result.
    """
    # sort by index
    df = df_cut.sort_index()

    # unique run+cycle values
    df['run_cycle'] = 'run'+df['run'].astype(str) + '_cyc' + df['cycle'].astype(str)
    run_cycles = df['run_cycle'].unique()
    wfss = []
    shift_fromss = []
    #wfsd = pd.DataFrame()
    for run_cycle in run_cycles:
        #print(run_cycle)
        d = df[df.run_cycle==run_cycle]
        idxs = d.index.values
        #print('idxs', idxs.shape)
        shift_froms = d[shift_from][idxs]
        shift_fromss.append(shift_froms)
        #print('shift_froms', shift_froms.shape)
        
        filename = '/global/cfs/cdirs/m2676/data/krstc/LH5/raw/krstc_'+run_cycle+'_raw.lh5'
        with h5py.File(filename, 'r') as f:
            # get all the wfs in this cycle with indices=idxs
            wfs = f['ORSIS3302DecoderForEnergy']['raw']['waveform']['values'][idxs]
            # append to wfss
            wfss.append(wfs)

    wfss = np.concatenate(wfss)
    wfss = np.array(wfss)
    #print('wfss', wfss.shape)
    shift_fromss = np.concatenate(shift_fromss)
    shift_fromss = np.array(shift_fromss)
    #print('shift_fromss', shift_fromss.shape)

    # find baseline: take mean of [100:2000] for each wf in wfss
    baselines = np.mean(wfss[:,100:2000], axis=1)
    # subtract bls from each wf in wfss
    wfss = wfss - baselines[:,None]

    # shift each wf in wfss

    shifted_wfss = []
    #print('shift from size', shift_fromss.shape)
    for wf, shfrom in zip(wfss, shift_fromss):
        #print('shfrom',shfrom)
        shifted_wfss.append(np.roll(wf, int((shift_to-shfrom)/10)))

    

    # shifted_wfss = np.roll(wfss, ((shift_to-shift_froms)/10).astype(int), axis=1)

    time = np.arange(0, wfss.shape[1]*10, 10)
    # if plot_original:
    #     for wf in wfss:
    #         plt.plot(time, wf)
    if plot_shifted:
        for wf in shifted_wfss:
            plt.plot(time, wf, **kwargs)
        plt.xlabel('time [ns]')
    
    superpulse = np.mean(np.array(shifted_wfss), axis=0)

    #superpulse_label = df_cut['run'].unique()[0]
    if plot_superpulse:
        plt.plot(time, superpulse, label=superpulse_label, **kwargs)
        plt.xlabel('time [ns]')
        plt.legend()

    plt.xlim(shift_to-3000, shift_to+3000)
    return superpulse



def get_1D_hist(series, runtimes, 
                range=None, 
                binsize=None, 
                normalized=True, 
                verbose=False, 
                label=None,
                plot=True,
                **kwargs):
    percentile_20 = series.quantile(0.2)
    percentile_80 = series.quantile(0.8)
    if range is None:
        range = (percentile_20, percentile_80)
        print('range:', range)
    if binsize is None:
        binsize = (percentile_80 - percentile_20)/100
        print('binsize:', binsize)
    hist, bins, vars = pgh.get_hist(series, bins=np.arange(*range, binsize), range=range)
    if normalized:
        hist = hist/runtimes.unique().sum()/binsize
        if verbose:
            print('total runtime:',runtimes.unique().sum()/60, 'mins')
    if label is None:
        label = series.name

    if plot:
        pgh.plot_hist(hist, bins, label=label, **kwargs)
        plt.xlabel(series.name)

    if verbose:
        print(series.name, series.describe())

        
    plt.legend()
    
    return hist, bins, vars

def get_subtracted_1D_hist(subtractor_series, subtractor_runtimes, subtractee_series, subtractee_runtimes,
                               range=None, 
                                binsize=None, 
                                normalized=True, 
                                verbose=False, 
                                label=None,
                                title = None,
                                plot=True,
                                color=None):
    
    hist_subtractee, _, _ = get_1D_hist(subtractee_series, subtractee_runtimes, range=range, binsize=binsize, normalized=normalized, verbose=verbose, plot=False)
    hist, bins, var = get_1D_hist(subtractor_series, subtractor_runtimes, range=range, binsize=binsize, normalized=normalized, verbose=verbose, plot=False)

    subtracted_hist = hist - hist_subtractee

    if plot:
        pgh.plot_hist(subtracted_hist, bins, label=label, color=color)
        plt.title(str(title))

    return subtracted_hist, bins, var

def get_2D_hist(x,y, runtimes, 
                ranges=[None, None], 
                binsizes=[None,None],
                normalized = False, # first normalizes
                vlim=None, # then changes scale, using a tuple like (0, 100)
                log = False, #then take the log
                cbar_label = 'counts/s',
                plot= True,
                cmap = 'jet',
                **kwargs):

    percentile_20 = [x.quantile(0.2), y.quantile(0.2)]
    percentile_80 = [x.quantile(0.8), y.quantile(0.8)]
    if ranges[0] is None:
        ranges[0] = (percentile_20[0], percentile_80[0])
        print('xrange:',ranges[0])
    if ranges[1] is None:
        ranges[1] = (percentile_20[1], percentile_80[1])
        print('yrange:',ranges[1])
    if binsizes[0] is None:
        binsizes[0] = (ranges[0][1] - ranges[0][0])/100
        print('xbinsize:',binsizes[0])
    if binsizes[1] is None:
        binsizes[1] = (ranges[1][1] - ranges[1][0])/100
        print('ybinsize:',binsizes[1])

    bins = [np.arange(*ranges[0], binsizes[0]), np.arange(*ranges[1], binsizes[1])]
    hist, xbins, ybins = np.histogram2d(x, y, bins=bins)
    hist = hist.T

    if normalized:
        hist = hist/runtimes.unique().sum()/(binsizes[0]*binsizes[1])

    
    if vlim is not None:
        hist = (vlim[1]-vlim[0])/(hist.max()-hist.min())*(hist-hist.min()) + vlim[0]
    if log:
        hist = np.log(hist)

    if plot:
        #plt.grid(False)
        plt.pcolormesh(xbins[:-1], ybins[:-1], hist, cmap=cmap, **kwargs)
        
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(cbar_label)

    return hist, xbins, ybins


def get_subtracted_2D_hist(subtractor_x, subtractor_y, subtractor_runtimes, subtractee_x, subtractee_y, subtractee_runtimes, 
                ranges=[None, None], 
                binsizes=[None,None],
                normalized = False, # first normalizes
                vlim=(0.001,1), # then changes scale, using a tuple like (0, 100)
                log = False, #then take the log
                plot = True,
                cbar_label = 'counts/s',
                
                cmap = 'jet',
                **kwargs):
    
    hist_subtractor, xbins, ybins = get_2D_hist(subtractor_x, subtractor_y, subtractor_runtimes, ranges=ranges, binsizes=binsizes, normalized=normalized, vlim=None, log=False, plot=False)

    hist_subtractee, _, _ = get_2D_hist(subtractee_x, subtractee_y, subtractor_runtimes, ranges=ranges, binsizes=binsizes, normalized=normalized, vlim=None, log=False, plot=False)

    hist = hist_subtractor - hist_subtractee

    if vlim is not None:
        hist = (vlim[1]-vlim[0])/(hist.max()-hist.min())*(hist-hist.min()) + vlim[0]
    if log:
        hist = np.log(hist)

    if plot:
        plt.grid(False)
        plt.pcolormesh(xbins[:-1], ybins[:-1], hist, cmap=cmap, **kwargs)
        
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(cbar_label)

    return hist, xbins, ybins


    


def gaussian_func(x,A,mu,sig):
    return A*np.exp(-(x-mu)**2/(2*sig**2))

def gaussian_fit_binned(x, y, plot=False, guess=None, x_range=None):

    if x_range is None:
        argmax = np.argmax(y)
        # print('argmax:',argmax)
        start_bin = (argmax-argmax*0.15).astype(int)
        end_bin = (argmax+argmax*0.15).astype(int)
        #print('start_bin:',start_bin)
        #print('end_bin:',end_bin)
        x = x[start_bin:end_bin]
        y = y[start_bin:end_bin]
    
    if guess==None:
        guess = [np.max(y), np.mean(x), np.std(x)]
        #print('guess:',guess)
    
    try:
        pars, covs = curve_fit(gaussian_func, x, y, p0=guess)
    except:
        pars=[None,None,None]
        covs = [None, None, None]
    
    if plot:
        plt.plot(x, gaussian_func(x, *pars), color='r')

    return {'A':pars[0],
            'mu':pars[1],
            'sig':pars[2],
            'err_A':np.sqrt(covs[0,0]),
            'err_mu':np.sqrt(covs[1,1]),
            'err_sig':np.sqrt(covs[2,2])}

def quadratic(x, p0, p1, p2, p3):
    return p0 + p1*x + p2*x**2
def quadratic_fit_binned(x, y, plot=False, guess=None, x_range=None):
    fit_params = curve_fit(quadratic, x, y)
    x = np.linspace(x.min(), x.max(), (x.max()-x.min())/100)
    y = quadratic(x, *fit_params[0])
    plt.plot(x, y, label='quadratic fit')
    return fit_params
