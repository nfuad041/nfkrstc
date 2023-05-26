#!/usr/bin/env python3
import os
import json
import h5py
import argparse
import pandas as pd
import numpy as np
from pprint import pprint
import tinydb as db
from tinydb.storages import MemoryStorage
import pdb

import matplotlib
# matplotlib.use('Agg') # when running on cori
import matplotlib.pyplot as plt
#plt.style.use('../clint.mpl')
from matplotlib.colors import LogNorm

from pygama.flow import DataGroup
import pygama.lgdo.lh5_store as lh5
import pygama.math.histogram as pgh
import pygama.math.peak_fitting as pgf

def main():
    doc="""
    analysis of Aug 2020 OPPI+CAGE commissioning runs (138-141)
    tasks:
    - load calibration from energy_cal
    - show 1460 peak stability
    - show removal of low-e retrigger noise
    - look at waveforms near 5 MeV, confirm they're muon-related
    - look at low-e waveforms, examine noise
    - determine pz correction value
    """
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'
    arg('-q', '--query', nargs=1, type=str,
        help="select file group to calibrate: -q 'run==1' ")
    args = par.parse_args()

    # load main DataGroup, select files from cmd line
    #dg = DataGroup('$CAGE_SW/processing/cage.json', load=True) #change this cage.json to krstc.json?
    dg = DataGroup('/global/homes/f/fnafis/krstc.json', load=True)
    
    if args.query:
        que = args.query[0]
        dg.fileDB.query(que, inplace=True)
    else:
        dg.fileDB = dg.fileDB[-1:]
    
    #dg.fileDB = dg.fileDB[-1:]
    view_cols = ['runtype', 'run', 'cycle', 'startTime', 'runtime', 'threshold']
    print(dg.fileDB[view_cols])

    # -- run routines --
    #show_raw_spectrum(dg)
    #show_cal_spectrum(dg)
    #show_wfs(dg)
    #data_cleaning(dg)
    #peak_drift(dg)
    #pole_zero(dg)
    #label_alpha_runs(dg)
    #power_spectrum(dg)
    
    
    
def show_raw_spectrum(dg): #plotting trapEmax which is peak of wave form after trapezoid filter is applied
    """
    show spectrum w/ onbd energy and trapE
    - get calibration constants for onbd energy and 'trapE' energy
    - TODO: fit each expected peak and get resolution vs energy
    """
    # get file list and load energy data (numpy array)
    # lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    lh5_dir = dg.lh5_dir
    dsp_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
    pdb.set_trace()
    
    
    edata = lh5.load_nda(dsp_list, ['trapEmax'], 'ORSIS3302DecoderForEnergy/dsp')
    rt_min = dg.fileDB['runtime'].sum()
    u_start = dg.fileDB.iloc[0]['startTime']
    t_start = pd.to_datetime(u_start, unit='s') # str

    print('Found energy data:', [(et, len(ev)) for et, ev in edata.items()])
    print(f'Runtime (min): {rt_min:.2f}')

    elo, ehi, epb, etype = 0, 8000, 10, 'trapEmax' #Histogram options

    ene_uncal = edata[etype] #energy_uncalibrated
    hist, bins, _ = pgh.get_hist(ene_uncal, range=(elo, ehi), dx=epb)

    # normalize by runtime
    hist_rt = np.divide(hist, rt_min * 60)

    plt.plot(np.nan, np.nan, '-w', lw=1, label=t_start)

    plt.semilogy(bins[1:], hist_rt, ds='steps', c='b', lw=1,
                 label=f'{etype}, {rt_min:.2f} mins')

    plt.xlabel(etype, ha='right', x=1)
    plt.ylabel('cts / sec', ha='right', y=1)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('raw_spectrum.png', dpi = 200)
    


def show_cal_spectrum(dg):
    """
    apply calibration to dsp file
    """
    # get file list and load energy data (numpy array)
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    dsp_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
    edata = lh5.load_nda(dsp_list, ['trapEmax'], 'ORSIS3302DecoderForEnergy/dsp')
    rt_min = dg.fileDB['runtime'].sum()
    u_start = dg.fileDB.iloc[0]['startTime']
    t_start = pd.to_datetime(u_start, unit='s') # str
    print('Found energy data:', [(et, len(ev)) for et, ev in edata.items()])
    print(f'Runtime (min): {rt_min:.2f}')

    # load calibration from peakfit
    cal_db = db.TinyDB(storage=MemoryStorage)
    with open('ecalDB.json') as f:
        raw_db = json.load(f)
        cal_db.storage.write(raw_db)
    runs = dg.fileDB.run.unique()
    if len(runs) > 1:
        print("sorry, I can't do combined runs yet")
        exit()
    run = runs[0]
    tb = cal_db.table("peakfit_trapEmax").all()
    df_cal = pd.DataFrame(tb)
    df_cal['run'] = df_cal['run'].astype(int)
    #pdb.set_trace()
    df_run = df_cal.loc[df_cal.run==run]
    cal_pars = df_cal.iloc[0][['cal0','cal1','cal2']]  #it was df_run.iloc... for some reason. that gave me index error.

    # compute calibrated energy
    pol = np.poly1d(cal_pars) # handy numpy polynomial object
    cal_data = pol(edata['trapEmax'])

    elo, ehi, epb, etype = 0, 4000, 1, 'trapEmax_cal' # gamma region
    #elo, ehi, epb, etype = 2500, 8000, 10, 'trapEmax_cal' # overflow region
    # elo, ehi, epb, etype = 0, 250, 1, 'trapEmax_cal' # low-e region

    hist, bins, _ = pgh.get_hist(cal_data, range=(elo, ehi), dx=epb)

    # normalize by runtime
    hist_rt = np.divide(hist, rt_min * 60)

    plt.plot(np.nan, np.nan, '-w', lw=1, label=f'start: {t_start}')

    plt.plot(bins[1:], hist_rt, ds='steps', c='b', lw=1,
                 label=f'{etype}, {rt_min:.2f} mins')

    plt.xlabel(etype, ha='right', x=1)
    plt.ylabel('cts / sec', ha='right', y=1)
    plt.legend(loc=1, fontsize=12)
    plt.tight_layout()
    plt.savefig('cal_spectrum.png')
    # plt.show()
    
    
def show_wfs(dg):
    """
    show waveforms in different energy regions.
    use the hit file to select events
    """
    # get file list and load hit data
    lh5_dir = dg.lh5_dir #if:user, then:dg.lh5_user_dir; else:dg.lh5_dir
    dsp_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
    dsp_list = dsp_list[0]
    #hit_list = lh5_dir + dg.fileDB['hit_path'] + '/' + dg.fileDB['hit_file']
    hit_list = dsp_list #since regular hist list not available now
    df_hit = lh5.load_dfs(hit_list, ['trapEmax', 'bl','AoE', 'dcr', 'tp_max'], 'ORSIS3302DecoderForEnergy/dsp')
    
    #df_hit = lh5.load_dfs(hit_list, ['trapEmax', 'trapEmax_cal', 'bl','AoE', 'dcr_raw', 'tp_0', 'tp_50'], 'ORSIS3302DecoderForEnergy/hit')
    # print(df_hit)
    # print(df_hit.columns)

    # settings
    etype = 'trapEmax'
    #etype = 'trapEmax_cal'
    nwfs = 20

    #creat new DCR
    const = 0.0555
    #df_hit['dcr_linoff'] = df_hit['dcr_raw'] + const*df_hit['trapEmax']
    df_hit['dcr_linoff'] = df_hit['dcr']
    #create 0-50
    #df_hit['tp0_50'] = df_hit['tp_50'] - df_hit['tp_0']
    df_hit['tp0_50'] = df_hit['tp_max']


    # elo, ehi, epb = 0, 100, 0.2 # low-e region
    # elo, ehi, epb = 0, 20, 0.2 # noise region
    elo, ehi, epb = 351, 355, 1 # 351 peak, cal
    # elo, ehi, epb = 1452, 1468, 1 # good physics events
#     elo, ehi, epb = 7100, 7200, 1 # good physics events, uncal
    # elo, ehi, epb = 6175, 6250, 1 # overflow peak
    # elo, ehi, epb = 5000, 5200, 0.2 # lower overflow peak

    # # diagnostic plot
    # hE, xE, vE = pgh.get_hist(df_hit[etype], range=(elo, ehi), dx=epb)
    # plt.plot(xE[1:], hE, c='b', ds='steps')
    # plt.show()
    # exit()

    # select bulk waveforms
    idx = df_hit[etype].loc[(df_hit[etype] >= elo) &
                            (df_hit[etype] <= ehi)].index[:nwfs]
    raw_store = lh5.Store()
    print(len(idx))
    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    lh5_dir = dg.lh5_dir
    raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
    f_raw = raw_list.values[0] # fixme, only works for one file rn
    data_raw, nrows = raw_store.read_object(tb_name, f_raw, start_row=0, n_rows=idx[-1]+1)

    bulk_wfs_all = (data_raw['waveform']['values']).nda
    bulk_wfs = bulk_wfs_all[idx.values, :]
    del bulk_wfs_all
    ts = np.arange(0, bulk_wfs.shape[1]-1, 1)
    #ts = np.arange(0, bulk_wfs.shape[1], 1)
    
    #pdb.set_trace()
	
    """
    # select alpha waveforms
    dlo = 25
    dhi = 200
    tlo = 100
    thi = 400
    blmin = 8500
    blmax = 10000
    alpha_idx = df_hit[etype].loc[(df_hit['dcr_linoff'] > dlo) & (df_hit['dcr_linoff'] < dhi)
                            & (df_hit['tp0_50'] > tlo) & (df_hit['tp0_50'] < thi) & (df_hit['bl'] > blmin) & (df_hit['bl'] < blmax)
                            & (df_hit[etype] < 12000)].index[:nwfs]


    raw_store = lh5.Store()
    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
    f_raw = raw_list.values[0] # fixme, only works for one file rn
    
    pdb.set_trace()
    data_raw, nrows = raw_store.read_object(tb_name, f_raw, start_row=0, n_rows=alpha_idx[-1]+1)

    alpha_wfs_all = data_raw['waveform']['values'].nda
    alpha_wfs = alpha_wfs_all[alpha_idx.values, :]
    ats = np.arange(0, alpha_wfs.shape[1]-1, 1)
    """
    
    # plot wfs
    for iwf in range(bulk_wfs.shape[0]):
        plt.plot(ts, bulk_wfs[iwf,:len(bulk_wfs[iwf])-1], lw=1, color = 'blue', label = 'Bulk')
        #plt.plot(ts, bulk_wfs[iwf], lw=1, color = 'blue', label = 'Bulk')
        

    plt.xlabel('time (clock ticks)', ha='right', x=1)
    plt.ylabel('ADC', ha='right', y=1)
    
    
    
   

#     # plot alpha wfs
#     for aiwf in range(alpha_wfs.shape[0]):
#         plt.plot(ats, alpha_wfs[aiwf,:len(alpha_wfs[aiwf])-1], lw=1, color = 'red', label = 'Alpha')

    #plt.title('Alpha versus bulk events')
    plt.title('right 351 Wfs run 82')
    plt.xlabel('time (clock ticks)', ha='right', x=1)
    plt.ylabel('ADC', ha='right', y=1)
    #plt.xlim(3500, 4500)
    #plt.ylim(9100, 10300)
    # plt.legend(loc='upper left')
    plt.show()
    plt.savefig('zoom_350_right_waveforms_run82.png', dpi=300)
    # plt.cla()


def data_cleaning(dg):
    """
    using parameters in the hit file, plot 1d and 2d spectra to find cut values.
    columns in file:
        ['trapE', 'bl', 'bl_sig', 'A_10', 'AoE', 'packet_id', 'ievt', 'energy',
        'energy_first', 'timestamp', 'crate', 'card', 'channel', 'energy_cal',
        'trapE_cal']
    note, 'energy_first' from first value of energy gate.
    """
    i_plot = 0 # run all plots after this number

    # get file list and load hit data
    lh5_dir = dg.lh5_dir #if user else dg.lh5_dir
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    
    dsp_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
    #hit_list = lh5_dir + dg.fileDB['hit_path'] + '/' + dg.fileDB['hit_file']
    hit_list = dsp_list
    #df_hit = lh5.load_dfs(hit_list, ['trapEmax'], 'ORSIS3302DecoderForEnergy/hit')
    df_hit = lh5.load_dfs(hit_list, ['trapEmax', 'bl', 'bl_sig'], 'ORSIS3302DecoderForEnergy/dsp')
    # print(df_hit)
    print(df_hit.columns)

    
    # get info about df -- 'describe' is very convenient
    #dsc = df_hit[['bl','bl_sig','A_10','ts_sec']].describe()
    # print(dsc)
    # exit()

    if i_plot <= 0:
        # bl vs energy

        elo, ehi, epb = 0, 50, 1
        blo, bhi, bpb = 0, 10000, 100
        nbx = int((ehi-elo)/epb)
        nby = int((bhi-blo)/bpb)

        #h = plt.hist2d(df_hit['trapEmax_cal'], df_hit['bl'], bins=[nbx,nby], range=[[elo, ehi], [blo, bhi]], cmap='jet')
        
        h = plt.hist2d(df_hit['trapEmax'], df_hit['bl'], bins=[nbx,nby], range=[[elo, ehi], [blo, bhi]], cmap='jet')
        
          

        cb = plt.colorbar(h[3], ax=plt.gca())
        plt.xlabel('trapEmax', ha='right', x=1)
        plt.ylabel('bl', ha='right', y=1)
        plt.tight_layout()
        # plt.show()
        #plt.savefig('oppi_bl_vs_e.png', dpi=300)
        cb.remove()
        plt.cla()
        pdb.set_trace()
        

        # make a formal baseline cut from 1d histogram
        hE, bins, vE = pgh.get_hist(df_hit['bl'], range=(blo, bhi), dx=bpb)
        xE = bins[1:]
        plt.semilogy(xE, hE, c='b', ds='steps')

        bl_cut_lo, bl_cut_hi = 8000, 9500
        plt.axvline(bl_cut_lo, c='r', lw=1)
        plt.axvline(bl_cut_hi, c='r', lw=1)

        plt.xlabel('bl', ha='right', x=1)
        plt.ylabel('counts', ha='right', y=1)
        # plt.show()
        plt.savefig('./plots/oppi_bl_cut.png')
        plt.cla()


    if i_plot <= 1:
        # A_10/trapEmax_cal vs trapEmax_cal (A/E vs E)

        # use baseline cut
        df_cut = df_hit.query('bl > 8000 and bl < 9500').copy()

        # add new A/E column
        df_cut['aoe'] = df_cut['A_10'] / df_cut['trapEmax_cal']

        # alo, ahi, apb = -1300, 350, 1
        # elo, ehi, epb = 0, 250, 1
        alo, ahi, apb = 0, 0.4, 0.005
        # elo, ehi, epb = 0, 3000, 10
        elo, ehi, epb = 0, 6000, 10

        nbx = int((ehi-elo)/epb)
        nby = int((ahi-alo)/apb)

        h = plt.hist2d(df_cut['trapEmax_cal'], df_cut['aoe'], bins=[nbx,nby],
                       range=[[elo, ehi], [alo, ahi]], cmap='jet', norm=LogNorm())

        plt.xlabel('trapEmax_cal', ha='right', x=1)
        plt.ylabel('A/E', ha='right', y=1)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./plots/oppi_aoe_vs_e.png', dpi=300)
        plt.cla()


    if i_plot <= 2:
        # show effect of baseline cut on low-energy spectrum

        df_cut = df_hit.query('bl > 8000 and bl < 9500')

        etype = 'trapEmax_cal'
        elo, ehi, epb = 0, 250, 0.5

        # no cuts
        h1, x1, v1 = pgh.get_hist(df_hit[etype], range=(elo, ehi), dx=epb)
        x1 = x1[1:]
        plt.plot(x1, h1, c='k', lw=1, ds='steps', label='raw')

        # baseline cut
        h2, x2, v2 = pgh.get_hist(df_cut[etype], range=(elo, ehi), dx=epb)
        plt.plot(x1, h2, c='b', lw=1, ds='steps', label='bl cut')

        plt.xlabel(etype, ha='right', x=1)
        plt.ylabel('counts', ha='right', y=1)
        plt.legend()
        # plt.show()
        plt.savefig('./plots/oppi_lowe_cut.png')
        plt.cla()

    if i_plot <= 3:
        # show DCR vs E
        etype = 'trapEmax_cal'
        elo, ehi, epb = 0, 6000, 10
        dlo, dhi, dpb = -1000, 1000, 10

        nbx = int((ehi-elo)/epb)
        nby = int((dhi-dlo)/dpb)

        h = plt.hist2d(df_cut['trapEmax_cal'], df_cut['dcr'], bins=[nbx,nby],
                       range=[[elo, ehi], [dlo, dhi]], cmap='jet', norm=LogNorm())

        plt.xlabel('trapEmax_cal', ha='right', x=1)
        plt.ylabel('DCR', ha='right', y=1)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./plots/oppi_dcr_vs_e.png', dpi=300)
        plt.cla()


def peak_drift(dg):
    """
    show any drift of the 1460 peak (5 minute bins)
    """
    cols = ['trapEmax', 'ts_glo']


    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    
    dsp_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
    #hit_list = lh5_dir + dg.fileDB['hit_path'] + '/' + dg.fileDB['hit_file']
    
    hit_list = dsp_list
    #pdb.set_trace()

    #df_hit = lh5.load_dfs(hit_list, cols, 'ORSIS3302DecoderForEnergy/hit')
    df_hit = lh5.load_dfs(hit_list, cols, 'ORSIS3302DecoderForEnergy/dsp')
    
    
    df_hit.reset_index(inplace=True)
    rt_min = dg.fileDB['runtime'].sum()
    print(f'runtime: {rt_min:.2f} min')
    
    
    
    

    # settings
    # use uncalibrated energy
    elo, ehi, epb, etype = 3400, 3800, 1, 'trapEmax'
    df_hit = df_hit.query(f'trapEmax > {elo} and trapEmax < {ehi}').copy()

    # use calibrated energy (hit file)
    # elo, ehi, epb, etype = 1450, 1470, 1, 'trapEmax_cal'
    # df_hit = df_hit.query(f'trapEmax_cal > {elo} and trapEmax_cal < {ehi}').copy()

    # # diagnostic plot
    hE, xE, vE = pgh.get_hist(df_hit[etype], range=(elo, ehi), dx=epb)
    plt.plot(xE[1:], hE, c='b', ds='steps', lw=1)
    # plt.show()
    plt.savefig('./plots/oppi_1460_hist.pdf')
    plt.cla()

    t0 = df_hit['ts_glo'].values[0]
    df_hit['ts_adj'] = (df_hit['ts_glo'] - t0) / 60 # minutes after 0

    tlo, thi, tpb = 0, df_hit['ts_adj'].max(), 1

    nbx = int((thi-tlo)/tpb)
    nby = int((ehi-elo)/epb)

    h = plt.hist2d(df_hit['ts_adj'], df_hit['trapEmax'], bins=[nbx,nby],
                   range=[[tlo, thi], [elo, ehi]], cmap='jet')

    plt.xlabel(f'Time ({tpb:.1f} min/bin)', ha='right', x=1)
    plt.ylabel('trapEmax', ha='right', y=1)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./plots/oppi_1460_drift.png', dpi=300)


def pole_zero(dg):
    """
    NOTE: I think this result might be wrong, for the CAGE amp it should be
    around 250 usec.  Need to check.
    """
    # load hit data
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
    dsp_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']
    #hit_list = lh5_dir + dg.fileDB['hit_path'] + '/' + dg.fileDB['hit_file']
    hit_list = dsp_list
    #df_hit = lh5.load_dfs(hit_list, ['trapEmax'], 'ORSIS3302DecoderForEnergy/hit')
    df_hit = lh5.load_dfs(hit_list, ['trapEmax'], 'ORSIS3302DecoderForEnergy/dsp')
    df_hit.reset_index(inplace=True)
    rt_min = dg.fileDB['runtime'].sum()
    # print(f'runtime: {rt_min:.2f} min')

    # load waveforms
    #etype = 'trapEmax_cal'
    etype = 'trapEmax'
    nwfs = 20
    elo, ehi = 1455, 1465

    # select waveforms
    idx = df_hit[etype].loc[(df_hit[etype] >= elo) &
                            (df_hit[etype] <= ehi)].index[:nwfs]
    raw_store = lh5.Store()
    tb_name = 'ORSIS3302DecoderForEnergy/raw'
    raw_list = lh5_dir + dg.fileDB['raw_path'] + '/' + dg.fileDB['raw_file']
    f_raw = raw_list.values[0] # fixme, only works for one file rn
    data_raw = raw_store.read_object(tb_name, f_raw, start_row=0, n_rows=idx[-1]+1)
    
    #pdb.set_trace()

    #wfs_all = data_raw['waveform']['values'].nda
    wfs_all = data_raw[0]['waveform']['values'].nda
    wfs = wfs_all[idx.values, :]
    df_wfs = pd.DataFrame(wfs)
    # print(df_wfs)

    # simple test function to compute pole-zero constant for a few wfs.
    # the final one should become a dsp processor
    clock = 1e8 # 100 MHz
    istart = 5000
    iwinlo, iwinhi, iwid = 500, 2500, 20 # two-point slope
    # ts = np.arange(istart, df_wfs.shape[1]-1, 1) / 1e3 # usec
    ts = np.arange(0, df_wfs.shape[1]-1-istart, 1) / 1e3 # usec

    def get_rc(row):
        # two-point method
        wf = row[istart:-1].values
        wflog = np.log(wf)
        win1 = np.mean(np.log(row[istart+iwinlo : istart+iwinlo+iwid]))
        win2 = np.mean(np.log(row[istart+iwinhi : istart+iwinhi+iwid]))
        slope = (win2 - win1) / (ts[iwinhi] - ts[iwinlo])
        tau = 1/slope

        # # diagnostic plot: check against expo method
        # guess_tau = 60
        # a = wf.max()
        # expdec = lambda x : a * np.exp(-x / guess_tau)
        # logdec = lambda x : np.log(a * np.exp(-x / guess_tau))
        # slopeway = lambda x: wflog[0] + x / tau
        # plt.plot(ts, wflog, '-r', lw=1)
        # plt.plot(ts, logdec(ts), '-b', lw=1)
        # plt.plot(ts, slopeway(ts), '-k', lw=1)
        # plt.show()
        # exit()

        return tau

        # return tau

    res = df_wfs.apply(get_rc, axis=1)

    tau_avg, tau_std = res.mean(), res.std()
    print(f'average RC decay constant: {tau_avg:.2f} pm {tau_std:.2f}')
    
    #pdb.set_trace()
    
    #Why not print a new wave form?


def label_alpha_runs(dg):
    """
    example of filtering the fileDB for alpha runs, adding new information
    from a text file, and saving it to a new file, alphaDB.
    """
    # load fileDB
    df_fileDB = pd.read_hdf(os.path.expandvars(dg.config['fileDB']))

    # print(df_fileDB.columns)
    # ['unique_key', 'YYYY', 'mm', 'dd', 'cycle', 'daq_dir', 'daq_file', 'run',
    #  'runtype', 'detector', 'skip', 'raw_file', 'raw_path', 'dsp_file',
    #  'dsp_path', 'hit_file', 'hit_path', 'startTime', 'threshold', 'daq_gb',
    #  'stopTime', 'runtime']

    view_cols = ['unique_key', 'run', 'cycle', 'runtype', 'detector', 'skip']
    # print(df_fileDB[view_cols].to_string())

    # select alpha files only
    df_alphaDB = df_fileDB.query("runtype == 'alp'")
    # print(df_alphaDB[view_cols])

    # load our beam position info -- manually curated list
    df_beam = pd.read_csv(os.path.expandvars('$CAGE_SW/processing/metadata/scan_key.txt'))
    # print(df_beam)

    # add beam position columns to df_alphaDB (our subset)
    g = df_alphaDB.groupby(['run'])
    srun = np.array(df_alphaDB['run'])


    def add_info(df_run, df_beam):
        run = df_run.iloc[0]['run']
        pos_vals = df_beam.loc[df_beam.run == run]
        if len(pos_vals) == 0:
            df_run['radius'] = np.nan
            df_run['source'] = np.nan
            df_run['rotary'] = np.nan
        else:
            df_run['radius'] = pos_vals.iloc[0]['radius']
            df_run['source'] = pos_vals.iloc[0]['source']
            df_run['rotary'] = pos_vals.iloc[0]['rotary']
        return df_run

    df_alphaDB = g.apply(add_info, df_beam)

    view_cols += ['radius','source', 'rotary']
    print(df_alphaDB[view_cols].to_string())

    # two options to proceed here:
    # 1. move this function to setup.py and have it overwrite the fileDB
    # 2. just write df_alphaDB to a separate analysis file here

    df_alphaDB.to_hdf(os.path.expandvars('$CAGE_SW/processing/alphaDB.h5'), key='alphaDB')


def power_spectrum(dg):
    """
    plot power spectral density for groups of runs.
    note.  typical cycle files have ~120,000 wfs.
    """
    import scipy.signal as signal

    view_cols = ['runtype', 'run', 'cycle', 'startTime', 'runtime', 'threshold']

    sto = lh5.Store()
    lh5_dir = os.path.expandvars(dg.config['lh5_dir'])

    # n_wfs = np.inf # np.inf to select all
    n_wfs = int(1e3)
    clk = 100e6 # Hz
    nseg = 3500 # num baseline samples (cage wfs are usually length 8192)

    runs = dg.fileDB['run'].unique()
    # cmap = plt.cm.get_cmap('jet', len(runs))
    # iplt = 0

    def psd_run(df_run):

        run = int(df_run.iloc[0]['run'])
        # print(df_run[view_cols])

        tb_name = 'ORSIS3302DecoderForEnergy/raw'
        raw_list = lh5_dir + df_run['raw_path'] + '/' + df_run['raw_file']

        # for now, just grab wfs from the first cycle file.
        # that should be PLENTY for a power spectrum plot
        f_raw = raw_list.values[0]
        data_raw, n_rows = sto.read_object(tb_name, f_raw, start_row=0, n_rows=n_wfs)
        wfs_all = data_raw['waveform']['values'].nda

        # wfs = wfs_all[idx.values, :] # can slice them by np array
        wfs = wfs_all[:, 0:nseg] # baseline only (8192 samples in cage)
        print(wfs.shape)

        f, p = signal.welch(wfs, clk, nperseg=nseg)
        ptot = np.sum(p, axis=0)
        y = ptot / wfs.shape[0]
        plt.semilogy(f, y, '-', lw=2, label=f'run {run}')
        # iplt += 1

        # exit()

    dg.fileDB.groupby(['run']).apply(psd_run)#, iplt)

    plt.xlabel('Frequency (Hz)', ha='right', x=0.9)
    plt.ylabel('PSD (ADC^2 / Hz)', ha='right', y=1)
    plt.legend(loc=1)
    #plt.savefig('./plots/psd_runs.pdf')
    plt.savefig('power_spectrum.png')
    
#main()
