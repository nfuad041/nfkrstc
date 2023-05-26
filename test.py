#!/usr/bin/env python3
from pygama import DataGroup
import pygama.lh5 as lh5
import pygama.analysis.histograms as pgh
import pygama.analysis.peak_fitting as pgf

import matplotlib
# matplotlib.use('Agg') # when running on cori
import matplotlib.pyplot as plt
plt.style.use('clint.mpl')
from matplotlib.colors import LogNorm
import numpy as np

import calibration as cb


dg = DataGroup('/global/homes/f/fnafis/krstc/krstc.json', load=True)
# get file list and load energy data (numpy array)
# lh5_dir = os.path.expandvars(dg.config['lh5_dir'])
lh5_dir = dg.lh5_dir
dsp_list = lh5_dir + dg.fileDB['dsp_path'] + '/' + dg.fileDB['dsp_file']

dsp_list = dsp_list[0:3]
dsp_list

edata = lh5.load_nda(dsp_list, ['trapEmax'], 'ORSIS3302DecoderForEnergy/dsp')

elo, ehi, epb, etype = 0, 8000, 10, 'trapEmax' #Histogram options

ene_uncal = edata[etype] #energy_uncalibrated
raw_spectrum, raw_spectrum_bins, _ = pgh.get_hist(ene_uncal, range=(elo, ehi), dx=epb)

print(type(ene_uncal))

cb.calibrate_tl208(ene_uncal,plotFigure=True)
