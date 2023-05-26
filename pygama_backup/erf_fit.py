import numpy as np
import scipy as scp
from scipy.optimize import curve_fit
from numba import guvectorize

from iminuit.cost import LeastSquares
from iminuit.cost import ExtendedBinnedNLL
from iminuit import Minuit

# def func(x,mu,sig,amp):
#     return amp*0.5*(1 + scp.special.erf((x-mu)/(sig*np.sqrt(2))))


# @guvectorize(['(float32[:], float64, float64, float64)'], '(n)->(),(),()')
# def erf_fit(wf_in, mu, sig, amp):
#     time = np.linspace(0,wf_in.shape[0]*10, wf_in.shape[0])
#     pars, covs = curve_fit(func, time, wf_in)
#     mu = pars[0]
#     sig = pars[1]
#     amp = pars[2]
    
    
def func(x,mu,sig):
    return 0.5*(1+scp.special.erf((x-mu)/(sig*np.sqrt(2))))

@guvectorize(['void(float32[:], float64[:], float64[:], float64[:])'], '(n)->(),(),()', nopython=False, cache=True)
def erf_fit_scipy(wf_in, mu, sig, amp):
    time = np.linspace(0, wf_in.shape[0]-1, wf_in.shape[0]) #conversion of clockticks->ns happens during the processing chain
    samples_to_integrate = 1000
    ampp = np.mean(wf_in[wf_in.shape[0]-1-samples_to_integrate:wf_in.shape[0]-1])
    #amp = np.mean(wf_in)
    wf_in = wf_in/ampp
    
    pars, covs = curve_fit(func, time, wf_in, p0=[40000, 10])
    
    
    mu[0] = pars[0]
    sig[0] = pars[1]
    amp[0] = ampp
    #print(mu, sig, amp)
    

from jax.scipy.special import erf

def func_iminuit(x,mu,sig):
    return 0.5*(1+erf((x-mu)/(sig*np.sqrt(2))))

@guvectorize(['void(float32[:], float64[:], float64[:], float64[:])'], '(n)->(),(),()', nopython=False, cache=True)
def erf_fit_iminuit(wf_in, mu=40000, sig=10, amp=400):
    time = np.linspace(0, wf_in.shape[0]-1, wf_in.shape[0]) #conversion of clockticks->ns happens during the processing chain, so no need to multiply by 10
    samples_to_integrate = 1000
    ampp = np.mean(wf_in[wf_in.shape[0]-1-samples_to_integrate:wf_in.shape[0]-1])
    wf_in = wf_in/ampp

    ydata_err = 0.1 
    least_squares = LeastSquares(time, wf_in, ydata_err, func)
    m = Minuit(least_squares, mu=mu, sig=sig)
    m.migrad()
    
    mu[0] = m.values['mu']
    sig[0] = m.values['sig']
    amp[0] = ampp


@guvectorize(['void(float32[:], float64[:], float64[:], float64[:])'], '(n)->(),(),()', nopython=False, cache=True)
def erf_fit_jdet(wf, mu, sig, amp):
    #print(list(zip(time,wf))[0:10])
    #iMu = int(pars_guess[0]/10)
    #iSig = int(pars_guess[1]/10)
    time = np.linspace(0, wf.shape[0]-1, wf.shape[0])
    
    pars_guess=[40000,10]
    fNSamplesToIntegrate=1000
    
    iMu = 4000 #int(tp50/10)
    iSig= 10 #int(sig/10)
    
    
    iMin = 0
    iMax = wf.shape[0]-1

    iMuLast = iMu
    iSigLast = iSig


    iMuMin = 3900
    iMuMax = 4200
    
    iSigMin = 0.1
    iSigMax = 200

    iMuTol = 2
    iSigTol = 2

    chi2 = 0
    
    fNIterMax = 20
    fNStepsToRamp = 5
    
    ampp = np.mean(wf[wf.shape[0]-1-fNSamplesToIntegrate:wf.shape[0]-1])
    
    for iIter in range(fNIterMax):
    
        i = np.arange(iMin, iMax)
        x = (i- iMu)/iSig
        f = 0.5*(1+ scp.special.erf(x/np.sqrt(2)))
        d = wf[i]/ampp - f
        chi2 = (d*d).sum()

        g = np.exp(-x*x/2)
        dg = (d*g).sum()
        dgx = (d*g*x).sum()
        dgx2 = (d*g*x*x).sum()
        dgx3 = (d*g*x*x*x).sum()
        g2 = (g*g).sum()
        g2x = (g*x*x).sum()
        g2x2 = (g*g*x*x).sum()

        gnorm = 1/np.sqrt(2*np.pi)/iSig
        g2 *= gnorm
        g2x *= gnorm
        g2x2 *= gnorm

        H11 = g2 + dgx/iSig
        H12 = g2x + dgx2/iSig - dg/iSig
        H22 = g2x2 + dgx3/iSig - 2*dgx/iSig
        detH = H11*H22 - H12*H12

        stepDownFactor = 1

        if (iIter < fNStepsToRamp):
                stepDownFactor = 1 + fNStepsToRamp - iIter

        iMu -= (H22*dg - H12*dgx)/detH/stepDownFactor
        if (iMu < iMuMin): iMu=iMuMin
        if (iMu >= iMuMax): iMu = iMuMax

        iSig -= (-H12*dg + H11*dgx)/detH/stepDownFactor
        if (iSig < iSigMin): iSig = iSigMin
        if (iSig >= iSigMax): iSig = iSigMax

        if (abs(iMu-iMuLast) < iMuTol) and (abs(iSig-iSigLast) < iSigTol):
            #if verbose: print('\niMu:',iMu, '\niMuLast:',iMuLast, '\niMuTol:',iMuTol)
            break
        iMuLast = iMu
        iSigLast = iSig
        
    mu[0] = iMu
    sig[0] = iSig
    amp[0] = ampp
        
    #return iMu*10, iSig*10, amp

