import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk

from astropy import constants
G = constants.G.cgs.value
Msun = constants.M_sun.cgs.value
Rsun = constants.R_sun.cgs.value
Rearth = constants.R_earth.cgs.value

import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from pytransit import QuadraticModel
tm = QuadraticModel(interpolate=False)

# load TESS limb darkening coefficients
ldc_T = pd.read_csv('ldc_tess.csv')
ldc_T_Zs = np.array(ldc_T.Z, dtype=float)
ldc_T_Teffs = np.array(ldc_T.Teff, dtype=int)
ldc_T_loggs = np.array(ldc_T.logg, dtype=float)
ldc_T_u1s = np.array(ldc_T.aLSM, dtype=float)
ldc_T_u2s = np.array(ldc_T.bLSM, dtype=float)

def get_ldc(logg, Teff):
    this_logg = ldc_T_loggs[np.argmin(np.abs(ldc_T_loggs-logg))]
    this_Teff = ldc_T_Teffs[np.argmin(np.abs(ldc_T_Teffs-Teff))]
    mask = (ldc_T_Teffs == this_Teff) & (ldc_T_loggs == this_logg)
    u1, u2 = ldc_T_u1s[mask], ldc_T_u2s[mask]
    return u1[0], u2[0]

def transit_model(t, period, depth, inc, Ms, Rs, logg, Teff, max_exptime):
    k = np.sqrt(depth)
    t0 = np.zeros(len(k))
    P_orb = np.full_like(k, period)
    a = (G*Ms*Msun/(2*np.pi)**2 * (period*86400)**2)**(1/3) / (Rs*Rsun)
    a = np.full_like(k, a)
    incs = inc*(np.pi/180.)
    ecc = np.zeros(len(k))
    w = np.zeros(len(k))
    pvp = np.array([k, t0, P_orb, a, incs, ecc, w]).T
    u1, u2 = get_ldc(logg, Teff)
    u1 = np.full_like(k, u1)
    u2 = np.full_like(k, u2)
    ldc = np.array([u1, u2]).T
    tm.set_data(t, exptimes=max_exptime, nsamples=20)
    flux = tm.evaluate_pv(pvp=pvp, ldc=ldc)
    return flux

def fit_transit(t, y, yerr, period, depth_guess, Ms, Rs, logg, Teff, max_exptime):
    '''Fit transit model to the input time sequence, and return fitting parameters "rp_rs", "inc", and "BIC"'''
    t = np.array(t)
    y = np.array(y)
    yerr = np.array(yerr)

    def func(depth, inc):  return 0.5*np.sum(((transit_model(t, period, depth, inc, Ms, Rs, logg, Teff, max_exptime) - y)/yerr)**2, axis=1)
    def func2(depth, inc):  return 0.5*np.sum(((transit_model(t, period, depth, inc, Ms, Rs, logg, Teff, max_exptime) - y)/yerr)**2)
    a = (G*Ms*Msun/(2*np.pi)**2 * (period*86400)**2)**(1/3)
    inc_min = np.arccos((Rs*Rsun)/a) * 180./np.pi

    depths = np.linspace(depth_guess, 1, 500)
    incs = np.arange(np.ceil(inc_min), 91, 1)
    depths, incs = np.meshgrid(depths, incs)[0].flatten(), np.meshgrid(depths, incs)[1].flatten()
    all_BICs = func(depths, incs)

    best_idx = all_BICs.argmin()
    best_depth, best_inc = np.array([depths[best_idx]]), np.array([incs[best_idx]])

    k = 2
    n = len(t)
    BIC = k*np.log(n) - 2*(-func2(best_depth, best_inc))
    return {"rp_rs": np.sqrt(best_depth), "inc": best_inc, "BIC": BIC}

def fit_sin(t, y, yerr, period):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "per", "offset", and "BIC"'''
    t = np.array(t)
    y = np.array(y)
    yerr = np.array(yerr)

    def func(A, P, O):  return 0.5*np.sum(((1 - A * np.sin(2*np.pi/P*t + O) - y)/yerr)**2)

    pers = period/np.arange(1,31,1)
    amps = np.linspace(0, np.std(y) * 2, 100)
    offs = np.array([-np.pi/2, np.pi/2])
    all_BICs = np.zeros([len(pers), len(amps), len(offs)])
    for i in range(len(pers)):
        for j in range(len(amps)):
            for k in range(len(offs)):
                all_BICs[i,j,k] = func(amps[j], pers[i], offs[k])

    idx1, idx2, idx3 = np.unravel_index(all_BICs.argmin(), all_BICs.shape)
    best_P, best_A, best_O = pers[idx1], amps[idx2], offs[idx3]

    k = 3
    n = len(t)
    BIC = k*np.log(n) - 2*(-func(best_A, best_P, best_O))
    return {"amp": best_A, "per": best_P, "offset": best_O, "BIC": BIC}

def bin_folded_lc(time, flux, flux_err, n_points):
    n_points_per_bin = n_points/(time.max() - time.min())
    bin_edges = np.arange(time.min(), time.max(), 1/n_points_per_bin)
    time_binned = np.array([np.mean([bin_edges[i], bin_edges[i+1]]) for i in range(len(bin_edges)-1)])
    bin_idxs = np.array([np.argwhere((time > bin_edges[i]) & (time < bin_edges[i+1]))[:,0] for i in range(len(bin_edges)-1)], dtype=object)
    flux_binned = np.array([np.mean(flux[idxs]) for idxs in bin_idxs])
    flux_err_binned = np.array([np.sqrt(np.sum(flux_err[idxs]**2))/len(flux_err[idxs]) for idxs in bin_idxs])
    return time_binned, flux_binned, flux_err_binned

def remove_nans(t, y, yerr):
    mask = ~np.isnan(y)
    return t[mask], y[mask], yerr[mask]

def calculate_dBIC(lc_folded, period, depth_guess, Ms, Rs, logg, Teff, max_exptime, plot=False):
    '''Calculate Delta BIC between the two models'''
    # bin the folded light curve a little
    t = lc_folded.time.value
    y = lc_folded.flux.value
    yerr = lc_folded.flux_err.value
    t, y, yerr = bin_folded_lc(t, y, yerr, 1000)
    t, y, yerr = remove_nans(t, y, yerr)
    # calculate dBIC
    res_sin = fit_sin(t, y, yerr, period)
    res_tra = fit_transit(t, y, yerr, period, depth_guess, Ms, Rs, logg, Teff, max_exptime)
    dBIC = res_sin["BIC"] - res_tra["BIC"]
    
    test_sin_flux = 1 - res_sin["amp"] * np.sin(2*np.pi/res_sin["per"]*t + res_sin["offset"])
    test_transit_flux = transit_model(t, period,
                              res_tra["rp_rs"]**2, res_tra["inc"], Ms, Rs, logg, Teff, max_exptime)
    
    if plot == True:
        print(res_sin)
        print(res_tra)
        print("dBIC = ", dBIC)
        plt.figure(figsize=(10,8))
        plt.plot(t, y, 'k.')
        plt.plot(t, test_sin_flux, 'r--', lw=5)
        plt.plot(t, test_transit_flux, 'b-', lw=5)
        plt.xlim([-1, 1])
        plt.xlabel("Days from Transit Center", fontsize=16)
        plt.ylabel("Normalized Flux", fontsize=16)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10,8))
        plt.plot(t, y, 'k.')
        plt.plot(t, test_sin_flux, 'r--', lw=5)
        plt.plot(t, test_transit_flux, 'b-', lw=5)
        # plt.xlim([-1, 1])
        plt.xlabel("Days from Transit Center", fontsize=16)
        plt.ylabel("Normalized Flux", fontsize=16)
        plt.tight_layout()
        plt.show()

    return dBIC