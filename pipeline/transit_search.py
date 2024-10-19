import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
from wotan import flatten
from wotan.gaps import get_gaps_indexes
from astropy import constants
from quick_fits import calculate_dBIC, remove_nans
from FA_tests import symmetry_test

G = constants.G.cgs.value
Msun = constants.M_sun.cgs.value
Rsun = constants.R_sun.cgs.value
Rearth = constants.R_earth.cgs.value

# load in momentum dump times
df_mom = pd.read_csv("Table_of_momentum_dumps.csv", skiprows=4)[1:]
dumps = df_mom[" TJD"].values

# load in my stars
df_Astars = pd.read_csv("FINAL_SAMPLE.csv")
IDs = df_Astars.ID.values
Tmags = df_Astars.Tmag.values
Teffs = df_Astars.Teff.values
masses = df_Astars.mass.values
rads = df_Astars.rad.values
loggs = df_Astars.logg.values

def BLS_periodogram(lc, bls_periods, Ms, Rs):
    """
    Search for periodic signals using a BLS algorithm.
    If SDE >= 7.1, check to see if sinusoid or transit model
    is preferred. 
    Params:
        lc: Light curve to search (lk.Lightcurve object)
        bls_periods: Periods to test in BLS (days)
        Ms: Mass of target star (M_Sun)
        Rs: Radius of target star (R_Sun)
    Returns:
        BLS outputs
    """
    all_periods = np.array([])
    all_powers = np.array([])
    all_epochs = np.array([])
    all_depths = np.array([])
    all_durations = np.array([])
    for i in range(19):
        start = 0.5 * (1+i)
        end = 0.5 * (2+i)
        if end < 10:
            per_mask = (bls_periods >= start) & (bls_periods < end)
        else:
            per_mask = (bls_periods >= start) & (bls_periods <= end)
        bls_durations = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        best_a = (G*Ms*Msun/(4*np.pi**2) * (end * 86400)**2)**(1/3)
        max_dur = end/np.pi * np.arcsin(Rs*Rsun/best_a)
        dur_mask = bls_durations < max_dur
        bls = lc.to_periodogram(method='bls', period=bls_periods[per_mask], frequency_factor=500, duration=bls_durations[dur_mask])
        all_periods = np.concatenate([all_periods, bls.period.value])
        all_powers = np.concatenate([all_powers, bls.power.value])
        all_epochs = np.concatenate([all_epochs, bls.transit_time.value])
        all_depths = np.concatenate([all_depths, bls.depth.value])
        all_durations = np.concatenate([all_durations, bls.duration.value])
    return all_powers, all_periods, all_epochs, all_durations, all_depths

def run_transit_search(i):

    results = np.zeros(12)
    this_ID = IDs[i]
    this_Tmag = Tmags[i]
    this_Teff = Teffs[i]
    this_mass = masses[i]
    this_rad = rads[i]
    this_logg = loggs[i]

    print("Starting", this_ID, i)

    # get the qlp data and make a light curve
    # try until you get the data
    search_res = None
    while search_res is None:
        try:
            search_res = lk.search_lightcurve(target='TIC '+str(this_ID), mission='TESS')
        except:
            pass
    
    if "QLP" in search_res.author.data:
        try:
            search_res = search_res[search_res.author.data == "QLP"].download_all()
        except:
            results[0] = this_ID
            results[-1] = -1
            return results
    else:
        results[0] = this_ID
        results[-1] = -2
        return results

#     search_res = search_res[search_res.author.data == "QLP"].download_all()
    time = np.array([])
    flux = np.array([])
    flux_err = np.array([])
    for k in range(len(search_res)):
        # only include first 5 years of TESS data
        # update if you want to use later TESS sectors
        if search_res[k].sector <= 69:
            q = search_res[k].quality.value
            qmask = (q == 0)
            time = np.concatenate([time, search_res[k].time.value[qmask]])
            flux = np.concatenate([flux, search_res[k].flux.value[qmask]])
            # year 5 data often has all nans in "flux_err" field, so use errors of detrended flux if needed
            if (np.all(np.isnan(search_res[k].flux_err.value))) and ("det_flux_err" in list(search_res[k].to_pandas())):
                flux_err = np.concatenate([flux_err, search_res[k].det_flux_err.value[qmask]])
            else:
                flux_err = np.concatenate([flux_err, search_res[k].flux_err.value[qmask]])

    # allocate max of 60 knots per sector
    n_sectors = len(search_res)

    # remove nans
    time, flux, flux_err = remove_nans(time, flux, flux_err)
    # if flux is all nans, stop here
    if len(flux) < 50:
        results[0] = this_ID
        results[-1] = -3
        return results
    
    # flatten the light curve
    segs = np.argwhere(np.diff(time) > 0.5).T[0] + 1
    segs = np.concatenate([[0], segs, [len(time)]])
    flat_time = np.array([])
    flat_flux = np.array([])
    flat_flux_err = np.array([])
    for k in range(len(segs)-1):
        this_time = time[segs[k]:segs[k+1]]
        this_flux = flux[segs[k]:segs[k+1]]
        this_flux_err = flux_err[segs[k]:segs[k+1]]
        duration = this_time[-1] - this_time[0]   
        if duration < 2:
            continue     
        flattened_flux, trend_lc, nsplines = flatten(
            this_time,                   # Array of time values
            this_flux,                   # Array of flux values
            method='pspline',
            max_splines=int(duration/0.5),  # The maximum number of knots to be tested
            edge_cutoff=0.25,        # Remove edges
            stdev_cut=3,            # Larger outliers are removed in each iteration
            return_trend=True,      # Return trend and flattened light curve
            return_nsplines=True,   # Return chosen number of knots
            verbose=False
            )

        these_dumps = dumps[(dumps > this_time.min()) & (dumps < this_time.max())]
        bitmask = np.ones(len(this_time), dtype=bool)
        for this_dump in these_dumps:
            if this_time.max() < 1682.5:
                bitmask[(this_time > this_dump-0.25) & (this_time < this_dump+0.25)] = False
            elif (this_time.min() > 1682.5) & (this_time.max() < 2034.5):
                bitmask[(this_time > this_dump-0.50) & (this_time < this_dump+0.50)] = False
            else:
                bitmask[(this_time > this_dump-0.75) & (this_time < this_dump+0.75)] = False
        
        this_lc = lk.LightCurve(
            time=this_time[bitmask], 
            flux=flattened_flux[bitmask], 
            flux_err=this_flux_err[bitmask]
        ).remove_outliers(sigma=10).remove_nans()
        flat_time = np.concatenate([flat_time, this_lc.time.value])
        flat_flux = np.concatenate([flat_flux, this_lc.flux.value])
        flat_flux_err = np.concatenate([flat_flux_err, this_lc.flux_err.value])

    lc = lk.LightCurve(time=flat_time, flux=flat_flux, flux_err=flat_flux_err)

    # identify sectors to determine cadence for supersampling
    sectors = np.zeros(len(search_res), dtype=int)
    for i in range(len(search_res)):
        sectors[i] = search_res[i].sector
    if (sectors.min() >= 56):
        max_exptime = 0.00232
    elif (sectors.min() < 56) & (sectors.min() >= 27):
        max_exptime = 0.00694
    else:
        max_exptime = 0.02083

    # run bls to find possible transits
    periodogram_periods = 1/np.linspace(0.1, 2, 100000)
    try:
        bls_powers, bls_periods, bls_epochs, bls_durations, bls_depths = BLS_periodogram(lc, periodogram_periods, this_mass, this_rad)
    except:
        results[0] = this_ID
        results[-1] = -4
        return results
    
    # identify signals with SDE >= 10 and dBIC >= 50 as likely planets
    bls_SDEs = (bls_powers - bls_powers.mean()) / bls_powers.std()
    mask = bls_SDEs >= 10
    peak_SDEs = bls_SDEs[mask]
    peak_periods = bls_periods[mask]
    peak_depths = bls_depths[mask]
    peak_epochs = bls_epochs[mask]
    peak_durs = bls_durations[mask]
    planet = 0
    while (len(peak_periods) > 0) and (planet == 0):
        # find the peaks in this periodogram and group based on distance to peaks/harmonics
        this_peak = np.argmax(peak_SDEs)
        this_peak_SDE = peak_SDEs[this_peak]
        this_peak_period = peak_periods[this_peak]
        this_peak_depth = peak_depths[this_peak]
        this_peak_epoch = peak_epochs[this_peak]
        this_peak_dur = peak_durs[this_peak]
        # mask out all nearby peaks (within +/- 2%) as well as those of harmonics
        this_mask = (
            ~((peak_periods >= this_peak_period/10*0.98) & (peak_periods <= this_peak_period/10*1.02)) &
            ~((peak_periods >= this_peak_period/9*0.98) & (peak_periods <= this_peak_period/9*1.02)) &
            ~((peak_periods >= this_peak_period/8*0.98) & (peak_periods <= this_peak_period/8*1.02)) &
            ~((peak_periods >= this_peak_period/7*0.98) & (peak_periods <= this_peak_period/7*1.02)) &
            ~((peak_periods >= this_peak_period/6*0.98) & (peak_periods <= this_peak_period/6*1.02)) &
            ~((peak_periods >= this_peak_period/5*0.98) & (peak_periods <= this_peak_period/5*1.02)) &
            ~((peak_periods >= this_peak_period/4*0.98) & (peak_periods <= this_peak_period/4*1.02)) &
            ~((peak_periods >= this_peak_period/3*0.98) & (peak_periods <= this_peak_period/3*1.02)) &
            ~((peak_periods >= this_peak_period/2*0.98) & (peak_periods <= this_peak_period/2*1.02)) &
            ~((peak_periods >= 1*this_peak_period*0.98) & (peak_periods <= 1*this_peak_period*1.02)) &
            ~((peak_periods >= 2*this_peak_period*0.98) & (peak_periods <= 2*this_peak_period*1.02)) &
            ~((peak_periods >= 3*this_peak_period*0.98) & (peak_periods <= 3*this_peak_period*1.02)) &
            ~((peak_periods >= 4*this_peak_period*0.98) & (peak_periods <= 4*this_peak_period*1.02)) &
            ~((peak_periods >= 5*this_peak_period*0.98) & (peak_periods <= 5*this_peak_period*1.02)) &
            ~((peak_periods >= 6*this_peak_period*0.98) & (peak_periods <= 6*this_peak_period*1.02)) &
            ~((peak_periods >= 7*this_peak_period*0.98) & (peak_periods <= 7*this_peak_period*1.02)) &
            ~((peak_periods >= 8*this_peak_period*0.98) & (peak_periods <= 8*this_peak_period*1.02)) &
            ~((peak_periods >= 9*this_peak_period*0.98) & (peak_periods <= 9*this_peak_period*1.02)) &
            ~((peak_periods >= 10*this_peak_period*0.98) & (peak_periods <= 10*this_peak_period*1.02)) 
        )
        # run fits to determine if planetary
        lc_folded = lc.fold(period=this_peak_period, epoch_time=this_peak_epoch)
        dBIC = calculate_dBIC(lc_folded, this_peak_period, this_peak_depth, this_mass, this_rad, this_logg, this_Teff, max_exptime)
        # symmetry test
        try:
            chisq = symmetry_test(lc_folded, this_peak_dur)
        except:
            chisq = 10
        if (dBIC >= 50) & (chisq < 2):
            planet = 1

        # go to next peak
        peak_SDEs = peak_SDEs[this_mask]
        peak_periods = peak_periods[this_mask]
        peak_depths = peak_depths[this_mask]
        peak_epochs = peak_epochs[this_mask]
        peak_durs = peak_durs[this_mask]
  
    if planet == 1:
        results = np.array([this_ID, this_Tmag, this_Teff, this_mass, this_rad, this_peak_epoch, this_peak_period, this_peak_dur, this_peak_depth, this_peak_SDE, dBIC, chisq])

    return results