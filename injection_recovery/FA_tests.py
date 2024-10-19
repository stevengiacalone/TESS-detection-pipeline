import numpy as np
import lightkurve as lk
import pandas as pd

def chisq(obs1, obs2, error1, error2):
    return np.sum((obs1 - obs2) ** 2 / (error1 ** 2 + error2 ** 2))

def symmetry_test(folded_lc, event_dur):
    """
    Checks for transit symmetry using a Chi-square statistic.
    Events deemed sufficiently asymmetric are labeled as false alarms.
    """
    
    transit_mask = np.abs(folded_lc.time.value) < 2*event_dur
    folded_lc = lk.TessLightCurve(time=folded_lc.time.value[transit_mask],
                                  flux=folded_lc.flux.value[transit_mask],
                                  flux_err=folded_lc.flux_err.value[transit_mask])
    
    right_lc = folded_lc[folded_lc.time.value > 0]
    x_right, y_right, yerr_right = right_lc.time.value, right_lc.flux.value, right_lc.flux_err.value
    left_lc = folded_lc[folded_lc.time.value < 0]
    x_left, y_left, yerr_left = -left_lc.time.value, left_lc.flux.value, left_lc.flux_err.value

    if (len(x_right) == 0) | (len(x_left) == 0):
        return 100

    bin_max = x_right.max()
    bin_edges = np.linspace(0, bin_max, 50)

    x_right_binned = np.array([np.mean([bin_edges[i], bin_edges[i+1]]) for i in range(len(bin_edges)-1)])
    bin_idxs = np.array(
        [np.argwhere((x_right > bin_edges[i]) & (x_right < bin_edges[i+1]))[:,0] for i in range(len(bin_edges)-1)], 
        dtype=object
    )
    y_right_binned = np.array([np.mean(y_right[idxs]) for idxs in bin_idxs])
    yerr_right_binned = np.array([np.sqrt(np.sum(yerr_right[idxs]**2))/len(yerr_right[idxs]) for idxs in bin_idxs])

    x_left_binned = np.array([np.mean([bin_edges[i], bin_edges[i+1]]) for i in range(len(bin_edges)-1)])
    bin_idxs = np.array(
        [np.argwhere((x_left > bin_edges[i]) & (x_left < bin_edges[i+1]))[:,0] for i in range(len(bin_edges)-1)], 
        dtype=object
    )
    y_left_binned = np.array([np.mean(y_left[idxs]) for idxs in bin_idxs])
    yerr_left_binned = np.array([np.sqrt(np.sum(yerr_left[idxs]**2))/len(yerr_left[idxs]) for idxs in bin_idxs])
    
    nan_mask = (np.isnan(y_right_binned)) | (np.isnan(y_left_binned))
    
    this_chisq = chisq(
        y_right_binned[~nan_mask], 
        y_left_binned[~nan_mask], 
        yerr_right_binned[~nan_mask], 
        yerr_left_binned[~nan_mask]
    ) / len(y_right_binned[~nan_mask])
    
    return this_chisq