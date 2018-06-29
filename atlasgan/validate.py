"""
This module contains code for evaluating metrics in GAN validation.
"""

# External imports
import numpy as np
import scipy.stats

def ks_metric(x, y):
    """Negative log of the KS test p-value"""
    pval = scipy.stats.ks_2samp(x, y).pvalue
    # Clip the p-value to avoid log(0)
    min_pval = np.exp(-700)
    if pval < min_pval: pval = min_pval
    return -np.log(pval)

def compute_metrics(real_vars, fake_vars):
    """
    Compute the physics variable metrics.
    - The negative log of the KS-test pvalues.
    """
    results = {}
    for key in fake_vars.keys():
        results['ks_' + key] = ks_metric(real_vars[key], fake_vars[key])
    return results
