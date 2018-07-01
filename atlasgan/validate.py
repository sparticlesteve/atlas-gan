"""
This module contains code for evaluating metrics in GAN validation.
"""

# External imports
import numpy as np
import scipy.stats

def ks_metric(x, y):
    """Negative log of the KS test p-value"""
    min_pval = np.exp(-700)
    # If either array empty, we will get errors. So we use minimum p-value
    if len(x) == 0 or len(y) == 0:
        pval = min_pval
    else:
        pval = scipy.stats.ks_2samp(x, y).pvalue
    # Clip the p-value to avoid log(0)
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
