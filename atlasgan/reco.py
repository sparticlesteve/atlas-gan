"""
ATLAS image reconstruction functions.
This module uses the pyjet package.
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# System imports
import os

# External imports
import numpy as np
import pandas as pd
from pyjet import cluster, DTYPE_PTEPM


# TODO: Make this less hardcoded
def reconstruct_jets(image, eta_range=(-2.5, 2.5), phi_range=(-np.pi, np.pi)):
    """Apply FastJet (via pyjet) to the image and return a list of PseudoJets"""
    (eta_min, eta_max), (phi_min, phi_max) = eta_range, phi_range
    extent = eta_min, eta_max, phi_min, phi_max
    bins = 64
    eta_edges = np.linspace(eta_min, eta_max, bins + 1)
    phi_edges = np.linspace(phi_min, phi_max, bins + 1)
    # Take bin centers as eta, phi coordinates
    eta = (eta_edges[1:] + eta_edges[:-1]) / 2
    phi = (phi_edges[1:] + phi_edges[:-1]) / 2
    X, Y = np.meshgrid(eta, phi)

    clusters = np.zeros(eta.shape[0] * phi.shape[0], dtype=DTYPE_PTEPM)
    E = image.T
    clusters['pT'] = E.ravel()
    clusters['eta'] = X.ravel()
    clusters['phi'] = Y.ravel()
    clusters['pT'] = clusters['pT'] / np.cosh(clusters['eta'])
    event = clusters[clusters['pT']>100]
    sequence = cluster(event, R=1.0, p=-1)
    jets = sequence.inclusive_jets(ptmin=200000)
    jets = [jet for jet in jets if (abs(jet.eta) < 2.0)]
    return jets

def build_jets_dataframe(event, jets):
    """Given a list of PseudoJets, convert into a Pandas dataframe"""
    jpt = np.array([j.pt for j in jets])
    jeta = np.array([j.eta for j in jets])
    jphi = np.array([j.phi for j in jets])
    jm = np.array([j.mass for j in jets])
    return pd.DataFrame(data=dict(event=event, pt=jpt, eta=jeta, phi=jphi, m=jm))

def reconstruct_events(samples):
    """Apply jet reconstruction on all samples and return a Pandas dataframe"""
    return pd.concat([build_jets_dataframe(i, reconstruct_jets(samples[i]))
                      for i in range(samples.shape[0])])

def compute_physics_variables(images):
    reco_jets = reconstruct_events(images)
    ev_groups = reco_jets.groupby('event')
    nJet = ev_groups.size()
    sumMass = ev_groups.apply(lambda j: j.m.sum())
    return dict(nJet=nJet, sumMass=sumMass,
                jetPt=reco_jets.pt, jetEta=reco_jets.eta,
                jetPhi=reco_jets.phi)
