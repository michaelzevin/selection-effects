#!/usr/bin/env python

"""
Simple function for generating detection weights

Uses grid of detection probabilities to estimate
detection probabilities and weights

Anticipates data as Pandas dataframe with series ['m1', 'q', 'z']
"""

import numpy as np
import pandas as pd

from astropy.cosmology import Planck18

from sklearn.neighbors import KNeighborsRegressor

def normalize(x, xmin, xmax, a=0, b=1):
    # normalizes data on range [a,b]
    data_norm = (b-a)*(x-xmin) / (xmax-xmin) + a
    return data_norm


def selection_function(data, grid, pdet_only=False, **kwargs):
    """
    Gives a relative weight to each system in the dataframe `df` based on its proximity
    to the points on the grid. 
    
    Each system in `df` should have a primary mass `m1`, secondary mass `m2`, and redshift `z`
    
    This function will find the nearest Euclidean neighbor in [log(m1), q, z] space
    
    Need to specify bounds (based on the trained grid) so that the grid and data get normalized properly
    """
    # get values from grid for training
    m1_grid = np.asarray(grid['m1'])
    q_grid = np.asarray(grid['q'])
    z_grid = np.asarray(grid['z'])
    pdets = np.asarray(grid['pdet'])
    
    # get bounds based on grid
    m1_bounds = (np.round(m1_grid.min(), 2), np.round(m1_grid.max(), 2))
    q_bounds = (np.round(q_grid.min(), 2), np.round(q_grid.max(), 2))
    z_bounds = (np.round(z_grid.min(), 2), np.round(z_grid.max(), 2))
    
    # normalize to unit cube
    logm1_grid_norm = normalize(np.log10(m1_grid), np.log10(m1_bounds[0]), np.log10(m1_bounds[1]))
    q_grid_norm = normalize(q_grid, q_bounds[0], q_bounds[1])
    z_grid_norm = normalize(z_grid, z_bounds[0], z_bounds[1])
    
    # train nearest neighbor algorithm
    X = np.transpose(np.vstack([logm1_grid_norm, q_grid_norm, z_grid_norm]))
    y = np.transpose(np.atleast_2d(pdets))
    nbrs = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='ball_tree', leaf_size=30, p=2, metric='minkowski')
    nbrs.fit(X, y)
    
    # get values from dataset and normalize
    m1_data = np.asarray(data['m1'])
    q_data = np.asarray(data['q'])
    z_data = np.asarray(data['z'])
    logm1_data_norm = normalize(np.log10(m1_data), np.log10(m1_bounds[0]), np.log10(m1_bounds[1]))
    q_data_norm = normalize(q_data, q_bounds[0], q_bounds[1])
    z_data_norm = normalize(z_data, z_bounds[0], z_bounds[1])

    # Only fit for things within the grid range, assign zeros to anything outside
    good_idxs = np.argwhere(((logm1_data_norm >= 0) & (logm1_data_norm <= 1)) & \
        ((q_data_norm >= 0) | (q_data_norm <= 1)) & ((z_data_norm >= 0) & (z_data_norm <= 1))).flatten()
    
    # get pdets for the testing data
    X_fit = np.transpose(np.vstack([logm1_data_norm[good_idxs], 
            q_data_norm[good_idxs], z_data_norm[good_idxs]]))
    pdets = np.zeros(len(data))
    pdets[good_idxs] = nbrs.predict(X_fit).flatten()
    assert all([((p<=1) & (p>=0)) for p in pdets]), 'pdet is not between 0 and 1'

    
    if pdet_only==True:
        return pdets
    else:
        # cosmological VT term for fitted data
        if 'cosmo' in kwargs:
            cosmo = kwargs['cosmo']
        else:
            cosmo = Planck18
        cosmo_weight = cosmo.differential_comoving_volume(z_data) * (1+z_data)**(-1.0)
        combined_weight = pdets * cosmo_weight.value
        #combined_weight /= np.sum(combined_weight)
        return pdets, combined_weight
    
