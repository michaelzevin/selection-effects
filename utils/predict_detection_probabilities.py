#!/usr/bin/env python

"""
Simple function for generating detection weights

Uses grid of detection probabilities to estimate
detection probabilities and weights

Anticipates data as Pandas dataframe with series ['m1', 'q', 'z']
"""

import numpy as np
import pandas as pd

import os
import pickle

from astropy.cosmology import Planck18

from sklearn.neighbors import KNeighborsRegressor

def normalize(x, xmin, xmax, a=0, b=1):
    # normalizes data on range [a,b]
    data_norm = (b-a)*(x-xmin) / (xmax-xmin) + a
    return data_norm


def pdets_from_grid(data, grid, pdet_only=False, chieff=False, **kwargs):
    """
    Gives relative weight to each system in `data`
    based on its proximity to the points on the grid.

    Each system in `data` should have a primary mass `m1`, mass ratio `q`, and redshift `z`
    Can also include effective spin by setting chieff=True

    This function will determine detection probabilities using nearest neighbor algorithm
    in [log(m1), q, log(z), chieff] space

    Need to specify bounds (based on the trained grid) so that the grid and data get
    normalized properly
    """
    # get values from grid for training
    pdets = np.asarray(grid['pdet'])
    m1_grid = np.asarray(grid['m1'])
    q_grid = np.asarray(grid['q'])
    z_grid = np.asarray(grid['z'])
    if chieff:
        chieff_grid = np.asarray(grid['chieff'])

    # get bounds based on grid
    m1_bounds = (np.round(m1_grid.min(), 5), np.round(m1_grid.max(), 5))
    q_bounds = (np.round(q_grid.min(), 5), np.round(q_grid.max(), 5))
    z_bounds = (np.round(z_grid.min(), 5), np.round(z_grid.max(), 5))
    if chieff:
        chieff_bounds = (np.round(chieff_grid.min(), 5), np.round(chieff_grid.max(), 5))

    # normalize to unit cube
    logm1_grid_norm = normalize(np.log10(m1_grid), np.log10(m1_bounds[0]), np.log10(m1_bounds[1]))
    q_grid_norm = normalize(q_grid, q_bounds[0], q_bounds[1])
    logz_grid_norm = normalize(np.log10(z_grid), np.log10(z_bounds[0]), np.log10(z_bounds[1]))
    if chieff:
        chieff_grid_norm = normalize(chieff_grid, chieff_bounds[0], chieff_bounds[1])

    # train nearest neighbor algorithm
    if chieff:
        X = np.transpose(np.vstack([logm1_grid_norm, q_grid_norm, logz_grid_norm, chieff_grid_norm]))
    else:
        X = np.transpose(np.vstack([logm1_grid_norm, q_grid_norm, z_grid_norm]))
    y = np.transpose(np.atleast_2d(pdets))
    nbrs = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='ball_tree', leaf_size=30, p=2, metric='minkowski')
    nbrs.fit(X, y)

    # get values from dataset and normalize
    m1_data = np.asarray(data['m1'])
    q_data = np.asarray(data['q'])
    z_data = np.asarray(data['z'])
    if chieff:
        chieff_data = np.asarray(data['chieff'])

    logm1_data_norm = normalize(np.log10(m1_data), np.log10(m1_bounds[0]), np.log10(m1_bounds[1]))
    q_data_norm = normalize(q_data, q_bounds[0], q_bounds[1])
    logz_data_norm = normalize(np.log10(z_data), np.log10(z_bounds[0]), np.log10(z_bounds[1]))
    if chieff:
        chieff_data_norm = normalize(chieff_data, chieff_bounds[0], chieff_bounds[1])

    # get pdets for the testing data
    if chieff:
        X_fit = np.transpose(np.vstack([logm1_data_norm, q_data_norm,
                                        logz_data_norm, chieff_data_norm]))
    else:
        X_fit = np.transpose(np.vstack([logm1_data_norm,
                                        q_data_norm, logz_data_norm]))
    pdets = nbrs.predict(X_fit).flatten()
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


def create_pickle_from_pdet_grid(gridpath, sensitivity_key, chieff=False, **kwargs):
    """
    Creates pickled file of k-nearest-neighbors model for a particular sensitivity.
    """
    grid = pd.read_hdf(gridpath, key=sensitivity_key)

    # get values from grid for training
    pdets = np.asarray(grid['pdet'])
    m1_grid = np.asarray(grid['m1'])
    q_grid = np.asarray(grid['q'])
    z_grid = np.asarray(grid['z'])
    if chieff:
        chieff_grid = np.asarray(grid['chieff'])

    # get bounds based on grid
    m1_bounds = (np.round(m1_grid.min(), 5), np.round(m1_grid.max(), 5))
    q_bounds = (np.round(q_grid.min(), 5), np.round(q_grid.max(), 5))
    z_bounds = (np.round(z_grid.min(), 5), np.round(z_grid.max(), 5))
    if chieff:
        chieff_bounds = (np.round(chieff_grid.min(), 5), np.round(chieff_grid.max(), 5))

    # normalize to unit cube
    logm1_grid_norm = normalize(np.log10(m1_grid), np.log10(m1_bounds[0]), np.log10(m1_bounds[1]))
    q_grid_norm = normalize(q_grid, q_bounds[0], q_bounds[1])
    logz_grid_norm = normalize(np.log10(z_grid), np.log10(z_bounds[0]), np.log10(z_bounds[1]))
    if chieff:
        chieff_grid_norm = normalize(chieff_grid, chieff_bounds[0], chieff_bounds[1])

    # train nearest neighbor algorithm
    if chieff:
        X = np.transpose(np.vstack([logm1_grid_norm, q_grid_norm, logz_grid_norm, chieff_grid_norm]))
    else:
        X = np.transpose(np.vstack([logm1_grid_norm, q_grid_norm, z_grid_norm]))
    y = np.transpose(np.atleast_2d(pdets))
    nbrs = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='ball_tree', leaf_size=30, p=2, metric='minkowski')
    nbrs.fit(X, y)

    # need to also save bounds of grid for normalization
    grid_bounds = {}
    grid_bounds['m1'] = m1_bounds
    grid_bounds['q'] = q_bounds
    grid_bounds['z'] = z_bounds
    if chieff:
        grid_bounds['chieff'] = chieff_bounds

    import pdb; pdb.set_trace()
    # save pickled model in the same directory as the grid
    pkl_dict = {'model': nbrs, 'bounds': grid_bounds}
    filepath = os.path.join(os.path.dirname(gridpath), sensitivity_key+'.pkl')
    pickle.dump(pkl_dict, open(filepath, 'wb'))



def pdets_from_pickle(data, picklepath, chieff=False, **kwargs):
    """
    Reads in pickled model file, gives relative weight to each system in `data`
    based on its proximity to the points on the grid.

    Each system in `data` should have a primary mass `m1`, mass ratio `q`, and redshift `z`
    Can also include effective spin by setting chieff=True

    This function will determine detection probabilities using nearest neighbor algorithm
    in [log(m1), q, log(z), chieff] space

    Need to specify bounds (based on the trained grid) so that the grid and data get
    normalized properly
    """
    # read in pickle file
    pkl_dict = pickle.load(open(picklepath, 'rb'))

    # get bounds based on grid
    m1_bounds = pkl_dict['bounds']['m1']
    q_bounds = pkl_dict['bounds']['q']
    z_bounds = pkl_dict['bounds']['z']
    if chieff:
        chieff_bounds = pkl_dict['bounds']['chieff']

    # get trained nearest neighbors model
    nbrs = pkl_dict['model']

    # get values from dataset and normalize
    m1_data = np.asarray(data['m1'])
    q_data = np.asarray(data['q'])
    z_data = np.asarray(data['z'])
    if chieff:
        chieff_data = np.asarray(data['chieff'])

    logm1_data_norm = normalize(np.log10(m1_data), np.log10(m1_bounds[0]), np.log10(m1_bounds[1]))
    q_data_norm = normalize(q_data, q_bounds[0], q_bounds[1])
    logz_data_norm = normalize(np.log10(z_data), np.log10(z_bounds[0]), np.log10(z_bounds[1]))
    if chieff:
        chieff_data_norm = normalize(chieff_data, chieff_bounds[0], chieff_bounds[1])

    # get pdets for the testing data
    if chieff:
        X_fit = np.transpose(np.vstack([logm1_data_norm, q_data_norm,
                                        logz_data_norm, chieff_data_norm]))
    else:
        X_fit = np.transpose(np.vstack([logm1_data_norm,
                                        q_data_norm, logz_data_norm]))
    pdets = nbrs.predict(X_fit).flatten()
    assert all([((p<=1) & (p>=0)) for p in pdets]), 'pdet is not between 0 and 1'

    return pdets
