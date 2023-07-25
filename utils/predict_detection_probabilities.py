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

class LVKWeighter(object):

    def prepareData(self, data, fields=['m1', 'q', 'z']):
                
        # Set bounds
        bounds = { field : (np.round(data[field].min(), 5),
                            np.round(data[field].max(), 5)) for field in fields}

        grids = {}
        
        # Normalize and apply logarithms if necessary
        for field in fields:

            # These fields get log grids
            if field in ['m1', 'z']:
                grids[field] = normalize(np.log10(data[field]),
                                         np.log10(bounds[field][0]),
                                         np.log10(bounds[field][1]))
            else:
                grids[field] = normalize(data[field],
                                         bounds[field][0],
                                         bounds[field][1])

        # Return the bounds and normalized and logarithmed grids
        return (bounds, grids)
    
    def __init__(self, grid, chieff=False):

        fields = ['m1', 'q', 'z']

        if chieff:
            fields.append('chieff')

        # Normalize and logarithm the data
        bounds, grids = self.prepareData(grid, fields)
        
        # Train nearest neighbor algorithm
        X = np.transpose(np.vstack(list(grids.values())))
        y = np.transpose(np.atleast_2d(grid['pdet']))

        # This is the only object we need to persist
        self.nbrs = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='ball_tree', leaf_size=30, p=2, metric='minkowski')
        self.nbrs.fit(X, y)

    # Perform the estimation
    def estimate(self, data, chieff=False, pdet_only=False, **kwargs):

        fields = ['m1', 'q', 'z']

        if chieff:
            fields.append('chieff')

        # Preprocess the data
        bounds, grids = self.prepareData(data, fields)

        # Get it ready to go 
        X_fit = np.transpose(np.vstack(list(grids.values())))

        # Use the persisted trained object
        pdets = self.nbrs.predict(X_fit).flatten()
        print(pdets)
        
        assert all([((p<=1) & (p>=0)) for p in pdets]), 'pdet is not between 0 and 1'

        if pdet_only==True:
            return pdets
        else:
            # cosmological VT term for fitted data
            if 'cosmo' in kwargs:
                cosmo = kwargs['cosmo']
            else:
                cosmo = Planck18

            cosmo_weight = cosmo.differential_comoving_volume(data['z']).value * (1+data['z'])**(-1.0)
            print(cosmo_weight)
            
            combined_weight = pdets * cosmo_weight
            #combined_weight /= np.sum(combined_weight)

            print(combined_weight)
            return pdets, combined_weight
