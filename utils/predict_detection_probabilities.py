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

    def prepareData(self, data, setbounds=False):
                
        # Set bounds if they 
        if setbounds:
            self.bounds = { field : (np.round(data[field].min(), 5),
                                     np.round(data[field].max(), 5)) for field in self.fields}

        grids = {}
        
        # Normalize and apply logarithms if necessary
        for field in self.fields:

            # These fields get log grids
            if field in ['m1', 'z']:
                grids[field] = normalize(np.log10(data[field]),
                                         np.log10(self.bounds[field][0]),
                                         np.log10(self.bounds[field][1]))
            else:
                grids[field] = normalize(data[field],
                                         self.bounds[field][0],
                                         self.bounds[field][1])

        # Return the bounds and normalized and logarithmed grids
        return grids
    
    def __init__(self, gridspec, chieff=False, rebuild=False):

        # Specify the data format
        self.fields = ['m1', 'q', 'z']

        if chieff:
            self.fields.append('chieff')

        # Load the grid from the file and then ditch this
        # so we don't carry this process baggage with us
        self.gridfile, self.key = gridspec.split(':')

        # If we don't have a cache storage directory, make one
        if not os.path.exists("selection_cache"):
            print("LVKWeighter: cache storage directory missing, making...")
            os.mkdir("selection_cache")

        # See if we have a cached object for us already present
        cache_name = "selection_cache/%s_%s.lvkw" % (self.gridfile, self.key)

        # Load the cached version or rebuild it from the hdf5
        if os.path.exists(cache_name) and not rebuild:
            print("LVKWeighter: found cache file %s, loading..." % cache_name)
            tmp = pickle.load(open(cache_name, 'rb'))

            # Assign local attributes from the tmp
            self.gridfile = tmp.gridfile
            self.key = tmp.key
            self.fields = tmp.fields
            self.bounds = tmp.bounds
            self.nbrs = tmp.nbrs
        else:
            # Generate it and store it
            print("LVKWeighter: no cache file found, training...")
            
            # Load the grid 
            grid = pd.read_hdf(self.gridfile, key=self.key)

            # Store the bounds of the training data
            grids = self.prepareData(grid, setbounds=True)

            # Train nearest neighbor algorithm
            X = np.transpose(np.vstack(list(grids.values())))
            y = np.transpose(np.atleast_2d(grid['pdet']))

            # This is the only object we need to persist
            self.nbrs = KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='ball_tree', leaf_size=30, p=2, metric='minkowski')
            self.nbrs.fit(X, y)

            # Now delete the fit data from inside the object (we don't need it anymore and its 25% of the size)
            del(self.nbrs._fit_X)
            
            # Store ourselves pickled in cache
            pickle.dump(self, open(cache_name, 'wb'))

            # Report
            print("LVKWeighter: cache file %s done." % cache_name)

    # A multiprocessing wrapper around estimate_core()
    def estimate(self, data, pool=None, pdet_only=False, **kwargs):

        if self.bounds is None:
            raise Exception("Bounds have not been set.  Somehow the object was not initialized.")
        
        # Make a judgment call here as to when its worth going in parallel
        if not pool or (len(data) / pool._processes < 1e4):
            return self.estimate_core((data, pdet_only, kwargs))
        else:
            # Swim in the pool
            results = pool.map(self.estimate_core, [(data_chunk,
                                                     pdet_only,
                                                     kwargs) for data_chunk in np.array_split(data, pool._processes)])

            # Assemble the results
            return (np.concatenate(x) for x in zip(*results))
        
    # Perform the estimation
    def estimate_core(self, args):

        # Unpack the arguments
        data, pdet_only, kwargs = args
        
        # Preprocess the data
        grids = self.prepareData(data)

        # Get it ready to go 
        X_fit = np.transpose(np.vstack(list(grids.values())))

        # Use the persisted trained object
        pdets = self.nbrs.predict(X_fit).flatten()
        
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
            combined_weight = pdets * cosmo_weight
            #combined_weight /= np.sum(combined_weight)

            return pdets, combined_weight
