#!/usr/bin/env python

"""
Quick function to apply selection effects using a pre-computed grid
"""

import numpy as np
import pandas as pd
import argparse
import sys
sys.path.insert(1, '../utils')
from detection_weights import selection_function

### Argument handling
argp = argparse.ArgumentParser()
argp.add_argument("--data-path", type=str, required=True, help="Path to population model you wish to apply selection effects to.")
argp.add_argument("--grid-path", type=str, required=True, help="Path to grid that has detection probabilities.")
argp.add_argument("--sensitivity", type=str, required=True, help="Nickname for the sensitivity, which should be a dataset key in the grid.")
args = argp.parse_args()

data = pd.read_hdf(args.data_path, key='underlying')
grid = pd.read_hdf(args.grid_path, key=args.sensitivity)

data['pdet'] = np.nan
data['combined_weight'] = np.nan

# NOTE: Edit these to adjust your data to have the appropriately-named series ('m1', 'q', 'z')!
valid_idxs = list(data.loc[data['tlb_merge'] > 0].index)
data.loc[valid_idxs, 'q'] = data.loc[valid_idxs,'m2'] / data.loc[valid_idxs,'m1']
data.loc[valid_idxs,'z'] = data.loc[valid_idxs,'z_merge']

data.loc[valid_idxs,'pdet'], data.loc[valid_idxs,'combined_weight'] = \
            selection_function(data.loc[valid_idxs], grid)

data = data.drop(columns=['q','z'])


# save to disk
data.to_hdf(args.data_path, key=args.sensitivity, mode='a')

