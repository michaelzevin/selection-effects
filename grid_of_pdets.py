#!/usr/bin/env python

"""
Function for incorporating selection effects on a population of binaries.
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py
import time
import argparse
import itertools
from functools import partial
import multiprocessing
import pdb

from tqdm import tqdm

import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value

from utils import selection_effects

### Specify PSD information
_PSD_defaults = selection_effects._PSD_defaults


### Argument handling
argp = argparse.ArgumentParser()
argp.add_argument("--output-path", type=str, required=True, help="Path to output hdf5 file. By default, the key of the dataset will be ")
argp.add_argument("--psd-path", type=str, required=True, help="Path to directory with PSD files, saved in same format as Observing Scenarios data release.")
argp.add_argument("--psd", type=str, required=True, help="Nickname for PSD setup to be used as defined in _PSD_defaults.")
argp.add_argument("--snr-thresh", type=float, help="SNR threshold for detection, if not supplied will fall back to the defaults in _PSD_defaults.")
argp.add_argument("--m1-min", type=float, default=5.0, help="Minimum value for the primary mass grid. Default=5.0")
argp.add_argument("--m1-max", type=float, default=500.0, help="Maximum value for the primary mass grid. Default=200.0")
argp.add_argument("--m1-num", type=int, default=100, help="Number of values in the m1 grid (log-spaced). Default=100")
argp.add_argument("--q-min", type=float, default=0.1, help="Minimum value for the mass ratio grid. Default=0.1")
argp.add_argument("--q-max", type=float, default=1.0, help="Maximum value for the mass ratio grid. Default=1.0")
argp.add_argument("--q-num", type=int, default=100, help="Number of values in the q grid (linear-spaced). Default=100")
argp.add_argument("--z-min", type=float, default=0.001, help="Minimum value for the mass ratio grid. Default=0.01")
argp.add_argument("--z-max", type=float, default=2.0, help="Maximum value for the mass ratio grid. Default=2.0")
argp.add_argument("--z-num", type=int, default=100, help="Number of values in the z grid (linear in dL). Default=100")
argp.add_argument("--Ntrials", type=int, default=1000, help="Define the number of monte carlo trails used for calculating the average SNR. Default=1000")
argp.add_argument("--multiproc", type=int, help="Number of cores you want to use. If unspecified, will parallelize over all cores available on machine.")

start_time = time.time()

args = argp.parse_args()

# Determine number of cores for parallelization
Ncore = args.multiproc if args.multiproc is not None else os.cpu_count()
print("Parallelizing over {:d} cores...\n".format(Ncore))

# Construct grid
m1_vals = np.logspace(np.log10(args.m1_min), np.log10(args.m1_max), args.m1_num) # flat in log
q_vals = np.linspace(args.q_min, args.q_max, args.q_num) # flat in lin
Dc_low = cosmo.comoving_distance(args.z_min) # flat in Dc
Dc_high = cosmo.comoving_distance(args.z_max)
Dc_vals = np.linspace(Dc_low.value, Dc_high.value, args.z_num)*u.Mpc
z_vals = np.asarray([z_at_value(cosmo.comoving_distance, Dc) for Dc in Dc_vals])
# combine to create grid [m1, q, z]
grid = np.asarray(list(itertools.product(m1_vals, q_vals, z_vals)))
# convert q values to m2 values
grid = np.append(grid, np.atleast_2d(grid[:,0]*grid[:,1]).T, axis=1)
# convert to dataframe
grid = pd.DataFrame(grid, columns=['m1','q','z','m2'])
# print info about grid
print("Grid info:")
print("  m1: [{:0.1f}, {:0.1f}] ({:d} values)".format(args.m1_min, args.m1_max, args.m1_num))
print("  q: [{:0.1f}, {:0.1f}] ({:d} values)".format(args.q_min, args.q_max, args.q_num))
print("  z: [{:0.1f}, {:0.1f}] ({:d} values)\n".format(args.z_min, args.z_max, args.z_num))

# Determine sensitivity and network configuration
ifos = _PSD_defaults[args.psd]
# get SNR threshold
if args.snr_thresh is not None:
    snr_thresh = args.snr_thresh
else:
    if "network" in args.psd:
        snr_thresh = _PSD_defaults['snr_network']
    else:
        snr_thresh = _PSD_defaults['snr_single']
# print info
print("Network configuration (SNR threshold = {:0.1f}):".format(snr_thresh))
for k, v in ifos.items():
    print("  {:s}: {:s}".format(k,v))

### Main Function ###
# set up partial functions and organize data for multiprocessing
func = partial(selection_effects.detection_probability, ifos=ifos, rho_thresh=snr_thresh, Ntrials=args.Ntrials, psd_path=args.psd_path)

# prepare data for multiprocessing
systems_info = []
for idx, system in grid.iterrows():
    systems_info.append([system['m1'], system['m2'], system['z'], (0,0,0), (0,0,0)])

# calculate detection probabilities and optimal SNRs
if args.multiproc == 1:
    results = []
    for system in tqdm(systems_info):
        results.append(func(system))
else:
    mp = int(Ncore)
    pool = multiprocessing.Pool(mp)
    results = pool.imap(func, systems_info)
    results = np.asarray(list(results))
    pool.close()
    pool.join()

# add info to grid
results = np.reshape(results, (len(results),2))
grid['pdet'] = results[:,0]
grid['snr_opt'] = results[:,1]

# print time it took
end_time = time.time()
print("\nFinished! It took {:0.2f}s to run {:d} systems over {:d} cores!".format(end_time-start_time, len(grid), Ncore))

# save to disk
grid.to_hdf(os.path.join(args.output_path, 'pdet_grid.hdf5'), key=args.psd, mode='a')

