#!/usr/bin/env python

"""
Function for incorporating selection effects on a population of binaries.
"""

import os
import sys
import numpy as np
import pandas as pd
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

### Argument handling
argp = argparse.ArgumentParser()
argp.add_argument("--output-path", type=str, required=True, help="Path to output hdf5 file. By default, the key of the dataset will be the network configuration/sensitivity choice.")
argp.add_argument("--psd-H1", type=str, required=True, help="Path to Hanford PSD. Should be same simple format as Observing Scenarios (https://dcc.ligo.org/LIGO-T2000012-v1/public), saved as text file with frequency (first column) and ASD (second column). Must be provided, use this argument for single-detector runs.")
argp.add_argument("--psd-L1", type=str, help="Path to Livingston PSD. Should be same simple format as Observing Scenarios (https://dcc.ligo.org/LIGO-T2000012-v1/public), saved as text file with frequency (first column) and ASD (second column).")
argp.add_argument("--psd-V1", type=str, required=True, help="Path to Virgo PSD. Should be same simple format as Observing Scenarios (https://dcc.ligo.org/LIGO-T2000012-v1/public), saved as text file with frequency (first column) and ASD (second column).")
argp.add_argument("--grid-key", type=str, required=True, help="Key for saving the grid of detection probabilities.")
argp.add_argument("--snr-thresh", type=float, help="SNR threshold for detection, if not supplied will fall back to the defaults in _PSD_defaults.")
argp.add_argument("--approx", type=str, default="IMRPhenomPv2", help="Waveform approximant to use for pdet calculations. Default=IMRPhenomPv2.")
argp.add_argument("--m1-min", type=float, default=3.0, help="Minimum value for the primary mass grid. Default=3.0")
argp.add_argument("--m1-max", type=float, default=500.0, help="Maximum value for the primary mass grid. Default=500.0")
argp.add_argument("--m1-num", type=int, default=10, help="Number of values in the m1 grid (log-spaced). Default=10")
argp.add_argument("--q-min", type=float, default=0.1, help="Minimum value for the mass ratio grid. Default=0.1")
argp.add_argument("--q-max", type=float, default=1.0, help="Maximum value for the mass ratio grid. Default=1.0")
argp.add_argument("--q-num", type=int, default=10, help="Number of values in the q grid (linear-spaced). Default=10")
argp.add_argument("--z-min", type=float, default=0.001, help="Minimum value for the mass ratio grid. Default=0.01")
argp.add_argument("--z-max", type=float, default=5.0, help="Maximum value for the mass ratio grid. Default=3.0")
argp.add_argument("--z-num", type=int, default=10, help="Number of values in the z grid (linear in dL). Default=10")
argp.add_argument("--chieff-min", type=float, default=-1.0, help="Minimum value for the effective spin grid. Default=-1.0")
argp.add_argument("--chieff-max", type=float, default=1.0, help="Maximum value for the effective spin grid. Default=1.0")
argp.add_argument("--chieff-num", type=int, default=0, help="Number of values in the z grid (linear in dL). Default=0, which doesn't use spin in the grid.")
argp.add_argument("--Ntrials", type=int, default=1000, help="Define the number of monte carlo trails used for calculating the average SNR. Default=1000")
argp.add_argument("--multiproc", type=int, help="Number of cores you want to use. If unspecified, will parallelize over all cores available on machine.")

start_time = time.time()

args = argp.parse_args()

# Determine number of cores for parallelization
Ncore = args.multiproc if args.multiproc is not None else os.cpu_count()
print("Parallelizing over {:d} cores...\n".format(Ncore))

# Construct grid
m1_vals = np.logspace(np.log10(args.m1_min), np.log10(args.m1_max), args.m1_num) # flat in log(m1)
q_vals = np.linspace(args.q_min, args.q_max, args.q_num) # flat in q
Dc_low = cosmo.comoving_distance(args.z_min)
Dc_high = cosmo.comoving_distance(args.z_max)
Dc_vals = np.linspace(Dc_low.value, Dc_high.value, args.z_num)*u.Mpc
z_vals = np.asarray([z_at_value(cosmo.comoving_distance, Dc) for Dc in Dc_vals]) # flat in Dc
# if we decide to include spins...
chieff_vals = np.asarray([0]) if args.chieff_num==0 else np.linspace(args.chieff_min, args.chieff_max, args.chieff_num) # flat in chieff
# combine to create grid [m1, q, z, chieff]
grid = np.asarray(list(itertools.product(m1_vals, q_vals, z_vals, chieff_vals)))

# convert q values to m2 values
grid = np.append(grid, np.atleast_2d(grid[:,0]*grid[:,1]).T, axis=1)
# convert to dataframe
grid = pd.DataFrame(grid, columns=['m1','q','z','chieff','m2'])
# print info about grid
print("Grid info:")
print("  m1: [{:0.2f}, {:0.2f}] ({:d} values)".format(args.m1_min, args.m1_max, args.m1_num))
print("  q: [{:0.2f}, {:0.2f}] ({:d} values)".format(args.q_min, args.q_max, args.q_num))
print("  z: [{:0.2f}, {:0.2f}] ({:d} values)".format(args.z_min, args.z_max, args.z_num))
print("  chieff: [{:0.2f}, {:0.2f}] ({:d} values)\n".format(args.chieff_min, args.chieff_max, args.chieff_num))

# Determine sensitivity and network configuration
ifos = {'H1':str(args.psd_H1)}
if args.psd_L1:
    ifos['L1'] = str(args.psd_L1)
if args.psd_V1:
    ifos['V1'] = str(args.psd_V1)
# get SNR threshold
snr_thresh = args.snr_thresh
# print info
print("Network configuration (SNR threshold = {:0.2f}):".format(snr_thresh))
for k, v in ifos.items():
    print("  {:s}: {:s}".format(k,v.split('/')[-1].split('.')[0]))
print("")

# print waveform approximant that is used
print("Using the {:s} waveform approximant\n".format(args.approx))

### Main Function ###
# set up partial functions and organize data for multiprocessing
func = partial(selection_effects.detection_probability, ifos=ifos, rho_thresh=snr_thresh, Ntrials=args.Ntrials, approx=args.approx)

# prepare data for multiprocessing
print("Preparing data for multiprocessing...")
systems_info = []
for idx, system in tqdm(grid.iterrows(), total=len(grid)):
    systems_info.append([system['m1'], system['m2'], system['z'], (0,0,system['chieff']), (0,0,system['chieff'])])
    # FIXME: This assumes that the two components have same aligned spin magnitude so that we get grid in chieff

# calculate detection probabilities and optimal SNRs
print("\nCalculating detection probabilities...")
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
grid.to_hdf(args.output_path, key=args.grid_key, mode='a')

