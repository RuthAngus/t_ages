#!/usr/bin/python3

import os
import sys
import numpy as np
import pandas as pd
import h5py
import tqdm
import emcee
import corner
import h5py as h5

import stardate as sd
import stardate2 as sd2
from stardate import load_samples, read_samples
from isochrones import get_ichrone
mist = get_ichrone('mist')
tracks = get_ichrone('mist', tracks=True)

from multiprocessing import Pool

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
sys.path.append(os.getcwd())

def infer_stellar_age(df):
    print("testing 2")

#    # CALCULATE SOME USEFUL VARIABLES
#    teff_err = .5*(df["p20_cks_steff_err1"] - df["p20_cks_steff_err2"])
#    feh_err = .5*(df["p20_cks_smet_err1"] - df["p20_cks_smet_err2"])
#    prot_err = .5*(df["prot_err1"] - df["prot_err2"])
#    av_err = .5*(df["l20_Av_errp"] + df["l20_Av_errm"])
#    av = df["l20_Av"]
#    bprp = df["gaia_phot_bp_mean_mag"] - df["gaia_phot_rp_mean_mag"]

#    # Now make sure you don't initialize at Av = 0.
#    init_av = av*1
#    init_av_err = av*1
#    if av == 0:
#        init_av = .1
#    if av_err == 0:
#        init_av_err = .1

#    # CALCULATE INITS
#    mass, age, feh = (df["f18_Miso"], df["f18_logAiso"], df["p20_cks_smet"])
#    # "accurate=True" makes more accurate, but slower
#    track = tracks.generate(mass, age, feh, return_dict=True)
#    EEP_init = track["eep"]
#    inits = [EEP_init, df["f18_logAiso"], df["p20_cks_smet"],
#            1./(df["gaia_parallax"]*1e-3), init_av]
#    print("inits = ", inits)

#    # Set up the parameter dictionary.
#    iso_params = {"G": (df["gaia_phot_g_mean_mag"], .01),
#                  "teff": (df["p20_cks_steff"], teff_err),
#                  "feh": (df["p20_cks_smet"], feh_err),
#                  "parallax": (df["gaia_parallax"],
#                               df["gaia_parallax_error"])}
#    print("iso_params = ", iso_params)

#    # define filenames for saving samples and remove existing files.
#    fn = "samples/gyro_{}".format(str(int(df["dr25_kepid"])).zfill(9))
#    if os.path.exists(fn):
#        os.remove(fn)

#    #---GYRO-----------------------------------------------------------------
#    ndim, nwalkers = 1, 25
#    p0 = np.random.randn(nwalkers, ndim)*1e-2 + inits[1]
#    sampler = emcee.EnsembleSampler(
#        nwalkers, ndim, sd2.lnprob, args=[df["prot"], prot_err, bprp,
#                                          sd2.angus_2019_model])
#    print("Calculating gyro age...")
#    state = sampler.run_mcmc(p0, 10);
#    sampler.reset()
#    sampler.run_mcmc(state, 100);
#    gyro_samples = sampler.get_chain(flat=True)

#    # Save the samples
#    with h5py.File(f"{fn}.h5", "w") as f:
#        dset = f.create_dataset("gyro_samples", np.shape(gyro_samples),
#                                data=gyro_samples)

#    #---ISO------------------------------------------------------------------
#    fn = "samples/iso_{}".format(str(int(df["dr25_kepid"])).zfill(9))
#    iso_star = sd.Star(iso_params, prot=None, prot_err=None,
#                       Av=av, Av_err=init_av_err, filename=fn)

#    # Run the MCMC
#    iso_star.fit(inits=inits, max_n=1000, burnin=1, save_samples=True)

#    # Get the highest likelihood parameters.
#    fn2 = f"{fn}.h5"
#    flatsamples, _3Dsamples, posterior_samples, prior_samples = \
#        load_samples(fn2, burnin=0)
#    results = read_samples(flatsamples)
#    best = [float(results.EEP_ml.values),
#            np.log10(1e9*float(results.age_ml_gyr.values)),
#            float(results.feh_ml.values),
#            float(results.distance_ml.values), float(results.Av_ml.values)]

#    # fig = corner.corner(iso_star.samples);
#    # fig.savefig("corner_test")

#    #---COMBO----------------------------------------------------------------
#    fn = "samples/combo_{}".format(str(int(df["dr25_kepid"])).zfill(9))
#    combo_star = sd.Star(iso_params, prot=df["prot"], prot_err=prot_err,
#                         Av=av, Av_err=init_av_err, filename=fn)

#    # Run the MCMC
#    combo_star.fit(inits=best, max_n=1000, burnin=1, save_samples=True)

#    # # Get the highest likelihood parameters.
#    # fn2 = f"{fn}.h5"
#    # flatsamples, _3Dsamples, posterior_samples, prior_samples = \
#    #     load_samples(fn2, burnin=0)
#    # results = read_samples(flatsamples)
#    # fig = corner.corner(combo_star.samples);
#    # fig.savefig("corner_test")


##----------------------------------------------------------------------------

if __name__ == "__main__":
    print("testing 1")

    #  Load the data file.
    df = pd.read_csv("data/for_ruth_masses.csv")

    list_of_dicts = []
    for i in range(len(df)):
        list_of_dicts.append(df.iloc[i].to_dict())

    print(list_of_dicts[0])
    print(len(list_of_dicts))

    infer_stellar_age(list_of_dicts[0])
#    # p = Pool(24)
#    # list(p.map(infer_stellar_age, list_of_dicts))
