#!/usr/bin/python3

import os
import sys
import numpy as np
import pandas as pd
import h5py
import tqdm
import emcee

import stardate as sd
from stardate.lhf import age_model
from isochrones import get_ichrone
mist = get_ichrone('mist')
tracks = get_ichrone('mist', tracks=True)

from multiprocessing import Pool

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
sys.path.append(os.getcwd())

def infer_stellar_age(df):

    # CALCULATE SOME USEFUL VARIABLES
    teff_err = .5*(df["p20_cks_steff_err1"] - df["p20_cks_steff_err2"])
    feh_err = .5*(df["p20_cks_smet_err1"] - ["df.p20_cks_smet_err2"])
    prot_err = .5*(df["prot_err1"] - df["prot_err2"])
    av_err = .5*(df["l20_Av_errp"] + df["l20_Av_errm"])
    bprp = df["gaia_phot_bp_mean_mag"] - df["gaia_phot_rp_mean_mag"]
    av = df["l20_Av"]
    if av == 0:
        av = .1

    # CALCULATE INITS
    mass, age, feh = (df["f18_Miso"], df["f18_logAiso"], df["p20_cks_smet"])
    # "accurate=True" makes more accurate, but slower
    track = tracks.generate(mass, age, feh, return_dict=True)
    EEP_init = track["eep"]
    inits = [EEP_init, df["f18_logAiso"], df["p20_cks_smet"],
             1./(df["gaia_parallax"]*1e-3), df["l20_Av"]]

    # Set up the parameter dictionary.

    iso_params = {"G": (df["phot_g_mean_mag"], df["G_err"]),
                  "BP": (df["phot_bp_mean_mag"], df["bp_err"]),
                  "RP": (df["phot_rp_mean_mag"], df["rp_err"]),
                  "teff": (df["cks_steff"], df["cks_steff_err1"]),
                  "feh": (df["cks_smet"], df["cks_smet_err1"]),
                  "logg": (df["cks_slogg"], df["cks_slogg_err1"]),
                  "parallax": (df["parallax"], df["parallax_error"])}

    # Infer an age with isochrones and gyrochronology.

    gyro_fn = "samples/{}_gyro".format(str(int(df["kepid"])).zfill(9))
    iso_fn = "samples/{}_iso".format(str(int(df["kepid"])).zfill(9))

    # Get initialization
    bprp = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]
    log10_period = np.log10(df["Prot"])
    log10_age_yrs = age_model(log10_period, bprp)
    gyro_age = (10**log10_age_yrs)*1e-9

    eep = mist.get_eep(df["koi_smass"], np.log10(gyro_age*1e9),
                       df["cks_smet"], accurate=True)

    inits = [eep, np.log10(gyro_age*1e9), df["cks_smet"],
             (1./df["parallax"])*1e3, df["Av"]]

    # Set up the star object
    iso_star = sd.Star(iso_params, Av=df["Av"], Av_err=df["Av_std"],
                       filename=iso_fn)
    gyro_star = sd.Star(iso_params, prot=df["Prot"], prot_err=df["e_Prot"],
                        Av=df["Av"], Av_err=df["Av_std"], filename=gyro_fn)

    # Run the MCMC
    iso_sampler = iso_star.fit(inits=inits, max_n=300000, save_samples=True)
    gyro_sampler = gyro_star.fit(inits=inits, max_n=300000, save_samples=True)


if __name__ == "__main__":
    #  Load the data file.
    df = pd.read_csv("data/for_ruth_masses.csv")

    list_of_dicts = []
    for i in range(len(df)):
        list_of_dicts.append(df.iloc[i].to_dict())

    print(list_of_dicts[0])
    print(len(list_of_dicts))

    p = Pool(24)
    list(p.map(infer_stellar_age, list_of_dicts))
