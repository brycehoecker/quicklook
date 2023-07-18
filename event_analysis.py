import target_driver
import target_io
import target_calib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.gridspec as gridspec
import datetime
import time
import sys
import os
import pickle

runID = 327634
eventID = 1425

datadir = "/data/software/CCC/TargetCalib/trunk/test"
uncalfile = f"{datadir}/run{runID}.fits"
calfile = f"{datadir}/run{runID}_calibrated.r1"

homedir = os.environ['HOME']
savedir = "/data/analysis_output"

print(f"Reading file: {uncalfile}")

uncalreader = target_io.WaveformArrayReader(uncalfile)
n_pixels_uncal = uncalreader.fNPixels
n_samples_uncal = uncalreader.fNSamples
n_events_uncal = uncalreader.fNEvents

uncal_waveforms = np.zeros((n_pixels_uncal, n_samples_uncal), dtype=np.uint16)

uncalreader.GetR0Event(eventID, uncal_waveforms)
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(18, 15))

ax[0].plot(uncal_waveforms[1456, :])
ax[0].set_ylim(-100, 4000)

print(f"Reading file: {calfile}")

calreader = target_io.WaveformArrayReader(calfile)
n_pixels_cal = calreader.fNPixels
n_samples_cal = calreader.fNSamples

cal_waveforms = np.zeros((n_pixels_cal, n_samples_cal), dtype=np.float32)

calreader.GetR1Event(eventID, cal_waveforms)
ax[1].plot(cal_waveforms[1456, :])
ax[1].set_ylim(-100, 4000)

print(np.amax(cal_waveforms[1456, :]))

fig.savefig(f"{savedir}/run{runID}_ev{eventID}_pxl1104_comparison.png")
print("Event images saved to: {savedir}")

