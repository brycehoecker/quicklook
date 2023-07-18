import os
import sys
import pwd

import argparse
from datetime import datetime
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import analysis_quicklook
from apply_gains import apply_gains

parser = argparse.ArgumentParser()
parser.add_argument('-i', metavar="runID", help="Indicate the run number of the raw data file.")
parser.add_argument('-p', metavar="pedID", help="Indicate the run number of the pedestal file.")
parser.add_argument('-e', metavar="event_index", help="Indicate the event number to use when creating the movie.")

args = parser.parse_args()

runID = args.i
pedID = args.p
event_index = args.e
username = pwd.getpwuid(os.getuid()).pw_name
if username == "ctauser":
    datadir = "/data/local_outputDir"
    savedir = "/data/analysis_output/camera_movies/run{runID}"
else:
    datadir = "/mnt/lfs7/wipac/CTA/target5and7data/runs_320000_through_329999"
    savedir = f"/data/wipac/CTA/web/analysis_output/camera_movies/run{runID}"
try:
    os.mkdir(savedir)
except Exception:
    pass
savedir = f"{savedir}/ev{event_index}_movie"
try:
    os.mkdir(savedir)
except Exception:
    pass

modlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]

print(f"runID: {runID}")
print(f"pedID: {pedID}")
print(f"Event: {event_index}")

movie = analysis_quicklook.Quicklooker(pedID, runID, modlist, datadir, savedir)

waveforms = np.zeros((movie.n_pixels, movie.n_samples), dtype=np.float32)
physHeatArr = np.zeros((movie.n_modules, 8, 8))
movie.calreader.GetR1Event(int(event_index), waveforms)
movie.calreader.GetTimeStamp(int(event_index))
time_stamp = int(movie.calreader.fCPU_s)
time_str = datetime.utcfromtimestamp(time_stamp).strftime("%Y-%m-%d %H:%M:%S")

waveforms = apply_gains(waveforms)
maxZ = np.amax(waveforms)
minZ = 0 #np.amin(waveforms)
for sample in tqdm(range(128)):
    physHeatArr[:, movie.row, movie.col] = np.reshape(waveforms[:, sample], (movie.n_modules, -1))
    heatReflectFig = plt.figure('Heat Map Skyview', (18, 15))
    gs = gridspec.GridSpec(5, 5, figure=heatReflectFig)
    gs.update(wspace=0.01, hspace=0.01)
    gs2 = gridspec.GridSpec(5, 5)
    gs2.update(bottom=0.05, top=0.08, right=0.86)

    for mod_index in range(movie.n_modules):
        loc, locReflect = movie.calc_loc(mod_index)
        if loc[1] % 2 == 0:
            physHeatArr[mod_index, :, :] = np.rot90(physHeatArr[mod_index, :, :], k=2)
        plt.figure('Heat Map Skyview')
        ax = plt.subplot(gs[locReflect])
        c = ax.pcolormesh(physHeatArr[mod_index, :, ::-1], vmin=minZ, vmax=maxZ)
        ax.axis('off')
        ax.set_aspect('equal')
    heatReflectFig.subplots_adjust(right=0.8, top=0.9, bottom=0.1)

    ax_bot = plt.subplot(gs2[:])
    ax_bot.barh(0, sample, align='edge', height=1)
    ax_bot.set_xlabel("Time (ns)", fontsize=24)
    ax_bot.set_yticks([])
    ax_bot.tick_params(axis='x', labelsize=20)
    ax_bot.set_ylim(0, 1)
    ax_bot.set_xlim(0, 127)

    cbar_ax = heatReflectFig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = heatReflectFig.colorbar(c, cax=cbar_ax)
    cbar.set_label('Photoelectrons', rotation=270, size=20, labelpad=24)
    cbar_ax.tick_params(labelsize=16)
    nl = "\n"
    heatReflectFig.suptitle(f"Run {movie.runID} Event {event_index}{nl}{time_str} UTC", fontsize=30)
    heatReflectFig.savefig(f"{movie.savedir}/run{movie.runID}_ev{event_index}_sample{sample}.png")
    plt.clf()
print(f"Movie images saved to: {savedir}")

images = []
for time in tqdm(range(128)):
    images.append(imageio.imread(f"{savedir}/run{runID}_ev{event_index}_sample{time}.png"))

imageio.mimsave(f"{savedir}/run{runID}_ev{event_index}.gif", images)
imageio.mimsave(f"{savedir}/run{runID}_ev{event_index}_centered.gif", images[40:91])
