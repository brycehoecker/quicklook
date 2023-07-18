import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from numba import njit

import analysis_quicklook as aq

runID = 328555
pedID = 326709
mod_list = [1, 2, 3, 4, 5, 6, 7, 8, 9,
            100, 103, 106, 107, 108, 111,
            112, 114, 115, 119, 121, 123,
            124, 125, 126]
datadir = "/data/local_outputDir"
savedir = f"/data/analysis_output/quicklook_output/run{runID}"

make = aq.Quicklooker(pedID, runID, mod_list, datadir, savedir)

event = 338

mod_nums = [100,111,114,107,6,115,123,124,112,7,119,
            108,110,121,8,103,125,126,106,9,4,5,1,3,2]
fpm_nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
            17,18,19,20,21,22,23,24]

fpm_pos = np.mgrid[0:5,0:5]
fpm_pos = zip(fpm_pos[0].flatten(),fpm_pos[1].flatten())

mod_to_fpm = dict(zip(mod_nums,fpm_nums))
fpm_to_pos = dict(zip(fpm_nums,fpm_pos))

ch_nums = np.array([[21,20,17,16,5,4,1,0],
                    [23,22,19,18,7,6,3,2],
                    [29,28,25,24,13,12,9,8],
                    [31,30,27,26,15,14,11,10],
                    [53,52,49,48,37,36,33,32],
                    [55,54,51,50,39,38,35,34],
                    [61,60,57,56,45,44,41,40],
                    [63,62,59,58,47,46,43,42]])
rot_ch_nums = np.rot90(ch_nums, k=2)
ch_to_pos = dict(zip(ch_nums.reshape(-1), np.arange(64)))
rot_ch_to_pos = dict(zip(rot_ch_nums.reshape(-1), np.arange(64)))

num_columns = 5
total_cells = num_columns * num_columns * 64
indices = np.arange(total_cells).reshape(-1, int(np.sqrt(total_cells)))
grid_ind = []
for index, mod in enumerate(mod_list):
    i, j = fpm_to_pos[mod_to_fpm[mod]]
    ch_map = dict()
    if j % 2 == 0:
        ch_map = rot_ch_to_pos
    else:
        ch_map = ch_to_pos
    #print(f"Channel Map: {ch_map}")
    j = num_columns - 1 - j
    pix_ind = np.array(indices[(8*i):8*(i+1), (8*j):8*(j+1)]).reshape(-1)
    #print(f"Pixel Index: {pix_ind}")
    for asic in range(4):
        for ch in range(16):
            grid_ind.append(int(pix_ind[ch_map[asic * 16 + ch]]))

# constant values relating to the proportion of a square covered by part of a circle
a = 0.2138
b = 0.479
c = 0.985

# corresponds to an aperture with 2 pxl radius
aperture = [[0., a, b, a, 0.],
            [a, c, 1., c, a],
            [b, 1., 1., 1., b],
            [a, c, 1., c, a],
            [0., a, b, a, 0.]]

aperture = np.asarray(aperture)

charge = np.zeros((40, 40))

waveforms = np.zeros((make.n_pixels, make.n_samples), dtype=np.float32)
make.calreader.GetR1Event(int(event), waveforms)
make.calreader.GetTimeStamp(int(event))
ev_analysis = aq.EventAnalyzer(waveforms, make.n_samples, make.n_pixels)

for i, val in enumerate(ev_analysis.charge):
    #if val > 5000:
    charge[grid_ind[i] % 40, grid_ind[i] // 40] = val
    #else:
    #    charge[grid_ind[i] % 40, grid_ind[i] // 40] = 0.

charge = np.pad(charge, (2,), constant_values=(0,)) # padding the array will make it simple to walk the aperture about the camera image

signal = []
image = np.zeros((40, 40))
for i in range(40):
    for j in range(40):
        s = np.sum(charge[i:i+5, j:j+5] * aperture)
        if s > 30000:
            image[j, i] = charge[i+2, j+2]
        signal.append(s)

print(np.mean(signal))
"""
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11, 8))

ax.hist(signal, bins=30)
ax.set_xlabel("Signal from Aperture Window (ADC ns)", fontsize=24)
ax.set_ylabel("Counts", fontsize=24)
ax.tick_params(labelsize=16)
ax.set_title("Examining Aperture Image Cleaning", fontsize=26)
ax.set_yscale("log")

fig.savefig("run328555_ev113_subthreshold_zeroing_aperture_hist.png")
plt.clf()
"""
if np.amax(image) > 0:
    maxZ = np.amax(image)
else:
    maxZ = 1.
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 15))
c = ax.pcolormesh(image, vmin=0, vmax=maxZ)
ax.axis("off")
ax.set_aspect("equal")
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(c, cax=cbar_ax)
cbar.set_label("Charge (ADC ns)", rotation=270, size=20, labelpad=24)
cbar_ax.tick_params(labelsize=16)
#cbar = fig.colorbar(c, cax=ax)
fig.suptitle(f"Run {runID} - Event {event}", fontsize=30)
fig.savefig(f"run{runID}_ev{event}_camera_image_cleaned_noinitial.png")
