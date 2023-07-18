import numpy as np
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

shower_events = []

@njit
def f(x):
    if x > 800:
        return x
    else:
        return 0
drop_packets = []

for event in tqdm(range(0, make.n_events, 10)):
#for event in tqdm(range(61073, 61074)):
    waveforms = np.zeros((make.n_pixels, make.n_samples), dtype=np.float32)
    make.calreader.GetR1Event(int(event), waveforms)
    ev_analysis = aq.EventAnalyzer(waveforms, make.n_samples, make.n_pixels)
    charge = ev_analysis.charge
    no_charge = [val for val in charge if val == 0.0]
    #print(len(no_charge))
    if len(no_charge) > 750:
        drop_packets.append(int(event))
        continue
    else:
        continue
    """
    signal = [f(val) for val in charge]
    brights = [val for val in signal if val > 5000]
    #print(f"Bright Pixels in Event {int(event)}: {len(brights)})")
    if len(brights) > 300 or len(brights) < 5:
        continue
    shower_events.append(int(event))
    """
print(f"Dropped Packet Events / Total Events: {len(drop_packets)}/{len(range(0, make.n_events, 10))}")
#print(shower_events)

