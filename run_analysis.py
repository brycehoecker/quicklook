import os
import re

import numpy as np

import analysis_quicklook as aq

pedID = 328587
mod_list = [1, 2, 3, 4, 5, 6, 7, 8, 9,
            100, 103, 106, 107, 108, 111,
            112, 114, 115, 119, 121, 123,
            124, 125, 126]
datadir = "/data/wipac/CTA/target5and7data/runs_320000_through_329999"
runlist_file = "/data/wipac/CTA/target5and7data/runs_320000_through_329999/new_runs.txt"
datadir_files = os.listdir(datadir)

with open(runlist_file, 'r') as f:
    runs = []
    for line in f:
        runs.append(line)

runs = np.asarray(runs, dtype="int")
last_run_prev = max(runs)

rawdata = []

for line in datadir:
    if re.search(r"run[0-9]+.fits"):
        rawdata.append(line)

data = []

for line in rawdata:
    line = int(line.strip("run.fits"))
    if line > last_run_prev:
        data.append(line)

if not data:
    sys.exit()

newline = "\n"
with open(runlist_file, 'w') as f:
    for run in data:
        f.write(f"{run}{newline}")

for run in data:
    try:
        savedir = f"{datadir}/run{run}"
        make = aq.Quicklooker(pedID, run, mod_list, datadir, savedir)
        make.run_eventanalyzer()
        make.make_run_hists()
        make.make_temp_hists()
    except:
        pass
