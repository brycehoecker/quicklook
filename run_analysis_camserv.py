from multiprocessing import Process
import os
import sys
import re

import numpy as np

def main():
    pedID = 328587  # it would be better if this was temperature dependent
    datadir = "/data/local_outputDir"  # camera server only
    runlist_file = f"{datadir}/new_runs.txt"  # camera server only

    datadir_files = os.listdir(datadir)  # get list of strs of dir entries

    with open(runlist_file, 'r') as f:
        runs = []
        for line in f:
            runs.append(line)  # get list of strs of run numbers in file

    runs = np.asarray(runs, dtype="int")  # make these run numbers ints
    last_run_prev = max(runs)  # find the most recently processed run number

    rawdata = []

    for line in datadir_files:
        if re.search(r"run[0-9]+.fits", line):
            rawdata.append(line)  # put all run numbers in dir into list

    data = []

    for line in rawdata:
        line = int(line.strip("run.fits"))
        if line > last_run_prev:
            data.append(line)  # only keep the new one(s)

    if not data:
        sys.exit()  # if there are no new ones, quit here

    newline = "\n"
    with open(runlist_file, 'w') as f:
        # basic approach here for overcoming SegFaults: use a new process
        for run in data:
            try:
                proc = Process(target=run_summary, args=(run,))
                proc.start()
                proc.join()
                # write the processed run #s to file
                f.write(f"{run}{newline}")
            except Exception as ex:
                # not sure if this will ever actually trigger, but here it is
                print(f"Run {run} failed with the following exception: {ex}")
                if data[-1] == run:
                    f.write(f"{run}{newline}")

def run_summary(run):
    os.system(f"/home/ctauser/CameraSoftware/trunk/analysis/summarize_run.sh -r {run}")  # calls an outside script

if __name__ == "__main__":
    main()  # start the summarizing process

