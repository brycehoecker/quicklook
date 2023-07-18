from datetime import datetime
import os
import pwd
import sys

import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from tqdm import tqdm

from psct_reader import WaveformArrayReader as Reader

matplotlib.use("Agg")


def plot_charge_summary(time_stamp, charge_max, charge_mean, charge_std,
                        charge_uniformity, SAVEDIR, run):
    print("Generating charge summary plots...")
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 20))
    ax[0, 0].scatter(time_stamp, charge_max)
    ax[0, 0].set_title("Maximum Charge per Pixel in Event", fontsize=24)
    ax[0, 0].set_xlabel("Time from First Event (s)", fontsize=24)
    ax[0, 0].set_ylabel("Charge (Photoelectrons)", fontsize=24)

    ax[0, 1].scatter(time_stamp, charge_mean)
    ax[0, 1].set_title("Mean Charge in Event", fontsize=24)
    ax[0, 1].set_xlabel("Time from First Event (s)", fontsize=24)
    ax[0, 1].set_ylabel("Charge (Photoelectrons)", fontsize=24)

    ax[1, 0].scatter(time_stamp, charge_std)
    ax[1, 0].set_title("Standard Deviation of Charge in Event", fontsize=24)
    ax[1, 0].set_xlabel("Time from First Event (s)", fontsize=24)
    ax[1, 0].set_ylabel("Charge (Photoelectrons)", fontsize=24)

    ax[1, 1].scatter(time_stamp, charge_mean / charge_std)
    ax[1, 1].set_title("Charge Uniformity in Event", fontsize=24)
    ax[1, 1].set_xlabel("Time from First Event (s)", fontsize=24)
    ax[1, 1].set_ylabel("Charge (Photoelectrons)", fontsize=24)

    fig.suptitle(f"Charge Statistics in Run{run}", fontsize=28)
    fig.savefig(f"{SAVEDIR}/charge_stats_run{run}.png")
    plt.clf()


def plot_temperature_summary(DATADIR, SAVEDIR, run, time_stamp):
    print('Generating temperature histograms...')
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 20))

    ax[0, 0].hist(time_stamp, bins=int(time_stamp[-1]))
    ax[0, 0].set_title("Histogram of Timestamps", fontsize=24)
    ax[0, 0].set_xlabel("Time from First Event (s)", fontsize=24)
    ax[0, 0].set_ylabel("Rate (Hz)", fontsize=24)
    ax[0, 0].tick_params(labelsize=16)

    tempfile = f"{DATADIR}/{run}_temperatures.txt"
    tempdata = pd.read_csv(tempfile)
    shape = tempdata.shape
    temptime_start = [tempdata['timestamp_start'][i] for i in range(shape[0])]
    temptime_start = mdates.num2date(mdates.datestr2num(temptime_start))
    # date = datetime.date.fromisoformat(tempdata['timestamp_start'][1][:10])
    module_numbers = set()
    for sensor in tempdata.keys():
        if sensor == "timestamp_start":
            continue
        elif sensor == "timestamp_end":
            continue
        else:
            module_numbers.add(sensor.split("_")[0])
    avg_temp_per_module = {}
    avg_temps = []
    for mod in module_numbers:
        individual_sensor_values = []
        for i in range(4):
            values = tempdata[str(mod)+'_ADC'+str(i)]
            if all(val<100 for val in values):
                if all(val>4 for val in values):
                    individual_sensor_values.append(values)
        average_sensor_values = [sum(col) / float(len(col)) for col in zip(*individual_sensor_values)]
        if average_sensor_values != []:
            avg_temp_per_module[mod] = average_sensor_values
            avg_temps.append(average_sensor_values)
    camera_average = [float(sum(col))/len(col) for col in zip(*avg_temps)]
    avg_temp_per_module['camera_average'] = camera_average
    temps = avg_temp_per_module
    mod_location = [[4,5,1,3,2],[103,125,126,106,9],[119,108,110,121,8],[115,123,124,112,7],[100,111,114,107,6]]
    cmap = matplotlib.cm.get_cmap('rainbow')
    colors = [cmap(0),cmap(0.25),cmap(0.5),cmap(0.75),cmap(0.99)]
    markers = ['o', 's', 'v', 'd', 'X']
    for mod in sorted(temps.keys()):
        if mod == 'camera_average':
            marker = None
            color = 'black'
            label = 'Module Avg'
        else:
            label = 'Module ' + str(mod)
            for row in range(5):
                if int(mod) in mod_location[row]:
                    marker = markers[row]
                    for col in range(5):
                        if int(mod) == int(mod_location[row][col]):
                            color = colors[col]
        ax[0, 1].plot(temptime_start, temps[mod], label=label, color=color, marker=marker)
    ax[0, 1].set_xlabel('Time', fontsize=24)
    ax[0, 1].set_ylabel('Temperature (C)', fontsize=24)
    ax[0, 1].set_title('Temperature vs Time: All Modules', fontsize=24)
    ax[0, 1].legend(bbox_to_anchor=(1.01,1), loc='upper left', fontsize=16)
    ax[0, 1].tick_params(labelsize=16)
    #spl = [0]+[i for i in range(1,len(time_stamp)) if time_stamp[i]-time_stamp[i-1]>1]+[None]
    spl = [i for i in range(1,len(time_stamp)) if time_stamp[i]-time_stamp[i-1]>1]+[None]
    timestamp_groups = [time_stamp[b:e] for (b, e) in [(spl[i-1],spl[i]) for i in range(1,len(spl))]]
    rates = [len(timestamp_groups[i])/(timestamp_groups[i][-1] - timestamp_groups[i][0]) for i in range(len(timestamp_groups))]
    avg_temps = temps['camera_average']
    avg_temps.pop()
    temptime_start.pop()
    ax[1, 0].plot(rates)
    ax[1, 0].plot(avg_temps)
    ax[1, 0].set_title("Rate and Temperature", fontsize=24)
    ax[1, 0].set_xlabel("Time", fontsize=24)
    ax[1, 0].set_ylabel("Rate/Temp", fontsize=24)
    ax[1, 1].scatter(avg_temps, rates)
    ax[1, 1].set_title("Temperature vs Rate", fontsize=24)
    ax[1, 1].set_xlabel("Temperature (C)", fontsize=24)
    ax[1, 1].set_ylabel("Rate (Hz)", fontsize=24)
    fig.suptitle(f"Temperature and Rate for Run {run}", fontsize=28)
    fig.savefig(f"{SAVEDIR}/temp_rate_run{run}.png")
    plt.clf()


def plot_delta_t_summary(time_s, time_ns, time_stamp, run, SAVEDIR):
    print('Generating delta t histograms...')
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 20))

    # CPU Time
    time_s *= 10**9
    time_stamp_cpu = time_s + time_ns
    time_stamp_cpu -= time_stamp_cpu[0]
    time_stamp_cpu = [time/1000000000 for time in time_stamp_cpu]
    delta_time = [time_stamp_cpu[i+1] - time_stamp_cpu[i] for i in range(len(time_stamp_cpu)-1)]
    print('Plotting CPU timestamp histograms')

    ax[0, 1].hist(delta_time, bins=500, range=(0, 0.01), log=False)
    ax[0, 1].set_title("CPU delta t - linear", fontsize=30)
    ax[0, 1].set_xlabel("delta t (s)", fontsize=20)
    ax[0, 1].set_ylabel("counts", fontsize=30)
    ax[0, 1].tick_params(labelsize=20)

    ax[1, 1].hist(delta_time, bins=500, range=(0, 0.01), log=True)
    ax[1, 1].set_title("CPU delta t - log", fontsize=30)
    ax[1, 1].set_xlabel("delta t (s)", fontsize=20)
    ax[1, 1].set_ylabel("counts", fontsize=30)
    ax[1, 1].tick_params(labelsize=20)

    # TACK Time
    delta_time = [time_stamp[i+1] - time_stamp[i] for i in range(len(time_stamp)-1)]
    print('Plotting TACK timestamp histograms')

    ax[0, 0].hist(delta_time, bins=500, range=(0, 0.01), log=False)
    ax[0, 0].set_title("TACK delta t - linear", fontsize=30)
    ax[0, 0].set_xlabel("delta t (s)", fontsize=20)
    ax[0, 0].set_ylabel("counts", fontsize=30)
    ax[0, 0].tick_params(labelsize=20)

    ax[1, 0].hist(delta_time, bins=500, range=(0, 0.01), log=True)
    ax[1, 0].set_title("TACK delta t - log", fontsize=30)
    ax[1, 0].set_xlabel("delta t (s)", fontsize=20)
    ax[1, 0].set_ylabel("counts", fontsize=30)
    ax[1, 0].tick_params(labelsize=20)

    # Save Figure
    fig.suptitle(f"Delta t histograms for Run {run}", fontsize=28)
    fig.savefig(f"{SAVEDIR}/deltat_hist_run{run}.png")


def plot_rate_by_type(timestamps,
                      dropped_events,
                      flasher_events,
                      noise_events,
                      SAVEDIR,
                      run,
                      username,
                      date,):
    """
    Flasher events should be a list of flasher event indices; noise events
    should be a boolean mask. These will be used as fancy index on timestamp.
    """

    print("Plotting rate by type")
    classes = {}
    for ev in flasher_events:
        classes[ev] = "flasher"
    for i, ev in enumerate(noise_events):
        if ev == True:
            classes[i] = "noise"
    for i, ev in enumerate(dropped_events):
        if ev == True:
            classes[i] = "dropped"
    flasher_events_bool = np.zeros(len(timestamps))
    flasher_events_bool[flasher_events] = 1.
    flasher_events_bool = np.asarray(flasher_events_bool, dtype=bool)
    shower_events = ~(flasher_events_bool | noise_events | dropped_events)  # False in both
    for i, ev in enumerate(shower_events):
        if ev == True:
            classes[i] = "shower"
    flasher_timestamps = list(timestamps[flasher_events])
    dropped_timestamps = list(timestamps[dropped_events])
    noise_timestamps = list(timestamps[noise_events])
    shower_timestamps = list(timestamps[shower_events])
    time_hist, _ = np.histogram(timestamps, bins=int(timestamps[-1]/60), weights=[1./60]*len(timestamps))
    flasher_hist, _ = np.histogram(flasher_timestamps, bins=int(flasher_timestamps[-1]/60), weights=[1./60]*len(flasher_timestamps))
    dropped_hist, _ = np.histogram(dropped_timestamps, bins=int(dropped_timestamps[-1]/60), weights=[1./60]*len(dropped_timestamps))
    shower_hist, _ = np.histogram(shower_timestamps, bins=int(shower_timestamps[-1]/60), weights=[1./60]*len(shower_timestamps))
    noise_hist, _ = np.histogram(noise_timestamps, bins=int(noise_timestamps[-1]/60), weights=[1./60]*len(noise_timestamps))
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.scatter(np.linspace(0, int(timestamps[-1]/60)-1, int(timestamps[-1]/60)) + 0.5, time_hist, label="all")
    ax.scatter(np.linspace(0, int(flasher_timestamps[-1]/60)-1, int(flasher_timestamps[-1]/60)) + 0.5, flasher_hist, label="flasher")
    ax.scatter(np.linspace(0, int(dropped_timestamps[-1]/60)-1, int(dropped_timestamps[-1]/60)) + 0.5, dropped_hist, label="dropped")
    ax.scatter(np.linspace(0, int(shower_timestamps[-1]/60)-1, int(shower_timestamps[-1]/60)) + 0.5, shower_hist, label="shower")
    ax.scatter(np.linspace(0, int(noise_timestamps[-1]/60)-1, int(noise_timestamps[-1]/60)) + 0.5, noise_hist, label="noise")
    ax.set_xlabel("Time (min)", fontsize=22)
    ax.set_ylabel("Rate (Hz)", fontsize=22)
    ax.set_title(f"Run {run}, {date} UTC", fontsize=22)
    ax.legend(fontsize=18)
    ax.tick_params(labelsize=16)
    """
    rates = [1. / (timestamps[i+1] - timestamps[i])
             for i in range(len(timestamps) - 1)]
    rate_times = [(timestamps[i+1] - timestamps[i]) / 2. + timestamps[i]
                  for i in range(len(timestamps) - 1)]
    flasher_rates = [1. / (flasher_timestamps[i+1] - flasher_timestamps[i])
                     for i in range(len(flasher_timestamps)-1)]  # 1 / delta t
    flasher_rate_times = [(flasher_timestamps[i+1] - flasher_timestamps[i]) / 2.
                          + flasher_timestamps[i]
                          for i in range(len(flasher_timestamps) - 1)]  # midpoints
    noise_rates = [1. / (noise_timestamps[i+1] - noise_timestamps[i])
                   for i in range(len(noise_timestamps)-1)]
    noise_rate_times = [(noise_timestamps[i+1] - noise_timestamps[i]) / 2.
                        + noise_timestamps[i]
                        for i in range(len(noise_timestamps)-1)]
    shower_rates = [1. / (shower_timestamps[i+1] - shower_timestamps[i])
                    for i in range(len(shower_timestamps)-1)]
    shower_rate_times = [(shower_timestamps[i+1] - shower_timestamps[i]) / 2.
                         + shower_timestamps[i]
                         for i in range(len(shower_timestamps)-1)]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    ax.plot(rate_times, rates, label="All")
    ax.plot(flasher_rate_times, flasher_rates, label="flashers")
    ax.plot(noise_rate_times, noise_rates, label="noise")
    ax.plot(shower_rate_times, shower_rates, label="shower candidates")
    ax.set_yscale("log")
    ax.set_title("Rate by Event Type Over Time", fontsize=22)
    ax.set_xlabel("Relative Time (s)", fontsize=22)
    ax.set_ylabel("Rate (Hz)", fontsize=22)
    ax.legend(fontsize=18)
    ax.tick_params(labelsize=16)
    """
    fig.savefig(f"{SAVEDIR}/rate_by_type_run{run}.png")
    classes = {k: v for k, v in sorted(classes.items(),
                                       key=lambda item: item[0])}
    store_file = pd.DataFrame([classes.keys(), timestamps, classes.values()]).T
    store_file.rename(columns={0: "Event", 1: "TimeFromStart", 2:"Type"},
                      inplace=True)
    file_path = "/data/local_outputDir" if username == "ctauser" else "/data/wipac/CTA/target5and7data/runs_320000_through_329999"
    store_file.to_hdf(f"{file_path}/run{run}_time_class.h5", key="hdf", mode="w", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", metavar="run_number", help="Indicate the run number of the calibrated data file.")
    parser.add_argument("-p", action="count", help="Flag as a pedestal run for summary.")
    args = parser.parse_args()
    run = args.r
    ped = args.p
    username = pwd.getpwuid(os.getuid()).pw_name
    if username == "ctauser":
        DATADIR = "/data/local_outputDir"
        SAVEDIR = f"/data/analysis_output/quicklook_output/run{run}"
    else:
        DATADIR = "/data/wipac/CTA/target5and7data/spike_tagged_calibrated_data"
        SAVEDIR = f"/data/wipac/CTA/web/analysis_output/summary_plots/run{run}"
    try:
        os.mkdir(SAVEDIR)
    except FileExistsError:
        pass
    if ped == 1:
        reader = Reader(f"{DATADIR}/run{run}.fits")
    else:
        reader = Reader(f"{DATADIR}/cal{run}.r1")
    if ped != 1:
        charge_max = []
        charge_mean = []
        charge_std = []
        charge_uniformity = []
    time_s = []
    time_ns = []
    tack = []
    noise = []
    dropped = []
    reader.get_event(0)
    date = reader.cpu_s
    date = datetime.fromtimestamp(date)
    print(f"Reading data from {DATADIR}")
    for event in tqdm(range(reader.n_events)):
        reader.get_event(event)
        noise.append(reader.is_noise)
        if len(reader.charges[reader.charges == 0.0]) > (1536 / 2) - 100:
            dropped.append(True)
        else:
            dropped.append(False)
        if ped != 1:
            charge_max.append(np.max(reader.charges))
            charge_mean.append(np.mean(reader.charges))
            charge_std.append(np.std(reader.charges, ddof=1))
            charge_uniformity.append(np.mean(reader.charges)/np.std(reader.charges, ddof=1))
        time_s.append(reader.cpu_s)
        time_ns.append(reader.cpu_ns)
        tack.append(reader.tack)

    if ped != 1:
        charge_max = np.asarray(charge_max)
        charge_mean = np.asarray(charge_mean)
        charge_std = np.asarray(charge_std)
        charge_uniformity = np.asarray(charge_uniformity)
    time_s = np.asarray(time_s)
    time_ns = np.asarray(time_ns)
    tack = np.asarray(tack, dtype=float)
    time_stamp = tack.copy()
    time_stamp -= time_stamp[0]
    time_stamp /= 1.0e9
    if ped != 1:
        try:
            plot_charge_summary(time_stamp, charge_max, charge_mean, charge_std,
                                charge_uniformity, SAVEDIR, run)
        except Exception as ex:
            print(ex)
            pass
        try:
            plot_temperature_summary(DATADIR, SAVEDIR, run, time_stamp)
        except Exception as ex:
            print(ex)
            pass
        #try:
        if 1:
            plot_rate_by_type(time_stamp, dropped, reader.get_flasher_events(10., timestamps=time_stamp), noise, SAVEDIR, run, username, date)
        #except Exception as ex:
        #    print(ex)
        #    pass
    try:
        plot_delta_t_summary(time_s, time_ns, time_stamp, run, SAVEDIR)
    except Exception as ex:
        print(ex)
        pass

    # this should be all the information we need directly from the run itself
    # everything else for the plots comes from the temperature file
    # needs to know that this doesn't exist for the pedestal runs

    # should this automatically apply calibration if the calibrated file
    # does not yet exist?

    # Revisiting this now while onsite, 05 Jan 2021
    # Really unsure why there's a pedestal option at all here?
    # We don't do charge plots for a pedestal run, and we don't
    # seem to take temperature data for them either, so what is there to plot?
