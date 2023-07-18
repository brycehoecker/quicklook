import datetime
import pwd
import os
import sys
import time

import argparse
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspeci
import matplotlib.dates as mdates
from numba import jit, njit
from numba.typed import List
import numpy as np
from tqdm import tqdm

import target_driver
import target_io
import target_calib
from apply_gains import apply_gains


class EventAnalyzer():
    """Class that performs the first level of pSCT run analysis. Takes as input all of the waveforms from
       a single event as well as the number of samples in each waveform and the number of pixels. Creates
       data structure that provides access to calculated charge and peak-to-peak."""
    def __init__(self, waveforms, n_samples, n_pixels):
        """Performs basic initialization of parameters and then uses all of the methods in the class by
           default."""
        self.waveforms = waveforms
        self.n_samples = n_samples
        self.n_pixels = n_pixels
        self.charge = np.zeros(self.n_pixels)
        self.peak_to_peak = np.zeros(self.n_pixels)
        self.charge_stats = {'max': 0, 'mean': 0, 'std dev': 0}
        self.peaktopeak_stats = {'max': 0, 'mean': 0, 'std dev': 0}

        # self.remove_INFN_gain()
        self.calculate_charge()
        self.calculate_peak_to_peak()
        self.calculate_stats()

    def remove_INFN_gain(self):
        """Applies a first order correction to INFN modules having roughly twice the gain of US modules."""
        self.waveforms[:576,:] /= 2.0 # the first 576 pixels correspond to the INFN modules

    def calculate_charge(self):
        """Calculates the charge for every pixel by identifying the peak of the waveform and numerically
           integrating over a symmetric range about the peak. Accounts for peaks that are too close to the
           front or back of the waveform."""
        peak_position = np.argmax(self.waveforms, axis=1)
        lower = 8
        upper = 8

        charge_temp = []
        for i in range(len(peak_position)):
            if peak_position[i] < lower:
                charge_temp.append(np.sum(self.waveforms[i, :peak_position[i]+upper]))
            elif peak_position[i] >= self.n_samples - upper:
                charge_temp.append(np.sum(self.waveforms[i, peak_position[i]-lower:]))
            else:
                charge_temp.append(np.sum(self.waveforms[i, peak_position[i]-lower:peak_position[i]+upper]))
        self.charge = np.asarray(charge_temp)
        self.charge = apply_gains(self.charge)

    def calculate_peak_to_peak(self):
        """Calculates the peak-to-peak for every pixel by subtracting the maximum value in the waveform
           from the minimum value. Attempts to account for undershoot by limiting the minimum to be a
           nonnegative value."""
        low_peaks = np.amin(self.waveforms, axis=1)
        correct = []
        for i in range(len(low_peaks)):
            if low_peaks[i] < 0:
                correct.append(0)
            else:
                correct.append(low_peaks[i])
        correct = np.asarray(correct)
        self.peak_to_peak = np.amax(self.waveforms, axis=1) - correct

    def calculate_stats(self):
        """Calculates summary statistics from the charge and peak-to-peak data. These are very useful."""
        self.charge_stats['max'] = np.amax(self.charge)
        self.charge_stats['mean'] = np.mean(self.charge)
        self.charge_stats['std dev'] = np.std(self.charge)

        self.peaktopeak_stats['max'] = np.amax(self.peak_to_peak)
        self.peaktopeak_stats['mean'] = np.mean(self.peak_to_peak)
        self.peaktopeak_stats['std dev'] = np.std(self.peak_to_peak)


class Quicklooker():
    """General purpose utility class that can perform several different basic analysis tasks.
       Functionality began as a simple tool for performing a very basic cut on events to eliminate electronics
       noise triggered events from image generation. Currently, most of the class is dedicated to plotting
       various summary plots and collecting run statistics using the EventAnalyzer class. Recently, the
       capability to calculate Hillas parameters and perform a timing based flasher cut have been added."""
    def __init__(self, pedID, runID, mod_list, datadir, savedir, chPerPacket=32):
        """Establishes important constants and dictionaries needed by other methods and ensures that a
           calibrated data file is available for use."""
        print("Initializing quicklook!")
        self.runID = runID
        self.pedID = pedID
        self.mod_list = mod_list
        self.datadir = datadir
        self.savedir = savedir
        self.chPerPacket = chPerPacket

        try:
            print(f"Trying to create save directory at: {self.savedir}")
            os.mkdir(self.savedir)
        except:
            pass

        self.runfile = f"run{self.runID}.fits"
        self.pedfile = f"ped{self.pedID}.tcal"
        self.calfile = f"cal{self.runID}.r1"

        try:
            if os.path.isfile(f"{self.datadir}/{self.runfile}") is False:
                raise Exception
            else:
                pass
        except:
            sys.exit("Could not find raw data with that ID!\nQuitting now....")

        if os.path.isfile(f"{self.datadir}/{self.pedfile}") is False:
            self._generate_pedestals()
        else:
            print("Pedestal database found!")

        if os.path.isfile(f"{self.datadir}/{self.calfile}") is False:
            self._apply_calibration()
        else:
            print("Calibrated data found!")

        self.calreader = target_io.WaveformArrayReader(f"{self.datadir}/{self.calfile}")
        self.n_pixels = self.calreader.fNPixels
        self.n_samples = self.calreader.fNSamples
        self.n_events = self.calreader.fNEvents
        self.n_asics = 4
        self.n_channels = 16
        self.n_modules = len(self.mod_list)
        self.mod_pos = {4:5, 5:6, 1:7, 3:8, 2:9,
                103:11, 125:12, 126:13, 106:14, 9:15,
                119:17, 108:18, 110:19, 121:20, 8:21,
                115:23, 123:24, 124:25, 112:26, 7:27,
                100:28, 111:29, 114:30, 107:31, 6:32,
                101:14} #101 was formerly in slot 14 before it broke

        self.pos_grid = {5:(1,1), 6:(1,2), 7:(1,3), 8:(1,4), 9:(1,5),
                11:(2,1), 12:(2,2), 13:(2,3), 14:(2,4), 15:(2,5),
                17:(3,1), 18:(3,2), 19:(3,3), 20:(3,4), 21:(3,5),
                23:(4,1), 24:(4,2), 25:(4,3), 26:(4,4), 27:(4,5),
                28:(5,1), 29:(5,2), 30:(5,3), 31:(5,4), 32:(5,5)}

        self.row, self.col = self.row_col_coords(np.arange(64))


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
        self.grid_ind = List()
        for index, mod in enumerate(self.mod_list):
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
                    self.grid_ind.append(int(pix_ind[ch_map[asic * 16 + ch]]))
                #self.waveforms = np.zeros((self.n_pixels, self.n_samples), dtype=np.float32)
                #self.physHeatArr = np.zeros((self.n_modules, 8, 8))

                #self.RunQuicklook()

    def _find_nearest(self, array, value):
        """Simple method that returns the index corresponding to the closest value in an array to the
           provided input."""
        return ((np.abs(np.asarray(array) - value)).argmin())

    def hillas(self, charge_coords):
        """Calculates the Hillas parameters for an event."""
        x = 0
        y = 0
        x2 = 0
        y2 = 0
        xy = 0
        CHARGE = 0
        #print(charge_coords.shape)
        CHARGE = np.sum(charge_coords[2])
        x = np.sum(charge_coords[0] * charge_coords[2])
        y = np.sum(charge_coords[1] * charge_coords[2])
        x2 = np.sum(charge_coords[0] ** 2 * charge_coords[2])
        y2 = np.sum(charge_coords[1] ** 2 * charge_coords[2])
        xy = np.sum(charge_coords[0] * charge_coords[1] * charge_coords[2])

        x /= CHARGE
        y /= CHARGE
        x2 /= CHARGE
        y2 /= CHARGE
        xy /= CHARGE

        S2_x = x2 - x ** 2
        S2_y = y2 - y ** 2
        S_xy = xy - x * y
        d = S2_y - S2_x
        a = (d + np.sqrt(d ** 2 + 4 * S_xy ** 2)) / (2 * S_xy)
        b = y - a * x
        width = np.sqrt((S2_y + a ** 2 * S2_x - 2 * a * S_xy) / (1 + a ** 2))
        length = np.sqrt((S2_x + a ** 2 * S2_y + 2 * a * S_xy) / (1 + a ** 2))
        miss = np.abs(b / np.sqrt(1 + a ** 2))
        dis = np.sqrt(x ** 2 + y ** 2)

        q_coord = (x - charge_coords[0]) * (x / dis) + (y - charge_coords[1]) * (y / dis)
        q = np.sum(q_coord * charge_coords[2]) / CHARGE
        q2 = np.sum(q_coord ** 2 * charge_coords[2]) / CHARGE
        azwidth = q2 - q ** 2

        return [width, length, miss, dis, azwidth]

    def _generate_pedestals(self):
        """Private method that runs the generation of a pedestal database from the command line."""
        os.system(f"generate_ped_SCT -i {self.datadir}/run{self.pedID}.fits -o {self.datadir}/{self.pedfile}")

    def _apply_calibration(self):
        """Private method that applies pedestal subtraction from the command line."""
        os.system(f"apply_calibration_SCT -i {self.datadir}/{self.runfile} -p {self.datadir}/{self.pedfile} -o {self.datadir}/{self.calfile}")

    def row_col_coords(self, index):
        """Private method that performs a calculation relevant for camera image plotting."""
        # Convert bits 1, 3 and 5 to row
        row = 4*((index & 0b100000) > 0) + 2*((index & 0b1000) > 0) + 1*((index & 0b10) > 0)
        # Convert bits 0, 2 and 4 to col
        col = 4*((index & 0b10000) > 0) + 2*((index & 0b100) > 0) + 1*((index & 0b1) > 0)
        return (row, col)

    def calc_loc(self, mod_index):
        """Private method that performs a calculation relevant for camera image plotting."""
        reflectList = [4, 3, 2, 1, 0]
        loc = tuple(np.subtract(self.pos_grid[self.mod_pos[self.mod_list[mod_index]]], (1,1)))
        locReflect = tuple([loc[0], reflectList[loc[1]]])
        return loc, locReflect

    def run_quicklook(self):
        """Public method that applies an elementary cut on electronics noise triggered events and
           generates a run statistics file. Very time consuming and not used often anymore."""
        histfile = h5py.File(f"{self.savedir}/histograms_run{self.runID}.hdf5", 'w')
        histfile.attrs["name"] = f"histograms_run{self.runID}.hdf5"
        histfile.attrs["date"] = str(datetime.datetime.today())
        histfile.attrs["created_by"] = pwd.getpwuid(os.getuid()).pw_name
        histfile.attrs["run"] = self.runID

        with open(f"{savedir}/event_stats_run{runID}.txt", 'w') as statsfile:
            statsfile.write("Event, TimeStamp_s, TimeStamp_ns, Charge_Max, Charge_Mean, Charge_STD, P2P_Max, P2P_Mean, P2PSTD\n")
            for event in tqdm(range(self.n_events)):
                #print(f"Processing event: {event}")
                waveforms = np.zeros((self.n_pixels, self.n_samples), dtype=np.float32)
                physHeatArr = np.zeros((self.n_modules, 8, 8))

                self.calreader.GetR1Event(event, waveforms)
                self.calreader.GetTimeStamp(event)
                timestamp_s = self.calreader.fCPU_s
                timestamp_ns = self.calreader.fCPU_ns
                ev_analysis = EventAnalyzer(waveforms, self.n_samples, self.n_pixels)
                histevent = histfile.create_group(f"Event{event}")
                charge_dataset = histevent.create_dataset("Charge", data=ev_analysis.charge)
                peaktopeak_dataset = histevent.create_dataset("PeakToPeak", data=ev_analysis.peak_to_peak)
                histevent.attrs["event"] = event

                nl = "\n"
                stats_list = []
                stats_list.extend(list(ev_analysis.charge_stats.values()))
                stats_list.extend(list(ev_analysis.peaktopeak_stats.values()))
                statsfile.write(f"{event}, {timestamp_s}, {timestamp_ns}, {stats_list[0]}, {stats_list[1]}, {stats_list[2]}, {stats_list[3]}, {stats_list[4]}, {stats_list[5]}{nl}")

                maxZ = np.amax(ev_analysis.charge)
                charge = np.reshape(ev_analysis.charge, (self.n_modules, -1))
                physHeatArr[:, self.row, self.col] = charge
                if stats_list[1] > 2000:
                    heatReflectFig = plt.figure('Heat Map Skyview', (18, 15))
                    gs = gridspec.GridSpec(5, 5)
                    gs.update(wspace=0.04, hspace=0.04)

                    for mod_index in range(self.n_modules):
                        loc, locReflect = self.calc_loc(mod_index)
                        if loc[1] % 2 == 0:
                            physHeatArr[mod_index, :, :] = np.rot90(physHeatArr[mod_index, :, :], k=2)

                        plt.figure('Heat Map Skyview')
                        ax = plt.subplot(gs[locReflect])
                        c = ax.pcolor(physHeatArr[mod_index, :, ::-1], vmin=0, vmax=maxZ)
                        ax.axis('off')
                        ax.set_aspect('equal')

                    heatReflectFig.subplots_adjust(right=0.8, top=0.9, bottom=0.1)
                    cbar_ax = heatReflectFig.add_axes([0.85, 0.15, 0.05, 0.7])
                    cbar = heatReflectFig.colorbar(c, cax=cbar_ax)
                    cbar.set_label('Charge (Photoelectrons)', rotation=270, size=20, labelpad=24)
                    cbar_ax.tick_params(labelsize=16)
                    heatReflectFig.suptitle(f"Run {self.runID} - Event {event}")
                    heatReflectFig.savefig(f"{self.savedir}/run{self.runID}_ev{event}_charge_calibrated_skyheatmap.png")
                    plt.clf()
            print(f"Event images saved to: {self.savedir}")
        histfile.close()

    def run_eventanalyzer(self, n_events):
        """Public method that generates run statistics. Relatively quick."""
        histfile = h5py.File(f"{self.savedir}/histograms_run{self.runID}.hdf5", 'w')
        histfile.attrs["name"] = f"histograms_run{self.runID}.hdf5"
        histfile.attrs["date"] = str(datetime.datetime.today())
        histfile.attrs["created_by"] = pwd.getpwuid(os.getuid()).pw_name
        histfile.attrs["run"] = self.runID

        with open(f"{self.savedir}/event_stats_run{runID}_{n_events}.txt", 'w') as statsfile:
            statsfile.write("Event, TimeStamp_s, TimeStamp_ns, TACK_time_ns, Charge_Max, Charge_Mean, Charge_STD, P2P_Max, P2P_Mean, P2PSTD, Width, Length, Miss, Dis, Azwidth\n")
            for event in tqdm(range(n_events)):
            #for event in tqdm(range(5000)):
                #print(f"Processing event: {event}")
                waveforms = np.zeros((self.n_pixels, self.n_samples), dtype=np.float32)
                self.calreader.GetR1Event(event, waveforms)
                self.calreader.GetTimeStamp(event)
                timestamp_s = self.calreader.fCPU_s
                timestamp_ns = self.calreader.fCPU_ns
                tack_time = self.calreader.fTACK_time
                ev_analysis = EventAnalyzer(waveforms, self.n_samples, self.n_pixels)
                charge_coords = [[self.grid_ind[i] % 40 - 20.5, self.grid_ind[i] // 40 - 20.5, ev_analysis.charge[i]]
                                 for i in range(len(ev_analysis.charge))]
                hillas_params = self.hillas(np.asarray(charge_coords).T)
                histevent = histfile.create_group(f"Event{event}")
                charge_dataset = histevent.create_dataset("Charge", data=ev_analysis.charge)
                peaktopeak_dataset = histevent.create_dataset("PeakToPeak", data=ev_analysis.peak_to_peak)
                histevent.attrs["event"] = event

                nl = "\n"
                stats_list = []
                stats_list.extend(list(ev_analysis.charge_stats.values()))
                stats_list.extend(list(ev_analysis.peaktopeak_stats.values()))
                statsfile.write(f"{event}, {timestamp_s}, {timestamp_ns}, {tack_time}, {stats_list[0]}, {stats_list[1]}, {stats_list[2]}, {stats_list[3]}, {stats_list[4]}, {stats_list[5]}, {hillas_params[0]}, {hillas_params[1]}, {hillas_params[2]}, {hillas_params[3]}, {hillas_params[4]}{nl}")
        histfile.close()

    def make_run_hists(self, n_events):
        """Should follow run_eventanalyzer. Uses event statistics data file to create plots of charge
           and peak-to-peak."""
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 20))
        try:
            datafile = f"{self.savedir}/event_stats_run{self.runID}_{n_events}.txt"
            events, time_s, time_ns, charge_max, \
            charge_mean, charge_std, p2p_max, \
            p2p_mean, p2p_std, width, length, miss, dis, azwidth = np.loadtxt(datafile, delimiter=", ", skiprows=1, unpack=True)
        except:
            datafile = f"{self.savedir}/event_stats_run{self.runID}.txt"
            events, time_s, time_ns, charge_max, \
            charge_mean, charge_std, p2p_max, \
            p2p_mean, p2p_std, width, length, miss, dis, azwidth = np.loadtxt(datafile, delimiter=", ", skiprows=1, unpack=True)

        events = events.astype("int")
        time_s *= 10**9
        time_stamp = time_s + time_ns
        time_stamp -= time_stamp[0]

        ax[0, 0].scatter(time_stamp, charge_max)
        ax[0, 0].set_title("Maximum Charge per Pixel in Event", fontsize=24)
        ax[0, 0].set_xlabel("Time from First Event (ns)", fontsize=24)
        ax[0, 0].set_ylabel("Charge (Photoelectrons)", fontsize=24)

        ax[0, 1].scatter(time_stamp, charge_mean)
        ax[0, 1].set_title("Mean Charge in Event", fontsize=24)
        ax[0, 1].set_xlabel("Time from First Event (ns)", fontsize=24)
        ax[0, 1].set_ylabel("Charge (Photoelectrons)", fontsize=24)

        ax[1, 0].scatter(time_stamp, charge_std)
        ax[1, 0].set_title("Standard Deviation of Charge in Event", fontsize=24)
        ax[1, 0].set_xlabel("Time from First Event (ns)", fontsize=24)
        ax[1, 0].set_ylabel("Charge (Photoelectrons)", fontsize=24)

        ax[1, 1].scatter(time_stamp, charge_mean / charge_std)
        ax[1, 1].set_title("Charge Uniformity in Event", fontsize=24)
        ax[1, 1].set_xlabel("Time from First Event (ns)", fontsize=24)
        ax[1, 1].set_ylabel("Charge (Photoelectrons)", fontsize=24)

        fig.suptitle(f"Charge Statistics in Run{self.runID}", fontsize=28)
        if n_events is not None:
            fig.savefig(f"{self.savedir}/charge_stats_run{self.runID}_{n_events}.png")
        else:
            fig.savefig(f"{self.savedir}/charge_stats_run{self.runID}.png")
        plt.clf()

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 20))

        ax[0, 0].scatter(time_stamp, p2p_max)
        ax[0, 0].set_title("Maximum Peak-To-Peak in Event", fontsize=24)
        ax[0, 0].set_xlabel("Time from First Event (ns)", fontsize=24)
        ax[0, 0].set_ylabel("Peak-To-Peak (ADC)", fontsize=24)

        ax[0, 1].scatter(time_stamp, p2p_mean)
        ax[0, 1].set_title("Mean Peak-To-Peak in Event", fontsize=24)
        ax[0, 1].set_xlabel("Time from First Event (ns)", fontsize=24)
        ax[0, 1].set_ylabel("Peak-To-Peak (ADC)", fontsize=24)

        ax[1, 0].scatter(time_stamp, p2p_std)
        ax[1, 0].set_title("Standard Deviation of Peak-To-Peak in Event", fontsize=24)
        ax[1, 0].set_xlabel("Time from First Event (ns)", fontsize=24)
        ax[1, 0].set_ylabel("Peak-To-Peak (ADC)", fontsize=24)

        ax[1, 1].scatter(time_stamp, p2p_mean / p2p_std)
        ax[1, 1].set_title("Peak-To-Peak Uniformity in Event", fontsize=24)
        ax[1, 1].set_xlabel("Time from First Event (ns)", fontsize=24)
        ax[1, 1].set_ylabel("Peak-To-Peak (ADC)", fontsize=24)

        fig.suptitle(f"Peak-To-Peak Statistics in Run{self.runID}", fontsize=28)
        if n_events is not None:
            fig.savefig(f"{self.savedir}/p2p_stats_run{self.runID}_{n_events}.png")
        else:
            fig.savefig(f"{self.savedir}/p2p_stats_run{self.runID}.png")

        plt.clf()

    def make_temp_hists(self, n_events=None):
        print('Generating temperature histograms...')
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 20))
        print(n_events)

        '''
        try:
            datafile = f"{self.savedir}/event_stats_run{self.runID}_{n_events}.txt"
            events, time_s, time_ns, charge_max, \
            charge_mean, charge_std, p2p_max, \
            p2p_mean, p2p_std, width, length, miss, dis, azwidth = np.loadtxt(datafile, delimiter=", ", skiprows=1, unpack=True)
        except:
            datafile = f"{self.savedir}/event_stats_run{self.runID}.txt"
            events, time_s, time_ns, charge_max, \
            charge_mean, charge_std, p2p_max, \
            p2p_mean, p2p_std, width, length, miss, dis, azwidth = np.loadtxt(datafile, delimiter=", ", skiprows=1, unpack=True)
        '''

        datafile = f"{self.savedir}/event_stats_run{self.runID}_{n_events}.txt"
        events, time_s, time_ns, TACK_ns, charge_max, \
        charge_mean, charge_std, p2p_max, \
        p2p_mean, p2p_std, width, length, miss, dis, azwidth = np.loadtxt(datafile, delimiter=", ", skiprows=1, unpack=True)
        #"Event, TimeStamp_s, TimeStamp_ns, TACK_time_ns, Charge_Max, Charge_Mean, Charge_STD, P2P_Max, P2P_Mean, P2PSTD, Width, Length, Miss, Dis, Azwidth\n")
 
        events = events.astype("int")
        time_s *= 10**9
        time_stamp = time_s + time_ns
        time_stamp -= time_stamp[0]
        time_stamp = [time/1000000000 for time in time_stamp]

        print('Plotting timestamp histogram')
        ax[0, 0].hist(time_stamp, bins=int(time_stamp[-1]))
        ax[0, 0].set_title("Histogram of Timestamps", fontsize=24)
        ax[0, 0].set_xlabel("Time from First Event (s)", fontsize=24)
        ax[0, 0].set_ylabel("Rate (Hz)", fontsize=24)
        ax[0, 0].tick_params(labelsize=16)


        tempfile = f"{self.datadir}/{self.runID}_temperatures.txt"
        self.tempdata = pd.read_csv(tempfile)
        shape = self.tempdata.shape
        temptime_start = [self.tempdata['timestamp_start'][i] for i in range(shape[0])]
        temptime_start = mdates.num2date(mdates.datestr2num(temptime_start))
        date = datetime.date.fromisoformat(self.tempdata['timestamp_start'][1][:10])
        module_numbers = quicklook.list_modules()
        temps = quicklook.avg_temps()
        print('Number of Temp/Current measurements: {}'.format(len(temps)))

        print(len(temptime_start))

        mod_location = [[4,5,1,3,2],[103,125,126,106,9],[119,108,110,121,8],[115,123,124,112,7],[100,111,114,107,6]]
        cmap = matplotlib.cm.get_cmap('rainbow')
        colors = [cmap(0),cmap(0.25),cmap(0.5),cmap(0.75),cmap(0.99)]
        markers = ['o', 's', 'v', 'd', 'X']
        print('Plotting temp vs time')
        for mod in sorted(temps.keys()):
            if mod == 'camera_average':
                marker=None
                color='black'
                label='Module Avg'
            else:
                label = 'Module ' + str(mod)
                for row in range(5):
                    if int(mod) in mod_location[row]:
                        marker = markers[row]
                        for col in range(5):
                            if int(mod) ==  int(mod_location[row][col]):
                                color = colors[col]
            ax[0, 1].plot(temptime_start, temps[mod], label=label, color=color, marker=marker)
        #fig.autofmt_xdate()
        ax[0, 1].set_xlabel('Time', fontsize=24)
        ax[0, 1].set_ylabel('Temperature (C)', fontsize=24)
        ax[0, 1].set_title('Temperature vs Time: All Modules', fontsize=24)
        ax[0, 1].legend(bbox_to_anchor=(1.01,1), loc='upper left', fontsize=16)
        ax[0, 1].tick_params(labelsize=16)

        spl = [0]+[i for i in range(1,len(time_stamp)) if time_stamp[i]-time_stamp[i-1]>1]+[None]
        timestamp_groups = [time_stamp[b:e] for (b, e) in [(spl[i-1],spl[i]) for i in range(1,len(spl))]]
        print('Number of Subruns')
        print(len(timestamp_groups))
        rates = [len(timestamp_groups[i])/(timestamp_groups[i][-1] - timestamp_groups[i][0]) for i in range(len(timestamp_groups))]
        avg_temps = temps['camera_average']
        avg_temps.pop()

        temptime_start.pop()
        print('Plotting rate and temp plot')
        #ax[1, 0].plot(temptime_start, rates)
        #ax[1, 0].plot(temptime_start, avg_temps)
        ax[1, 0].plot(rates)
        ax[1, 0].plot(avg_temps)
        ax[1, 0].set_title("Rate and Temperature", fontsize=24)
        ax[1, 0].set_xlabel("Time", fontsize=24)
        ax[1, 0].set_ylabel("Rate/Temp", fontsize=24)

        '''
        print('Plotting temp vs rate')
        ax[1, 1].scatter(avg_temps, rates)
        ax[1, 1].set_title("Temperature vs Rate", fontsize=24)
        ax[1, 1].set_xlabel("Temperature (C)", fontsize=24)
        ax[1, 1].set_ylabel("Rate (Hz)", fontsize=24)
        '''
        fig.suptitle(f"Temperature and Rate for Run {self.runID}", fontsize=28)
        fig.savefig(f"{self.savedir}/temp_rate_run{self.runID}.png")
        if n_events is not None:
            fig.savefig(f"{self.savedir}/temp_rate_run{self.runID}_{n_events}.png")
        else:
            fig.savefig(f"{self.savedir}/temp_rate_run{self.runID}.png")

        plt.clf()

    def make_deltat_hist(self, n_events=None):
        print('Generating delta t histograms...')

        
        try:
            datafile = f"{self.savedir}/event_stats_run{self.runID}_{n_events}.txt"
            events, time_s, time_ns, TACK_ns, charge_max, \
            charge_mean, charge_std, p2p_max, \
            p2p_mean, p2p_std, width, length, miss, dis, azwidth = np.loadtxt(datafile, delimiter=", ", skiprows=1, unpack=True)
        except:
            datafile = f"{self.savedir}/event_stats_run{self.runID}.txt"
            events, time_s, time_ns, TACK_ns, charge_max, \
            charge_mean, charge_std, p2p_max, \
            p2p_mean, p2p_std, width, length, miss, dis, azwidth = np.loadtxt(datafile, delimiter=", ", skiprows=1, unpack=True)
        
        # Create Figure
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(25, 20))
        
        # CPU Time
        events = events.astype("int")
        time_s *= 10**9
        time_stamp = time_s + time_ns
        time_stamp -= time_stamp[0]
        time_stamp = [time/1000000000 for time in time_stamp]
        delta_time = [time_stamp[i+1] - time_stamp[i]  for i in range(len(time_stamp)-1)]
        print('Plotting CPU timestamp histograms')
               
        ax[0, 1].hist(delta_time, bins=500, range=(0,0.2), log=False)
        ax[0, 1].set_title("CPU delta t - linear", fontsize=30)
        ax[0, 1].set_xlabel("delta t (s)", fontsize=20)
        ax[0, 1].set_ylabel("counts", fontsize=20)
        ax[0, 1].tick_params(labelsize=20)

        ax[1, 1].hist(delta_time, bins=500, range=(0,0.2), log=True)
        ax[1, 1].set_title("CPU delta t - log", fontsize=30)
        ax[1, 1].set_xlabel("delta t (s)", fontsize=20)
        ax[1, 1].set_ylabel("counts", fontsize=20)
        ax[1, 1].tick_params(labelsize=20)

        # TACK Time
        time_stamp = TACK_ns
        time_stamp -= TACK_ns[0]
        time_stamp = [time/1000000000 for time in time_stamp]
        delta_time = [time_stamp[i+1] - time_stamp[i]  for i in range(len(time_stamp)-1)]
        print('Plotting TACK timestamp histograms')
        #zero_times = [time for time in delta_time if time<0.0001]
        #print(zero_times)
        #print(len(delta_time))
        #print(len(zero_times))

        ax[0, 0].hist(delta_time, bins=500, range=(0,0.2), log=False)
        ax[0, 0].set_title("TACK delta t - linear", fontsize=30)
        ax[0, 0].set_xlabel("delta t (s)", fontsize=20)
        ax[0, 0].set_ylabel("counts", fontsize=20)
        ax[0, 0].tick_params(labelsize=20)

        ax[1, 0].hist(delta_time, bins=500, range=(0,0.2), log=True)
        ax[1, 0].set_title("TACK delta t - log", fontsize=30)
        ax[1, 0].set_xlabel("delta t (s)", fontsize=20)
        ax[1, 0].set_ylabel("counts", fontsize=20)
        ax[1, 0].tick_params(labelsize=20)

        # Save Figure
        fig.suptitle(f"Delta t histograms for Run {self.runID}", fontsize=28)
        if n_events is not None:
            fig.savefig(f"{self.savedir}/deltat_hist_run{self.runID}_{n_events}.png")
        else:
            fig.savefig(f"{self.savedir}/deltat_hist_run{self.runID}.png")

        plt.clf()
        

        # New figure to track where dt=0
        zero_events = [i for (i,val) in enumerate(delta_time) if val==0.0]
        zero_times = [time_stamp[i] for i in zero_events]
        plt.hist(zero_times, bins=180, range=(0,max(time_stamp)))
        plt.title("Histogram of dt=0 event pairs by time", fontsize=50)
        plt.xlabel("time (s)", fontsize=30)
        plt.ylabel("rate (Hz)", fontsize=30)
        plt.tick_params(labelsize=30)
        plt.savefig(f"{self.savedir}/zero_events_hist_run{self.runID}.png")
        plt.clf()

        max_events = [i for (i,val) in enumerate(delta_time) if val>0.0062 if val<0.0064]
        print(max_events)

    def list_modules(self):
        module_numbers = set()
        for sensor in self.tempdata.keys():
            if sensor == 'timestamp_start':
                continue
            if sensor == 'timestamp_end':
                continue
            module_numbers.add(sensor.split('_')[0])
        self.module_numbers = module_numbers
        return module_numbers

    def avg_temps(self):
        avg_temp_per_module = {}
        avg_temps = []
        for mod in self.module_numbers:
            individual_sensor_values = []
            for i in range(4):
                values = self.tempdata[str(mod)+'_ADC'+str(i)]
                if all(val<100 for val in values):
                    if all(val>4 for val in values):
                        individual_sensor_values.append(values)
            average_sensor_values = [sum(col) / float(len(col)) for col in zip(*individual_sensor_values)]
            if average_sensor_values != []:
                avg_temp_per_module[mod] = average_sensor_values
                avg_temps.append(average_sensor_values)
        camera_average = [float(sum(col))/len(col) for col in zip(*avg_temps)]
        avg_temp_per_module['camera_average'] = camera_average
        return avg_temp_per_module

    def make_flasher_images(self):
        """Experimental. Do not use unless confident."""
        print("Starting to make flasher images.")
        datafile = f"{self.savedir}/event_stats_run{self.runID}.txt"
        self.flasherdir = f"{self.savedir}/flasher_images"
        try:
            os.mkdir(f"{self.flasherdir}")
        except:
            pass
        events, time_s, time_ns, charge_max, \
        charge_mean, charge_std, p2p_max, \
        p2p_mean, p2p_std, width, length, miss, dis, azwidth = np.loadtxt(datafile, delimiter=", ", skiprows=1, unpack=True)

        print("Got run statistics.")

        events = events.astype("int")
        time_s *= 10**9
        time_stamp = time_s + time_ns
        time_stamp -= time_stamp[0]


        #print("Performing initial cut.")
        #cut = [event for event in events if charge_mean[event] > 520]
        print("Performing flasher cut.")
        #flasher = [event for event in cut if charge_std[event] < (lambda x: 16/7 * x - 20000/7)(charge_mean[event])]
        time_stamp_delta_ts = [j - i for i, j in zip(time_stamp[:-1], time_stamp[1:])]
        flasher_delta_ts = [i for i in time_stamp_delta_ts if i > 1.0001e8 and i < 1.003e8]
        mean = np.mean(flasher_delta_ts)
        std = np.std(flasher_delta_ts)

        flasher_ev = self._find_nearest(time_stamp_delta_ts, mean)
        time_pos = time_stamp[flasher_ev]
        time_shift = list(np.asarray(time_stamp) - np.rint(time_pos) - mean / 2.0)
        time_res = [(int(np.rint(i)) % int(np.rint(mean))) - mean / 2.0 for i in time_shift]
        flasher = [i for i, res in enumerate(time_res) if np.abs(res) < 200 * std and charge_mean[i] > 520]

        for event in tqdm(flasher):
            #print(f"Processing event: {event}")
            waveforms = np.zeros((self.n_pixels, self.n_samples), dtype=np.float32)
            physHeatArr = np.zeros((self.n_modules, 8, 8))

            self.calreader.GetR1Event(int(event), waveforms)
            self.calreader.GetTimeStamp(int(event))
            ev_analysis = EventAnalyzer(waveforms, self.n_samples, self.n_pixels)
            maxZ = np.amax(ev_analysis.charge)
            charge = np.reshape(ev_analysis.charge, (self.n_modules, -1))
            physHeatArr[:, self.row, self.col] = charge
            heatReflectFig = plt.figure('Heat Map Skyview', (18, 15))
            gs = gridspec.GridSpec(5, 5)
            gs.update(wspace=0.04, hspace=0.04)

            for mod_index in range(self.n_modules):
                loc, locReflect = self.calc_loc(mod_index)
                if loc[1] % 2 == 0:
                    physHeatArr[mod_index, :, :] = np.rot90(physHeatArr[mod_index, :, :], k=2)

                plt.figure('Heat Map Skyview')
                ax = plt.subplot(gs[locReflect])
                c = ax.pcolor(physHeatArr[mod_index, :, ::-1], vmin=0, vmax=maxZ)
                ax.axis('off')
                ax.set_aspect('equal')

            heatReflectFig.subplots_adjust(right=0.8, top=0.9, bottom=0.1)
            cbar_ax = heatReflectFig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = heatReflectFig.colorbar(c, cax=cbar_ax)
            cbar.set_label('Charge (ADC ns)', rotation=270, size=20, labelpad=24)
            cbar_ax.tick_params(labelsize=16)
            heatReflectFig.suptitle(f"Run {self.runID} - Event {event}")
            heatReflectFig.savefig(f"{self.flasherdir}/run{self.runID}_ev{event}_charge_calibrated_skyheatmap_flasher.png")
            plt.clf()
        print(f"Event images saved to: {self.flasherdir}")

    def flasher_cuts(self, rate):
        print("Performing flasher cut.")
        datafile = f"{self.savedir}/event_stats_run{self.runID}.txt"

        try:
            events, time_s, time_ns, charge_max, charge_mean, \
            charge_std, p2p_max, p2p_mean, p2p_std, width, \
            length, miss, dis, azwidth = np.loadtxt(datafile, delimiter=", ", skiprows=1, unpack=True)
        except:
            print("Run statistics data has not been created or does not include all necessary parameters.")
            raise Exception

        events = events.astype("int")
        time_s *= 10**9
        time_stamp = time_s + time_ns
        time_stamp -= time_stamp[0]

        time_stamp_delta_ts = [j - i for i, j in zip(time_stamp[:-1], time_stamp[1:])]
        rate_use = (1 / rate) * 10. ** 9

        subrun_inds = [ind + 1 for ind, value in enumerate(time_stamp_delta_ts) if value > 1.0e9]
        subrun_inds.insert(0, 0)
        subrun_inds.append(len(events))
        print(subrun_inds)
        subrun_events = []
        for i in range(len(subrun_inds)-1):
            if i < len(subrun_inds) - 2:
                subrun_events.append(np.arange(subrun_inds[i], subrun_inds[i+1]))
            if i == len(subrun_inds) - 2:
                subrun_events.append(np.arange(subrun_inds[i], subrun_inds[i+1] + 1))

        subruns = []
        subrun_timestamps = []
        for i in range(len(subrun_inds)-1):
            subruns.append(time_stamp_delta_ts[subrun_inds[i]:subrun_inds[i+1]])
            subrun_timestamps.append(time_stamp[subrun_inds[i]:subrun_inds[i+1]])
        subruns = np.asarray(subruns)
        subrun_timestamps = np.asarray(subrun_timestamps)

        time_res = []
        std = []
        for i in range(len(subruns)):
            result = self._subrun_flashers(subruns[i], subrun_timestamps[i], rate_use)
            if result[0] is not None:
                time_res.append(result[0])
            if result[1] is not None:
                std.append(result[1])

        time_res = np.asarray(time_res)
        flashers = []
        for i in range(len(time_res)):
            data = time_res[i]
            sigma = std[i]
            flashers.extend([subrun_events[i][ev] for ev, value in enumerate(data) if np.abs(value) < 80 * sigma])

        return flashers

    def _subrun_flashers(self, subrun, time_stamp, rate_use):
        try:
            flasher_delta_ts = [i for i in subrun if i > rate_use + 0.0001e8 and i < rate_use + 0.003e8]
            if len(flasher_delta_ts) == 0:
                raise Exception
            mean = np.mean(flasher_delta_ts)
            std = np.std(flasher_delta_ts)
            #print(std)
            flasher_ev = self._find_nearest(subrun, mean)
            time_pos = time_stamp[flasher_ev]
            time_shift = list(np.asarray(time_stamp) - np.rint(time_pos) - mean / 2.0)
            time_res = [(int(np.rint(i)) % int(np.rint(mean))) - mean / 2.0 for i in time_shift]
            return (time_res, std)
        except Exception:
            #print("No flasher events found in subrun.")
            return ([0 for _ in range(len(subrun))], 0)

    def make_shower_images(self):
        print("Starting to make shower images.")
        datafile = f"{self.savedir}/event_stats_run{self.runID}.txt"
        self.showerdir = f"{self.savedir}/shower_images"
        try:
            os.mkdir(f"{self.showerdir}")
        except:
            pass
        events, time_s, time_ns, charge_max, \
        charge_mean, charge_std, p2p_max, \
        p2p_mean, p2p_std, width, length, miss, dis, azwidth = np.loadtxt(datafile, delimiter=", ", skiprows=1, unpack=True)

        print("Got run statistics.")

        events = events.astype("int")
        time_s *= 10**9
        time_stamp = time_s + time_ns
        time_stamp -= time_stamp[0]


        #print("Performing initial cut.")
        #cut = [event for event in events if charge_mean[event] > 520]
        print("Performing flasher cut.")
        #flasher = [event for event in cut if charge_std[event] < (lambda x: 16/7 * x - 20000/7)(charge_mean[event])]
        time_stamp_delta_ts = [j - i for i, j in zip(time_stamp[:-1], time_stamp[1:])]
        flasher_delta_ts = [i for i in time_stamp_delta_ts if i > 0.999e8 and i < 1.003e8]
        mean = np.mean(flasher_delta_ts)
        std = np.std(flasher_delta_ts)

        flasher_ev = self._find_nearest(time_stamp_delta_ts, mean)
        time_pos = time_stamp[flasher_ev]
        time_shift = list(np.asarray(time_stamp) - np.rint(time_pos) - mean / 2.0)
        time_res = [(int(np.rint(i)) % int(np.rint(mean))) - mean / 2.0 for i in time_shift]
        flasher = [i for i, res in enumerate(time_res) if np.abs(res) < 200 * std]
        flasher_cut = [i for i in flasher if charge_mean[i] > 700.]
        print("Performing shower cut.")
        showers = [i for i in range(len(time_stamp)) if charge_mean[i] > 700 and i not in flasher_cut]
        print("Creating camera images....")
        for event in tqdm(showers):
            #print(f"Processing event: {event}")
            waveforms = np.zeros((self.n_pixels, self.n_samples), dtype=np.float32)
            physHeatArr = np.zeros((self.n_modules, 8, 8))

            self.calreader.GetR1Event(int(event), waveforms)
            self.calreader.GetTimeStamp(int(event))
            ev_analysis = EventAnalyzer(waveforms, self.n_samples, self.n_pixels)
            maxZ = np.amax(ev_analysis.charge)
            charge = np.reshape(ev_analysis.charge, (self.n_modules, -1))
            physHeatArr[:, self.row, self.col] = charge
            heatReflectFig = plt.figure('Heat Map Skyview', (18, 15))
            gs = gridspec.GridSpec(5, 5)
            gs.update(wspace=0.04, hspace=0.04)

            for mod_index in range(self.n_modules):
                loc, locReflect = self.calc_loc(mod_index)
                if loc[1] % 2 == 0:
                    physHeatArr[mod_index, :, :] = np.rot90(physHeatArr[mod_index, :, :], k=2)

                plt.figure('Heat Map Skyview')
                ax = plt.subplot(gs[locReflect])
                c = ax.pcolor(physHeatArr[mod_index, :, ::-1], vmin=0, vmax=maxZ)
                ax.axis('off')
                ax.set_aspect('equal')

            heatReflectFig.subplots_adjust(right=0.8, top=0.9, bottom=0.1)
            cbar_ax = heatReflectFig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = heatReflectFig.colorbar(c, cax=cbar_ax)
            cbar.set_label('Charge (ADC ns)', rotation=270, size=20, labelpad=24)
            cbar_ax.tick_params(labelsize=16)
            heatReflectFig.suptitle(f"Run {self.runID} - Event {event}")
            heatReflectFig.savefig(f"{self.showerdir}/run{self.runID}_ev{event}_charge_calibrated_skyheatmap_shower.png")
            plt.clf()
        print(f"Event images saved to: {self.showerdir}")

    def create_image(self, event):
        """Generate a camera image from a single event and place it in the local folder. Useful for debugging."""
        waveforms = np.zeros((self.n_pixels, self.n_samples), dtype=np.float32)
        physHeatArr = np.zeros((self.n_modules, 8, 8))

        self.calreader.GetR1Event(int(event), waveforms)
        self.calreader.GetTimeStamp(int(event))
        ev_analysis = EventAnalyzer(waveforms, self.n_samples, self.n_pixels)
        maxZ = np.amax(ev_analysis.charge)
        charge = np.reshape(ev_analysis.charge, (self.n_modules, -1))
        physHeatArr[:, self.row, self.col] = charge
        heatReflectFig = plt.figure('Heat Map Skyview', (18, 15))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.04, hspace=0.04)

        for mod_index in range(self.n_modules):
            loc, locReflect = self.calc_loc(mod_index)
            if loc[1] % 2 == 0:
                physHeatArr[mod_index, :, :] = np.rot90(physHeatArr[mod_index, :, :], k=2)

            plt.figure('Heat Map Skyview')
            ax = plt.subplot(gs[locReflect])
            c = ax.pcolor(physHeatArr[mod_index, :, ::-1], vmin=0, vmax=maxZ)
            ax.axis('off')
            ax.set_aspect('equal')

        heatReflectFig.subplots_adjust(right=0.8, top=0.9, bottom=0.1)
        cbar_ax = heatReflectFig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = heatReflectFig.colorbar(c, cax=cbar_ax)
        cbar.set_label('Charge (ADC ns)', rotation=270, size=20, labelpad=24)
        cbar_ax.tick_params(labelsize=16)
        heatReflectFig.suptitle(f"Run {self.runID} - Event {event}")
        heatReflectFig.savefig(f"run{self.runID}_ev{event}_camera_image.png")
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="run_data", help="Indicate the run number of the raw data file.")
    parser.add_argument("-p", metavar="pedestal_data", help="Indicate the run number of the pedestal file.")
    parser.add_argument("-r", action="count", help="Use to create camera images using the current arbitrary cut method, as well as event statistics.")
    parser.add_argument("-e", action="count", help="Use to only generate event statistics and summary plots.")
    parser.add_argument("-n", metavar="n_events", help="Use with -e to only analyze the first n events.")
    parser.add_argument("-f", action="count", help="Use to plot flasher images. For analysis debugging only. Don't use unless event statistics file already exists.")
    parser.add_argument("-s", action="count", help="Use to plot shower images. For analysis debugging only. Don't use unless event statistics file already exists.")
    parser.add_argument("-t", action="count", help="Use to plot temp and rate. Don't use unless event statistics file already exists.")
    parser.add_argument("-d", action="count", help="Use to plot delta t histograms for a run")

    args = parser.parse_args()
    runID = args.i
    pedID = args.p
    r = args.r
    e = args.e
    n = args.n
    f = args.f
    s = args.s
    t = args.t
    d = args.d
    print(f"runID: {runID}")
    print(f"pedID: {pedID}")

    modlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]

    #datadir = "/data/wipac/CTA/target5and7data/runs_320000_through_329999"
    datadir = "/data/local_outputDir/"
    #savedir = f"{datadir}/run{runID}"
    savedir = f"/data/analysis_output/quicklook_output/run{runID}"

    try:
        if r == 1 and e is None and f is None and s is None:
            quicklook = Quicklooker(pedID, runID, modlist, datadir, savedir)
            quicklook.run_quicklook()
            quicklook.make_run_hists()
            quicklook.make_temp_hists()
        elif e == 1 and r is None and f is None and s is None:
            quicklook = Quicklooker(pedID, runID, modlist, datadir, savedir)
            if n is None:
                quicklook.run_eventanalyzer(quicklook.n_events)
                quicklook.make_run_hists(quicklook.n_events)
            else:
                quicklook.run_eventanalyzer(int(n))
                quicklook.make_run_hists(int(n))
            quicklook.make_temp_hists()
        elif f == 1 and r is None and e is None and s is None:
            quicklook = Quicklooker(pedID, runID, modlist, datadir, savedir)
            quicklook.make_flasher_images()
        elif s == 1 and r is None and e is None and f is None:
            quicklook = Quicklooker(pedID, runID, modlist, datadir, savedir)
            quicklook.make_shower_images()
        elif t ==1 and r is None and e is None and f is None and s is None:
            quicklook = Quicklooker(pedID, runID, modlist, datadir, savedir)
            quicklook.make_temp_hists(n)
        elif d == 1 and r is None and e is None and f is None and s is None and t is None:
            quicklook = Quicklooker(pedID, runID, modlist, datadir, savedir)
            quicklook.make_deltat_hist(n)
        elif [r, e, f, s, t, d].count(1) > 1:
            raise Exception
        elif r is None and e is None and f is None and s is None:
            raise Exception
        else:
            raise Exception
    except Exception as ex:
        sys.exit(ex)
