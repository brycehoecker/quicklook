import csv
import os
import sys

from datetime import datetime
import numpy as np
import matplotlib
import matplotlib.lines as lines
from matplotlib.patches import Circle, Ellipse
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm
from numba import njit
from numba.typed import List

import target_io
#from apply_gains import apply_gains

matplotlib.use('Agg')


@njit
def calculate_charge(waveforms: np.ndarray,
                     peak_position: int,
                     n_samples: int) -> np.ndarray:
    lower = 8
    upper = 8

    charge_temp = []
    for i in range(len(peak_position)):
        if peak_position[i] < lower:
            charge_temp.append(np.sum(waveforms[i, :peak_position[i] + upper]))
        elif peak_position[i] >= n_samples - upper:
            charge_temp.append(np.sum(waveforms[i, peak_position[i] - lower:]))
        else:
            charge_temp.append(
                    np.sum(
                        waveforms[
                            i, peak_position[i]
                            - lower:peak_position[i] + upper]))
    return np.asarray(charge_temp)


@njit
def clean_image(raw_charge: np.ndarray, grid_ind: List) -> np.ndarray:
    # constant values relating to the proportion of a square covered
    # by part of a circle
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
    charge_temp = np.zeros((40, 40))
    for i, val in enumerate(raw_charge):
        charge_temp[grid_ind[i] % 40, grid_ind[i] // 40] = val

    # charge = np.pad(charge, (2,), constant_values=(0,)
    charge = np.zeros((44, 44))
    charge[2:42, 2:42] = charge_temp

    image = np.zeros((40, 40))
    for i in range(40):
        for j in range(40):
            if charge[i+2, j+2] == 0.0:
                image[j, i] = np.NaN
                continue
            s = np.sum(charge[i:i+5, j:j+5] * aperture)
            if s > 150:
                image[j, i] = charge[i+2, j+2]
    return image

@njit
def hillas(charge_coords):
    """Calculates the Hillas parameters for an event."""
    x = 0
    y = 0
    x2 = 0
    y2 = 0
    xy = 0
    CHARGE = 0
    CHARGE = np.nansum(charge_coords[2])
    if CHARGE == 0.0:
        CHARGE = 1.0
    x = np.nansum(charge_coords[0] * charge_coords[2])
    y = np.nansum(charge_coords[1] * charge_coords[2])
    x2 = np.nansum(charge_coords[0] ** 2 * charge_coords[2])
    y2 = np.nansum(charge_coords[1] ** 2 * charge_coords[2])
    xy = np.nansum(charge_coords[0] * charge_coords[1] * charge_coords[2])

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
    q = np.nansum(q_coord * charge_coords[2]) / CHARGE
    q2 = np.nansum(q_coord ** 2 * charge_coords[2]) / CHARGE
    azwidth = q2 - q ** 2
    alpha = np.arcsin(miss / dis)
    return [x, y, width, length, miss, dis, azwidth, alpha]

class Cleaner():

    def __init__(self, runID=None, pedID=None, mod_list=None,
                 datadir=None, savedir=None, chPerPacket=32):

        print("Beginning image cleaning...")
        self.runID = runID
        self.pedID = pedID
        self.mod_list = mod_list
        self.datadir = datadir
        self.savedir = savedir
        self.chPerPacket = chPerPacket

        if self.runID is None:
            print("A run number must be specified!")
            sys.exit()

        self.runfile = f"run{self.runID}.fits"
        self.pedfile = f"ped{self.pedID}.tcal"
        self.calfile = f"cal{self.runID}.r1"

        try:
            if os.path.isfile(f"{self.datadir}/{self.runfile}") is False:
                raise IOError
            else:
                pass
        except IOError as ex:
            print(f"No raw data for run {self.runID} \
                     found at {self.datadir}!\nQuitting now...")
            sys.exit(ex)

        if self.pedID is None and os.path.isfile(
                f"{self.datadir}/{self.calfile}") is False:
            print("Either calibrated data must already exist or pedestal\
                   data must be specified! Quitting now...")
            sys.exit()

        if os.path.isfile(f"{self.datadir}/{self.pedfile}") is False:
            self._generate_pedestals()
        else:
            print("Pedestal database found!")

        if os.path.isfile(f"{self.datadir}/{self.calfile}") is False: # FIXME: This is temporary to force calibration using an updated pedestal file
            self._apply_calibration()
        else:
            print("Calibrated data found!")

        try:
            print(f"Trying to create save directory at: {self.savedir}")
            os.mkdir(self.savedir)
        except OSError:
            pass
        except Exception as ex:
            print(ex)
            pass

        self.calreader = target_io.WaveformArrayReader(
                f"{self.datadir}/{self.calfile}")
        self.n_pixels = self.calreader.fNPixels
        self.n_samples = self.calreader.fNSamples
        self.n_events = self.calreader.fNEvents
        self.n_asics = 4
        self.n_channels = 16
        self.n_modules = len(self.mod_list)
        self.mod_pos = {4: 5, 5: 6, 1: 7, 3: 8, 2: 9,
                        103: 11, 125: 12, 126: 13, 106: 14, 9: 15,
                        119: 17, 108: 18, 110: 19, 121: 20, 8: 21,
                        115: 23, 123: 24, 124: 25, 112: 26, 7: 27,
                        100: 28, 111: 29, 114: 30, 107: 31, 6: 32,
                        101: 14}  # 101 was formerly in slot 14 before it broke
        self.pos_grid = {5: (1, 1), 6: (1, 2), 7: (1, 3), 8: (1, 4), 9: (1, 5),
                         11: (2, 1), 12: (2, 2), 13: (2, 3), 14: (2, 4),
                         15: (2, 5), 17: (3, 1), 18: (3, 2), 19: (3, 3),
                         20: (3, 4), 21: (3, 5), 23: (4, 1), 24: (4, 2),
                         25: (4, 3), 26: (4, 4), 27: (4, 5), 28: (5, 1),
                         29: (5, 2), 30: (5, 3), 31: (5, 4), 32: (5, 5)}
        # Following items used for internal calculation, so not in self
        mod_nums = [100, 111, 114, 107, 6, 115, 123, 124, 112, 7, 119,
                    108, 110, 121, 8, 103, 125, 126, 106, 9, 4, 5, 1, 3, 2]
        fpm_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24]

        fpm_pos = np.mgrid[0:5, 0:5]
        fpm_pos = zip(fpm_pos[0].flatten(), fpm_pos[1].flatten())

        mod_to_fpm = dict(zip(mod_nums, fpm_nums))
        fpm_to_pos = dict(zip(fpm_nums, fpm_pos))

        ch_nums = np.array([[21, 20, 17, 16, 5, 4, 1, 0],
                            [23, 22, 19, 18, 7, 6, 3, 2],
                            [29, 28, 25, 24, 13, 12, 9, 8],
                            [31, 30, 27, 26, 15, 14, 11, 10],
                            [53, 52, 49, 48, 37, 36, 33, 32],
                            [55, 54, 51, 50, 39, 38, 35, 34],
                            [61, 60, 57, 56, 45, 44, 41, 40],
                            [63, 62, 59, 58, 47, 46, 43, 42]])
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
            # print(f"Channel Map: {ch_map}")
            j = num_columns - 1 - j
            pix_ind = np.array(indices[
                (8*i):8*(i+1), (8*j):8*(j+1)]).reshape(-1)
            # print(f"Pixel Index: {pix_ind}")
            for asic in range(4):
                for ch in range(16):
                    self.grid_ind.append(int(pix_ind[
                        ch_map[asic * 16 + ch]]))

    def _apply_calibration(self):
        """ Private method that applies pedestal subtraction from the
            command line.
        """
        os.system(
            f"apply_calibration_SCT -i {self.datadir}/{self.runfile} \
              -p {self.datadir}/{self.pedfile} -o {self.datadir}/\
              {self.calfile}")

    def _generate_pedestals(self):
        """ Private method that runs the generation of a pedestal
            database from the command line.
        """
        os.system(f"generate_ped_SCT -i {self.datadir}/run{self.pedID}\
                    .fits -o {self.datadir}/{self.pedfile}")

    def _make_camera_image(self, image, ev, time_s, h_params):
        time_s = int(time_s)
        time_str = datetime.utcfromtimestamp(time_s).strftime("%Y-%m-%d %H:%M:%S")
        nl = "\n"
        maxZ = np.nanmax(image)
        fig = plt.gcf()
        #print(fig.get_size_inches())
        if fig.get_size_inches().all() == np.array([18., 15.]).all():
            plt.close(fig)
            fig = plt.figure(figsize=(18, 15))
        # fig = plt.figure(figsize=(18, 15))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.04, hspace=0.04)
        #gs.tight_layout(fig, pad=0.0, h_pad=0.001, w_pad=0.001)


        for i in range(5):
            for j in range(5):
                sub_image = image[i*8:i*8+8, j*8:j*8+8]
                ax = plt.subplot(gs[4 - i, j])
                c = ax.pcolormesh(sub_image, vmin=0, vmax=maxZ)
                ax.axis("off")
                ax.set_aspect("equal")
                if i == 4 and j == 4:
                    #circle = Circle((0, 0), radius=1, clip_on=False, edgecolor="orange", fill=False)
                    #print(h_params)
                    x0 = h_params[0]
                    y0 = h_params[1]
                    width = 2 * h_params[2]
                    length = 2 * h_params[3]
                    alpha = h_params[-1]
                    print(f"Alpha: {np.degrees(alpha)}")
                    psi = np.abs(np.arctan(x0/y0))
                    print(f"Psi: {np.degrees(psi)}")
                    beta = np.degrees(psi - alpha)
                    print(f"Beta: {beta}")
                    ellipse = Ellipse((x0 + (-12-0.42*2), y0 + (-12-0.32*2)), width=width, 
                                       height=length, angle=beta, clip_on=False, edgecolor="red", 
                                       fill=False, linewidth=3)
                    #ax.add_patch(circle)
                    x0 += -12 - 0.42 * 2
                    y0 += -12 - 0.32 * 2
                    orx = -12 - 0.42 * 2
                    ory = -12 - 0.32 * 2
                    
                    gamma = np.radians(90. - np.abs(beta))
                    x1 = np.sign(x0) * length * np.cos(gamma) + x0
                    y1 = (length * np.sin(gamma) + y0)
                    x2 = x0 - np.sign(x0) * length * np.cos(gamma)
                    y2 = (y0 - length * np.sin(gamma))
                    
                    # print(y1)
                    # print(y2)
                    dis = lines.Line2D([orx, x0], [ory, y0], clip_on=False, color="orange", linewidth=3, linestyle="--")
                    maj = lines.Line2D([x1, x2], [y1, y2], clip_on=False, color="orange", linewidth=3, linestyle="-")
                    ax.add_line(dis)
                    ax.add_patch(ellipse)
                    ax.add_line(maj)

        fig.subplots_adjust(right=0.8, top=0.9, bottom=0.1)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_label("Charge (Photoelectrons)", rotation=270,
                       size=24, labelpad=24)
        cbar_ax.tick_params(labelsize=20)
        fig.suptitle(f"Run {self.runID} Event {ev}{nl}{time_str} UTC", fontsize=30)
        print("Saving camera image...")
        fig.savefig(
            f"{self.savedir}/run{runID}_ev{ev}_cleaned_image_hillas_overlay.png"
        )
        print("Saved!")
        #fig.clf()

    def _normalize_gain(self, waveforms):
        waveforms[:576, :] /= 2.0
        return waveforms
    
    def _write_row(self, writer):
        writer.writerow([self.ev, self.drop_packs, self.flashers,
                         self.noise, self.bright_noise, self.shower_cands])

    def clean_event(self, start=None, stop=None, make_images=True):
        if start is None:
            sys.exit()
        elif stop is None:
            stop = start + 1

        event_list = range(start, stop, 1)
        for ev in tqdm(event_list):
            ev = int(ev)
            waveforms = np.zeros((self.n_pixels, self.n_samples),
                                 dtype=np.float32)
            self.calreader.GetR1Event(ev, waveforms)
            self.calreader.GetTimeStamp(ev)
            time_s = self.calreader.fCPU_s
            time_ns = self.calreader.fCPU_ns
            # waveforms = self._normalize_gain(waveforms)
            peak_position = np.argmax(waveforms, axis=1)
            charge = calculate_charge(waveforms, peak_position, self.n_samples)
            charge = apply_gains(charge)
            image = clean_image(charge, self.grid_ind)
            charge_coords = [[i - 19.5 + 0.42 * ((i // 8) - 2), 
                j - 19.5 + 0.32 * ((j // 8) - 2),
                image[j, i]] 
                for i in range(40) for j in range(40)]
            #print(charge_coords)
            h_params = hillas(np.asarray(charge_coords).T)
            if make_images is True:
                self._make_camera_image(image, ev, time_s, h_params)

    def full_pass(self, n=None, track_classes=True,
                  make_images=False):
        if n is not None:
            n_events = n
        else:
            n_events = self.n_events

        if make_images is True:
            print("Cleaned images will be created for shower candidates.")
        if track_classes is True:
            print("Event classifications will be recorded after processing.")
            print("Creating event classification file...")
            if n is not None:
                filename = f"{self.savedir}/\
                             run{runID}_ev_classes_{n}_events.csv"
            else:
                filename = f"{self.savedir}/run{runID}_ev_classes.csv"
            f = open(filename, 'w', newline='')
            writer = csv.writer(f, delimiter=",")
            header = ["Event", "Dropped", "Flasher", "Noise",
                          "Shower_Candidate", "Bright_Noise"]
            writer.writerow(header)
        for ev in tqdm(range(n_events)):
            ev = int(ev)
            self.ev = ev
            self.drop_packs = 0
            self.flashers = 0
            self.noise = 0
            self.bright_noise = 0
            self.shower_cands = 0
            waveforms = np.zeros((self.n_pixels, self.n_samples),
                                 dtype=np.float32)
            self.calreader.GetR1Event(ev, waveforms)
            self.calreader.GetTimeStamp(ev)
            time_s = self.calreader.fCPU_s
            time_ns = self.calreader.fCPU_ns
            # waveforms = self._normalize_gain(waveforms)
            peak_position = np.argmax(waveforms, axis=1)
            charge = calculate_charge(waveforms, peak_position, self.n_samples)
            charge = apply_gains(charge)
            no_charge = [val for val in charge if val == 0.0]
            if len(no_charge) > 750:
                if track_classes is True:
                    self.drop_packs = 1
                    self._write_row(writer)
                continue
            brights = [val for val in charge if val > 20]
            if len(brights) > 800:
                if track_classes is True:
                    self.flashers = 1
                    self._write_row(writer)
                continue
            elif len(brights) < 5:
                if track_classes is True:
                    self.noise = 1
                    self._write_row(writer)
                continue
            if track_classes is True:
                self.shower_cands = 1
            image = clean_image(charge, self.grid_ind)
            image_size = [val for row in image for val in row  if val > 1.e-4]
            if len(image_size) <= 5:
                if track_classes is True:
                    self.bright_noise = 1
                    self._write_row(writer)
                continue
            if track_classes is True:
                self._write_row(writer)
            if make_images is True:
                self._make_camera_image(image, ev, time_s)
        if track_classes is True:
            f.close()

if __name__ == "__main__":
    runID = 328555
    pedID = 328587  # FIXME: Replace with new good pedestal file!
    mod_list = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                100, 103, 106, 107, 108,
                111, 112, 114, 115, 119,
                121, 123, 124, 125, 126]
    datadir = "/mnt/lfs7/wipac/CTA/target5and7data/runs_320000_through_329999"
    savedir = f"{datadir}/run{runID}/calibrated_images"

    cleaner = Cleaner(runID=runID, pedID=pedID, mod_list=mod_list,
                      datadir=datadir, savedir=savedir)
    # cleaner.full_pass(make_images=True)
    cleaner.clean_event(start=1826)
