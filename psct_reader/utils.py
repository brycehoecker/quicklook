from datetime import datetime
import heapq
from itertools import combinations
import logging

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as lines
from matplotlib.patches import Ellipse
import numpy as np
from numba import njit, prange
from numba.typed import List

class TrackingError(Exception): pass

@njit
def calculate_charge(waveforms: np.ndarray,
                     peak_position: np.ndarray,
                     n_samples: int) -> np.ndarray:
    """
    Calculate the charge in each pixel of the pSCT camera for an event.

    :param waveforms: np.ndarray, (1536, 128) calibrated camera data
    :param peak_position: np.ndarray, (1536) positions of waveform peaks
    :param n_samples: int, number of samples in each waveform
    :return: (1536) charge in pixels
    :rtype: np.ndarray
    """
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
def clean_image(raw_charge: np.ndarray,
                grid_ind: List,
                mean_noise: np.ndarray,
                dead_pixels: np.ndarray,
                nan: bool = True,
                thresh: float = 2.) -> np.ndarray:
    """
    Clean the camera image using the Wood, et al. (2015) circular aperture method.
    Takes in the calculated charges and some metainformation and returns a (40, 40) camera image.

    :param raw_charge: np.ndarray, (1536) charge in pixels
    :param grid_ind: numba.typed.List, used to correctly go from 1d charge list to 2d camera image
    :param mean_noise: np.ndarray, used to provide unbiased comparison for cleaning algorithm
    :param dead_pixels: np.ndarray, sets dead pixels to 0
    :param nan: bool = True, sets dead pixels to np.NaN
    :param thresh: float = 2., multiplier into baseline noise to determine threshold
    :return: (40, 40) cleaned camera image
    :rtype: np.ndarray
    """
    a = 0.2138
    b = 0.479
    c = 0.985
    aperture = [[0., a, b, a, 0.],
                [a, c, 1., c, a],
                [b, 1., 1., 1., b],
                [a, c, 1., c, a],
                [0., a, b, a, 0.]]
    aperture = np.asarray(aperture)
    raw_charge *= dead_pixels
    charge_temp = np.zeros((40, 40))
    for i, val in enumerate(raw_charge):
        charge_temp[grid_ind[i]//40, grid_ind[i]%40] = val
    noise_temp = np.zeros((40, 40))
    for i, val in enumerate(mean_noise):
        noise_temp[grid_ind[i]//40, grid_ind[i]%40] = val
    charge = np.zeros((44, 44))
    charge[2:42, 2:42] = charge_temp
    noise_pad = np.zeros((44, 44))
    noise_pad[2:42, 2:42] = noise_temp
    image = np.zeros((40, 40))
    for i in range(40):
        for j in range(40):
            if charge[i+2, j+2] == 0.0:
                if nan:
                    image[i, j] = np.NaN
                else:
                    image[i, j] = 0.0
                continue
            s = np.sum(charge[i:i+5, j:j+5] * aperture)
            noise = np.sum(noise_pad[i:i+5, j:j+5] * aperture)
            if s > noise*thresh:
                image[i, j] = charge[i+2, j+2]
    return image

@njit
def asic_island_cleaning(clean_image: np.ndarray) -> np.ndarray:
    pixel_counter = 0
    for i in prange(0, 40, 4):
        for j in prange(0, 40, 4):
            for pix in clean_image[i:i+4, j:j+4].flatten():
                if pix > 1e-4:
                    pixel_counter += 1
            if pixel_counter < 2:
                clean_image[i:i+4, j:j+4] = np.zeros((4, 4))
            pixel_counter = 0
    return clean_image

@njit
def hillas(charge_coords):
    """
    Calculates the Hillas parameters for a camera image.

    :param charge_coords: np.ndarray, (3, 1600) three lists; x, y, charge
    :return: list of Hillas parameters for event
    :rtype: List
    """
    x = 0
    y = 0
    x2 = 0
    y2 = 0
    xy = 0
    CHARGE = 0
    CHARGE = np.nansum(charge_coords[2])
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
    z = np.sqrt(d ** 2 + 4 * (S_xy ** 2))
    width = np.sqrt((S2_y + a ** 2 * S2_x - 2 * a * S_xy) / (1 + a ** 2))
    length = np.sqrt((S2_x + a ** 2 * S2_y + 2 * a * S_xy) / (1 + a ** 2))
    miss = b / np.sqrt(1 + a ** 2)
    dis = np.sqrt(x ** 2 + y ** 2)
    psi = np.arctan(((d + z) * x + 2 * S_xy * x) / ((2 * S_xy - (d - z)) * x))

    q_coord = (x - charge_coords[0]) * (x / dis) + (y - charge_coords[1]) * (y / dis)
    q = np.nansum(q_coord * charge_coords[2]) / CHARGE
    q2 = np.nansum(q_coord ** 2 * charge_coords[2]) / CHARGE
    azwidth = q2 - q ** 2
    alpha = np.arcsin(miss / dis)
    return [a, b, x, y, width, length, miss, dis, azwidth, alpha, psi]

def find_nearest(array, value):
    """
    Finds element of 1d numpy array closest to value and returns index.

    :param array: np.ndarray, a 1d numpy array
    :param value: float, a number
    :return: index in array of element closest to value
    :rtype: int
    """
    return ((np.abs(np.asarray(array) - value)).argmin())

def list_subruns(timestamps):
    """
    Takes list of timestamps and subdivides them into subruns based on temperature and current reading breaks.

    :param timestamps: np.ndarray, TACK timestamps for all events in a run
    :return: subruns, a list of lists, subdividing timestamps into subrun
    :rtype: List
    """
    delta_t = [j - i for i, j in zip(timestamps[:-1], timestamps[1:])]
    subruns = []
    temp = []
    for i, time in enumerate(timestamps[:-1]):
        if delta_t[i] < 1.0:
            temp.append(time)
            continue
        elif delta_t[i] >= 1.0:
            temp.append(time)
            subruns.append(temp)
            temp = []
            continue
        else:
            print("How'd we end up here?")
    subruns.append(temp)
    subruns[-1].append(timestamps[-1])
    return subruns

def get_flasher_events(subruns, opt_pers):
    """
    Uses list of subruns and their optimal flasher periods to create list of flasher identified events.

    :param subruns: List, the list of lists of timestamps returned by list_subruns
    :param opt_pers: List, a list of optimal periods, as generated one at a time by scan_rates
    :return: flasher events
    :rtype: List
    """
    ind = 0
    flasher_events = []
    for i, subrun in enumerate(subruns):
        hist = []
        bin_edges = []
        if not np.isnan(opt_pers[i]):
            hist, bin_edges = np.histogram(np.asarray(subrun) % opt_pers[i], bins=150)
            thresh = np.mean(hist)
            bins = [index for index, val in enumerate(hist) if val > thresh*2]
            bin_numbers = np.digitize(np.asarray(subrun) % opt_pers[i], bins=bin_edges) - 1
            for val in bin_numbers:
                if val in bins:
                    flasher_events.append(ind)
                ind += 1
        if np.isnan(opt_pers[i]):
            ind += len(subrun)
    return flasher_events

def scan_rates(rate, subrun):
    """
    Scans over a subrun of timestamps and uses the nominal flasher rate to find the optimal flasher period value.

    :param rate: float, nominal flasher rate
    :param subrun: List, list of TACK timestamps in this subrun
    :return: optimal period value for the subrun to find flashers
    :rtype: float
    """
    if len(subrun) < 20:
        return np.nan
    sub = np.asarray(subrun)
    period = 1.0 / rate
    per_space = np.linspace(period, period + 0.0002, num=1000, endpoint=True)
    opt_per = 0.0
    hist_max = 0.0
    for per in per_space:
        hist, _ = np.histogram(sub % per, bins=100)
        if np.amax(hist) > hist_max:
            hist_max = np.amax(hist)
            opt_per = per
    return opt_per

@njit
def get_bright_coordinates(peaks: np.ndarray, thresh: float) -> List:
    """
    Finds the pixels with waveform peaks above an empirical 2.0 ADC count / Poisson gain cutoff.

    :param peaks: np.ndarray, (40, 40) array of waveform peak values for one event
    :return: list of pixels with peaks above threshold
    :rtype: numpy.typed.List
    """
    bright_coordinates = List()
    for i in range(peaks.shape[0]):
        for j in range(peaks.shape[1]):
            # if peaks[i, j] > 2.0:
            if peaks[i, j] > thresh:
                bright_coordinates.append((i, j))
    return bright_coordinates

@njit
def n_reversed(list_: np.ndarray) -> int:
    """
    Produces an iterable of the input list, only in reverse. Like list.reverse() only compatible with numba.njit.

    :param list_: np.ndarray, a 1d numpy array
    :return: yields the list in reverse order
    :rtype: int
    """
    i = len(list_)
    while i > 0:
        i -= 1
        yield list_[i]

def is_noise(peaks: np.ndarray, brights=False, thresh=1.5) -> bool:
    """
    Takes array of waveform peak values and determines whether an event is due to electronics noise or not by looking for neighboring above threshold pixels.

    :param peaks: np.ndarray, (40, 40) array of waveform peak values for one event
    :param brights: bool = False, prints out bright coordinates when True
    :param thresh: float = 1.5, threshold for signal peaks in p.e.
    :return: Boolean value answering the question, "Is this event triggered on noise?"
    :rtype: bool
    """
    bright_coords = get_bright_coordinates(peaks, thresh)
    if brights is True:
        print(bright_coords)
    count = 0
    for pair in combinations(bright_coords, 2):
        x_dist = np.abs(pair[0][0] - pair[1][0])
        y_dist = np.abs(pair[0][1] - pair[1][1])
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
        if dist < np.sqrt(2)+0.1:
            #print(pair)
            count += 1
        if count == 4:
            break
    if count < 4:
        return True
    else:
        return False

def get_mean_noise():
    """
    Loads the noise averages file into a numpy array.

    :return: (1536) 1d array of noise averages, in pixel ordering
    :rtype: np.ndarray
    """
    mean_noise = np.load("/home/bmode/analysis/noise_averages.npz")["arr_0"]
    return mean_noise

def get_grid_ind():
    """
    Calculates the grid index, which is used to go from (1536) pixel data to (40, 40) camera image.

    :return: grid index
    :rtype: numba.typed.List
    """
    mod_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 103, 106, 107,
                108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]

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
    grid_ind = List()
    for index, mod in enumerate(mod_list):
        i, j = fpm_to_pos[mod_to_fpm[mod]]
        ch_map = dict()
        if j % 2 == 0:
            ch_map = rot_ch_to_pos
        else:
            ch_map = ch_to_pos
        j = num_columns - 1 - j
        pix_ind = np.array(indices[
            (8*i):8*(i+1), (8*j):8*(j+1)]).reshape(-1)
        for asic in range(4):
            for ch in range(16):
                grid_ind.append(int(pix_ind[
                    ch_map[asic * 16 + ch]]))
    return grid_ind

def get_dead_pixels():
    """
    Creates an array representing the currently identifying dead / dying pixels and sends them to 0.

    :return: (1536) dead pixels array, in 1d pixel order, used as a mask of sorts
    :rtype: np.ndarray
    """
    dead_pixels = np.ones(1536)
    dead_pixels[0:64] = 0.
    dead_pixels[192:256] = 0.
    dead_pixels[512:576] = 0.
    dead_pixels[688:704] = 0.
    dead_pixels[502] = 0.
    dead_pixels[426:428] = 0.
    dead_pixels[352] = 0.
    dead_pixels[357] = 0.
    dead_pixels[1016] = 0.
    dead_pixels[676] = 0.
    return dead_pixels

def plot_camera_image(image, run, ev, keep=False, directory=None):
    """
    Uses numpy and matplotlib to generate a somewhat realistic camera image. This can optionally be saved.

    :param image: np.ndarray, (40, 40) camera image, either cleaned or not
    :param run: int, used in title of plot and potentially in file name
    :param ev: int, used in title of plot and potentially in file name
    :param keep: bool, defaults to False, used to set whether the plot is saved to file or not
    :param directory: str, defaults to None, used to specify a specific directory for a saved plot
    """
    maxZ = np.nanmax(image)
    fig = plt.gcf()
    if fig.get_size_inches().all() == np.array([18., 15.]).all():
        plt.close(fig)
        fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.04, hspace=0.04)

    for i in range(5):
        for j in range(5):
            sub_image = image[i*8:i*8+8, j*8:j*8+8]
            ax = plt.subplot(gs[4 - i, j])
            c = ax.pcolormesh(sub_image, vmin=0, vmax=maxZ, cmap="viridis")
            ax.axis("off")
            ax.set_aspect("equal")

    fig.subplots_adjust(right=0.8, top=0.9, bottom=0.1)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(c, cax=cbar_ax)
    cbar.set_label("Charge (Photoelectrons)", rotation=270,
                   size=24, labelpad=24)
    cbar_ax.tick_params(labelsize=20)
    fig.suptitle(f"Run {run} Event {ev}", fontsize=30)
    if keep is True:
        if directory is None:
            fig.savefig(f"run{run}_ev{ev}_camera_image.png")
        else:
            fig.savefig(f"{directory}/run{run}_ev{ev}_camera_image.png")

def plot_camera_image_hillas(image,
                             run,
                             ev,
                             timestr,
                             h_params,
                             keep=False,
                             directory=None):
    """
    Same as other plot_camera_image, only adds the event time to the title and overlays the Hillas parameters.
    Should be used only with a cleaned image with precalculated Hillas parameters.

    :param image: np.ndarray, (40, 40) should be a cleaned camera image
    :param run: int, used in title of plot and potentially in file name
    :param ev: int, used in title of plot and potentially in file name
    :param timestr: str, used in title of plot
    :param h_params: List, used to generate Hillas overlay
    :param keep: bool, defaults to False, used to set whether the plot is saved to file or not
    :param directory: str, defaults to None, used to specify a specific directory for a saved plot
    """
    maxZ = np.nanmax(image)
    fig = plt.gcf()
    if fig.get_size_inches().all() == np.array([18., 15.]).all():
        plt.close(fig)
        fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.04, hspace=0.04)

    for i in range(5):
        for j in range(5):
            sub_image = image[i*8:i*8+8, j*8:j*8+8]
            ax = plt.subplot(gs[4 - i, j])
            c = ax.pcolormesh(sub_image, vmin=0, vmax=maxZ, cmap="viridis")
            ax.axis("off")
            ax.set_aspect("equal")
            if i == 4 and j == 4:
                #circle = Circle((0, 0), radius=1, clip_on=False, edgecolor="orange", fill=False)
                #print(h_params)
                x = h_params[0]
                y = h_params[1]
                x0 = h_params[0]
                y0 = h_params[1]
                width = 2*h_params[2]
                length = 2*h_params[3]
                alpha = h_params[-1]
                psi = np.abs(np.arctan(x0/y0))
                if x0 * y0 < 0.:
                    beta = np.degrees(psi - alpha)
                elif x0 >= 0. and y0 >= 0.:
                    beta = np.degrees(psi - alpha)
                elif x0 < 0. and y0 < 0.:
                    beta = np.degrees(psi + alpha)
                x0 += -12 - 0.42 * 2
                y0 += -12 - 0.32 * 2
                if y >= 0. and x >= 0.:
                    ellipse = Ellipse((x0, y0), width=width,
                                       height=length, angle=-beta, clip_on=False, edgecolor="red",
                                       fill=False, linewidth=3)
                elif y >= 0. and x < 0.:
                    ellipse = Ellipse((x0, y0), width=width,
                                       height=length, angle=beta, clip_on=False, edgecolor="red",
                                       fill=False, linewidth=3)
                elif y < 0. and x >= 0.:
                    ellipse = Ellipse((x0, y0), width=width,
                                       height=length, angle=beta, clip_on=False, edgecolor="red",
                                       fill=False, linewidth=3)
                else:
                    ellipse = Ellipse((x0, y0), width=width,
                                       height=length, angle=-beta, clip_on=False, edgecolor="red",
                                       fill=False, linewidth=3)
                orx = -12 - 0.42 * 2
                ory = -12 - 0.32 * 2
                if y >= 0. and x >= 0.:
                    gamma = np.radians(90. + beta)
                if y >= 0. and x <= 0.:
                    gamma = np.radians(90. + beta)
                if y < 0. and x >= 0.:
                    gamma = np.radians(90. - beta)
                elif y < 0. and x < 0.:
                    gamma = np.radians(90. - beta)
                    #gamma = 45. - (gamma - 45.)
                #gamma = np.radians(90. - np.abs(beta))
                #x1 = np.sign(x0) * length * np.cos(gamma) + x0
                x1 = -np.sign(x) * length * np.cos(gamma) + x0
                y1 = (length * np.sin(gamma) + y0)
                #x2 = x0 - np.sign(x0) * length * np.cos(gamma)
                x2 = x0 + np.sign(x) * length * np.cos(gamma)
                y2 = (y0 - length * np.sin(gamma))
                dis = lines.Line2D([orx, x0], [ory, y0], clip_on=False, color="orange", linewidth=3, linestyle="--")
                maj = lines.Line2D([x1, x2], [y1, y2], clip_on=False, color="orange", linewidth=3, linestyle="-")
                ax.add_line(dis)
                ax.add_patch(ellipse)
                ax.add_line(maj)

    fig.subplots_adjust(right=0.8, top=0.9, bottom=0.1)
    cbar_ax = fig.add_axes([0.81, 0.1, 0.03, 0.8])
    cbar = fig.colorbar(c, cax=cbar_ax)
    #cbar.set_label("Charge (Photoelectrons)", rotation=270,
    #               size=40, labelpad=40, color="red")
    #ticks = cbar.locator()
    #pad_ticks = [str(int(tick)).rjust(3, r" ") for tick in ticks]
    #cbar_ax.set_yticklabels(pad_ticks)
    cbar_ax.tick_params(labelsize=40)
    ax.text(14.51, -2.5, "Charge (Photoelectrons)", rotation=270, fontsize=40)
    ev = str(ev).rjust(6, "0")
    fig.suptitle(f"Prototype Schwarzschild-Couder Telescope Gamma Rays\nRun {run} Event {ev} ({timestr})", fontsize=39)
    #fig.suptitle(f"x: {x}, y: {y}", fontsize=24)
    if keep is True:
        if directory is None:
            fig.savefig(f"run{run}_ev{ev}_camera_image_hillas.png")
        else:
            fig.savefig(f"{directory}/run{run}_ev{ev}_camera_image_hillas.png")

def get_timestr(timestamp):
    """
    Takes CPU seconds timestamp for an event and creates a nice representative string in UTC time

    :param int timestamp: CPU seconds timestamp in UTC
    :return: time stamp in more readable formatting
    :rtype: str
    """
    time_str = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    return time_str

def pointing_from_utc(input_time, output_field, data):
    """
    Takes an input event time and a tracking log data field and interpolates to find the closest item in the log.

    :param str input_time: Time stamp for tracking log lookup in UTC datetime (YYYY-MM-DD hh:mm:ss)
    :param str output_type: Name of field in tracking log
    :param pd.DataFrame data: Pandas data frame containing the tracking log
    :return: Interpolated value of tracking log data, or closest value if Bool or the log has this time
    """
    utc = np.asarray(data["current_Time_DT"]) # UTC time to the second from tracking log
    output = np.asarray(data[output_field]) # put output field as array for interpolation or output
    for index, t in enumerate(utc):
        if input_time == t:
            return output[index]
    after = np.searchsorted(utc, input_time)
    before = after - 1
    t_before = datetime.strptime(utc[after], '%Y-%m-%d %H:%M:%S')
    t_after = datetime.strptime(utc[before], '%Y-%m-%d %H:%M:%S')
    t_input = datetime.strptime(input_time, '%Y-%m-%d %H:%M:%S')
    t_s_delta = (t_after - t_before).total_seconds()
    if t_s_delta > 300:
        raise TrackingError
    if output[before] == output[after]:
        return output[before] # This is likely to be RA / Dec or the Boolean tracking values
    if output_field == "is_moving" or output_field == "is_off":
        return True
    elif output_field == "is_on_source" or output_field == "is_tracking":
        return False
    else:
        t_s_delta_input = (t_input - t_before).total_seconds()
        x = [0, t_s_delta]
        y = [output[before], output[after]] # guaranteed to be a non-Bool value at this point
        result = np.interp(t_s_delta_input, x, y)
        return result
    logging.debug("A critical error has occurred if we ended up here.")
    return None

@njit
def get_charge_coords(img: np.ndarray, delta_x: float, delta_y:float, for_plot: bool = False) -> np.ndarray:
    charge_coords = np.zeros((3, 40*40), dtype=np.float32)
    if for_plot:
        for i in prange(40):
            for j in prange(40):
                ind = 40 * i + j
                charge_coords[0, ind] = i - 19.5 + 0.42 * ((i // 8) - 2.)
                charge_coords[1, ind] = j - 19.5 + 0.32 * ((j // 8) - 2.)
                charge_coords[2, ind] = img[j, i]
    else:
        for i in prange(40):
            for j in prange(40):
                ind = 40 * i + j
                charge_coords[0, ind] = i - 19.5 + 0.31 * ((i // 8) - 2.) + delta_x
                charge_coords[1, ind] = j - 19.5 + 0.31 * ((j // 8) - 2.) + delta_y
                charge_coords[2, ind] = img[j, i]
    return charge_coords

@njit
def cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

@njit
def get_delta_x(input_time, culmination_time):
    x3 = 2.22e-12
    x2 = 1.22e-8
    x1 = -2.69e-4
    x0 = -1.69
    x = input_time - culmination_time
    return cubic(x, x3, x2, x1, x0)

@njit
def get_delta_y(input_time, culmination_time):
    y3 = -4.57e-12
    y2 = 3.21e-8
    y1 = 3.28e-4
    y0 = -2.15
    y = input_time - culmination_time
    return cubic(y, y3, y2, y1, y0)

@njit
def largest_island(mask, indices, indptr):
    labels = np.zeros(len(mask), dtype=np.int16) # flattened image of labeled islands
    labels[mask] = -1 # basic idea here: uninteresting pixels are labeled 0, and cleaned pixels are labeled -1
                           # then as we walk through the algorithm, each -1 gets replaced by an appropriate island label
    cleaning_pixels = np.where(mask)[0] # this gets the locations of the cleaned pixels
    n_cleaning_pixels = len(cleaning_pixels) # needed for for loop
    current_island = 0 # to be incremented
    to_check = []
    for i in range(n_cleaning_pixels): # algorithm starts at the beginning of each island and then recursively
        idx = cleaning_pixels[i] # begins walking through the rest of it after that, so the actions in the
        if labels[idx] != -1: # while loop are the same as those in the first part of the for loop
            continue # then we skip the pixels that have already been given a label
        current_island += 1
        labels[idx] = current_island
        loc_nbrs = indices[indptr[idx]:indptr[idx+1]] # this is specifically using the CSR matrix representation
        for n in range(len(loc_nbrs)):
            neighbor = loc_nbrs[n]
            if labels[neighbor] == -1:
                to_check.append(neighbor)
        while len(to_check) > 0:
            idx = to_check.pop()
            labels[idx] = current_island
            loc_nbrs = indices[indptr[idx]:indptr[idx+1]]
            for n in range(len(loc_nbrs)):
                neighbor = loc_nbrs[n]
                if labels[neighbor] == -1:
                    to_check.append(neighbor)
    return labels


@njit
def idef_split(pixel: np.ndarray, phase: int) -> bool:
    """
    A function that identifies whether a waveform is a split or not. The main points:
    - Checks if there are groups of consecutive samples in the calibrated waveform above a certain threshold ("peaks"),
    then check if those peaks are contained within two blocks
    - Checks for spikes in the derivative of 1 - 2 samples, then checks if those spikes are on or right before the
    beginning/end of blocks, and if those blocks are separated by 64 ns (i.e. if the blocks are consecutive)

    :param pixel: a (128,) array which has the waveform information for a single pixel in a single event
    :param phase: the phase information for the event in question
    :return: returns True if the waveform is a split, False if not
    """
    deriv = np.abs(deriv_1d(pixel))
    phase_jumps = []
    peaks = []
    threshold = 100

    i = 0
    while i < len(pixel):
        if pixel[i] > threshold:
            temp = []
            for j in range(i, len(pixel)):
                if pixel[j] > threshold:
                    temp.append(j)
                else:
                    peaks.append(temp)
                    i = j
                    break
        i += 1

    if len(peaks) != 2 or len([i for i in peaks if len(i) > 2]) != len(peaks):
        return False

    for i in range(1, len(deriv)):
        if (deriv[i] + deriv[i - 1])/4 > 90:
            if (i + phase) % 32 == 0:
                phase_jumps.append(i)
            else:
                return False

    if len(phase_jumps) != 2:
        return False
    elif phase_jumps[1] - phase_jumps[0] != 64:
        return False
    else:
        for item in peaks:
            for samp in item:
                if samp in range(phase_jumps[0] - 1, phase_jumps[-1] + 2):
                    continue
                else:
                    return False

    return True


def correct_split(pix_arr, phase_points):
    """
    A basic correction for splits. Breaks the waveform into three complete blocks and two incomplete blocks ("block1",
    "block2", "block3", "beg", and "end), then exchanges the first complete block with the third.

    :param pix_arr: A (128,) array which has the waveform information for a single pixel in a single event
    :param phase_points:
    :return: pix_arr, modified so that the first and third complete blocks are inversed
    """
    block1 = pix_arr[phase_points[0]:phase_points[1]]
    block2 = pix_arr[phase_points[1]:phase_points[2]]
    block3 = pix_arr[phase_points[2]:phase_points[3]]
    beg = pix_arr[:phase_points[0]]
    end = pix_arr[phase_points[3]:]
    pix_arr = np.concatenate((beg, block3, block2, block1, end))
    return pix_arr

