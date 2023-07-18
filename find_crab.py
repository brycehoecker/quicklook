
#Document has been edited by bryce hoecker starting 11/8/21 and finishing on....

#!/home/brycecentos/svn_folder/Bryces_CTA_project/svn_folder/analysis)
import sys																#
sys.path.append("/home/brycecentos/svn_folder/Bryces_CTA_project/svn_folder/analysis")  	#
import argparse															#pip
import heapq															#
import astropy.units as u												#
from astropy.time import Time											#
from astropy.coordinates import SkyCoord, EarthLocation, AltAz			#
from itertools import combinations										#
from numba import njit													#
from numba.typed import List											#
import numpy as np														#
from tqdm import tqdm													#

import make_clean_showers as mcs										#
from apply_gains import apply_gains										#
import target_io														#


@njit
def clean_image(raw_charge: np.ndarray, grid_ind: List, mean_noise: np.ndarray) -> np.ndarray:
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
    raw_charge_copy = raw_charge
    raw_charge_copy[688:704] = 0.0
    raw_charge_copy[192:256] = 0.0
    raw_charge_copy[0:64] = 0.0
    raw_charge_copy[512:576] = 0.0
    for i, val in enumerate(raw_charge_copy):
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
                image[i, j] = np.NaN
                continue
            s = np.nansum(charge[i:i+5, j:j+5] * aperture)
            noise = np.nansum(noise_pad[i:i+5, j:j+5] * aperture)
            if s > noise*2:
                image[i, j] = charge[i+2, j+2]
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
    miss = b / np.sqrt(1 + a ** 2)
    dis = np.sqrt(x ** 2 + y ** 2)

    q_coord = (x - charge_coords[0]) * (x / dis) + (y - charge_coords[1]) * (y / dis)
    q = np.nansum(q_coord * charge_coords[2]) / CHARGE
    q2 = np.nansum(q_coord ** 2 * charge_coords[2]) / CHARGE
    azwidth = q2 - q ** 2
    alpha = np.arcsin(miss / dis)
    return [a, b, x, y, width, length, miss, dis, azwidth, alpha]

def find_nearest(array, value):
    return ((np.abs(np.asarray(array) - value)).argmin())

def list_subruns(timestamps):
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
def get_bright_coordinates(peaks: np.ndarray) -> List:
    bright_coordinates = List()
    for i in range(peaks.shape[0]):
        for j in range(peaks.shape[1]):
            if peaks[i, j] > 2.0:
                bright_coordinates.append((i, j))
    return bright_coordinates

@njit
def n_reversed(list_: np.ndarray) -> int:
    i = len(list_)
    while i > 0:
        i -= 1
        yield list_[i]
        
def is_noise(peaks: np.ndarray) -> bool:
    bright_coords = get_bright_coordinates(peaks)
    count = 0
    for pair in combinations(bright_coords, 2):
        x_dist = np.abs(pair[0][0] - pair[1][0])
        y_dist = np.abs(pair[0][1] - pair[1][1])
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
        if dist < np.sqrt(2):
            #print(pair)
            count += 1
        if count == 4:
            break
    if count < 4:
        return True
    else:
        return False

def linear(x, a, b):
    return(a*x + b)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", metavar="runID")
    args = parser.parse_args()
    runID = int(args.r)
    print(f"Processing run: {runID}")
    runs = {328555: "on", 328557: "allon", 328564: "off", 
            328565: "on", 328567: "on", 328569: "off", 328572: "off", 
            328573: "on", 328574: "on", 328581: "off", 328583: "off",
            328585: "on", 328592: "off", 328597: "off", 328599: "on",
            328606: "off", 328608: "on", 328610: "on", 328615: "off",
            328617: "on", 328619: "on", 328627: "off", 328629: "allon",
            328630: "allon", 328631: "allon", 328640: "off", 328642: "on",
            328646: "on", 328700: "on", 328717: "off", 328719: "on",
            328732: "on", 328733: "on", 328748: "on", 328750: "on",
            328761: "on", 328763: "on", 328770: "on", 328772: "on",
            328781: "on", 328792: "on", 328794: "on", 328821: "on",
            328846: "on", 328854: "on", 328856: "on", 328865: "on",
            328867: "on"}

    datadir = "/home/bmode/cam_data"
    calfile = f"{datadir}/cal{runID}.r1"
    reader = target_io.WaveformArrayReader(calfile)

    n_pixels = reader.fNPixels
    n_samples = reader.fNSamples
    n_events = reader.fNEvents
    waveforms = np.zeros((n_pixels, n_samples), dtype=np.float32)
    mean_noise = np.load("/home/bmode/analysis/noise_averages.npz")["arr_0"]

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
    
    crab = SkyCoord.from_name("crab")
    if runs[runID] == "off":
        crab_off = SkyCoord(76.1333, 22.0145, unit="deg")
    elif runs[runID] == "on":
        crab_off = SkyCoord(91.1333, 22.0145, unit="deg")
    elif runs[runID] == "allon":
        crab_off = SkyCoord.from_name("crab")
    psct_loc = EarthLocation(lat=31.6716989799*u.deg, lon=-110.951291195*u.deg, height=1268*u.m)
    a_x, b_x = (-0.014436381477488698, 1.093377089794729)
    a_y_1, b_y_1 = (-0.03798099601727016, 0.36721872391895594) # az < 180 deg
    a_y_2, b_y_2 = (-0.11408711008257931, 7.51779699821602) # az > 180 deg
    hillas_parameters = []
    cpu_s = []
    cpu_ns = []
    tacks = []
    events = []
    size = []
    frac_2 = []
    az_list = []
    el_list = []
    print("Reading TACK times...")
    for ev in range(n_events):
        ev = int(ev)
        reader.GetR1Event(ev, waveforms)
        tacks.append(reader.fTACK_time)
    print("Identifying flasher events...")
    timestamps = np.asarray(tacks)
    timestamps -= timestamps[0]
    timestamps = [time / 1_000_000_000 for time in timestamps]
    subruns = list_subruns(timestamps)
    opt_pers = []
    for subrun in subruns:
        opt_pers.append(scan_rates(10.0, subrun))
    flasher_events = get_flasher_events(subruns, opt_pers)
    non_flasher_events = [ev for ev in range(n_events) if ev not in flasher_events]
    tack_diff = tacks[-1] - tacks[0]
    tack_split = tack_diff / 2.0 + tacks[0]
    #on_off_ind = find_nearest(tacks, tack_split)
    tacks = []
    print("Calculating Hillas parameters. This might take a while...")
    for ev in non_flasher_events:
        ev = int(ev)
        reader.GetR1Event(ev, waveforms)
        reader.GetTimeStamp(ev)
        tack = reader.fTACK_time
        peak_position = np.argmax(waveforms, axis=1)
        charges = mcs.calculate_charge(waveforms, peak_position, n_samples)
        waveforms = apply_gains(waveforms)
        peaks = np.zeros((40, 40))
        for i, wf in enumerate(waveforms):
            peaks[grid_ind[i]%40, grid_ind[i]//40] = np.amax(wf[30:])
        charges = apply_gains(charges)
        no_charge = [val for val in charges if val == 0.0]
        if len(no_charge) > 750:
            continue
        noise = is_noise(peaks)
        if noise is True:
            continue
        cpu_s_single = reader.fCPU_s
        if tack <= tack_split:
            if runs[runID] == "on" or runs[runID] == "allon":                
                time = Time(cpu_s_single, format="unix")
                crab_azel = crab.transform_to(AltAz(obstime=time, location=psct_loc))
                az, el = (crab_azel.az.deg, crab_azel.alt.deg)
                delta_x = linear(az, a_x, b_x)
                delta_y = linear(el, a_y_1, b_y_1) if az < 180 else linear(el, a_y_2, b_y_2)
            elif runs[runID] == "off":
                time = Time(cpu_s_single, format="unix")
                crab_azel = crab_off.transform_to(AltAz(obstime=time, location=psct_loc))
                az, el = (crab_azel.az.deg, crab_azel.alt.deg)
                delta_x = linear(az, a_x, b_x)
                delta_y = linear(el, a_y_1, b_y_1) if az < 180 else linear(el, a_y_2, b_y_2)
        elif tack > tack_split:
            if runs[runID] == "on" or runs[runID] == "allon":
                time = Time(cpu_s_single, format="unix")
                crab_azel = crab_off.transform_to(AltAz(obstime=time, location=psct_loc))
                az, el = (crab_azel.az.deg, crab_azel.alt.deg)
                delta_x = linear(az, a_x, b_x)
                delta_y = linear(el, a_y_1, b_y_1) if az < 180 else linear(el, a_y_2, b_y_2)
            elif runs[runID] == "off":
                time = Time(cpu_s_single, format="unix")
                crab_azel = crab.transform_to(AltAz(obstime=time, location=psct_loc))
                az, el = (crab_azel.az.deg, crab_azel.alt.deg)
                delta_x = linear(az, a_x, b_x)
                delta_y = linear(el, a_y_1, b_y_1) if az < 180 else linear(el, a_y_2, b_y_2)
        image = clean_image(charges, grid_ind, mean_noise)
        image_size = [val for row in image for val in row if val > 5]
        charge_coords = [[i - 19.5 + 0.31 * ((i // 8) - 2) + delta_x, 
                          j - 19.5 + 0.31 * ((j // 8) - 2) + delta_y,
                          image[j, i]] 
                          for i in range(40) for j in range(40)]
        try:
            h_params = hillas(np.asarray(charge_coords).T)
        except:
            continue
        az_list.append(az)
        el_list.append(el)
        hillas_parameters.append(np.asarray(h_params))
        events.append(ev)
        size.append(np.nansum(image.flatten()))
        bright_2 = np.sum(heapq.nlargest(2, image.flatten()))
        frac_2.append(bright_2 / np.nansum(image.flatten()))
        cpu_s.append(reader.fCPU_s)
        cpu_ns.append(reader.fCPU_ns)
        tacks.append(tack)
    hillas_parameters = np.asarray(hillas_parameters)
    tack_diff = tacks[-1] - tacks[0]
    tack_split = tack_diff / 2.0 + tacks[0]
    on_off_ind = find_nearest(tacks, tack_split)
    if runs[runID] == "off":
        on = hillas_parameters[on_off_ind:]
        off = hillas_parameters[:on_off_ind]
    elif runs[runID] == "on":
        on = hillas_parameters[:on_off_ind]
        off = hillas_parameters[on_off_ind:]
    elif runs[runID] == "allon":
        on = hillas_parameters
        off = []

    with open(f"/data/user/bmode/crab_data/v0.3/on_data_run{runID}.txt", "w") as f:
        f.write("Run, Event, CPU_s, CPU_ns, TACK, Az, El, Size, Frac2, a, b, x, y, width, length, miss, dis, azwidth, alpha\n")
        nl = "\n"
        for i in range(len(on)):
            f.write(f"{runID}, {events[i]}, {cpu_s[i]}, {cpu_ns[i]}, {tacks[i]}, {az_list[i]}, {el_list[i]}, {size[i]}, {frac_2[i]}, {on[i][0]}, {on[i][1]}, {on[i][2]}, {on[i][3]}, {on[i][4]}, {on[i][5]}, {on[i][6]}, {on[i][7]}, {on[i][8]}, {on[i][9]}{nl}")

    with open(f"/data/user/bmode/crab_data/v0.3/off_data_run{runID}.txt", "w") as f:
        f.write("Run, Event, CPU_s, CPU_ns, TACK, Az, El, Size, Frac2, a, b, x, y, width, length, miss, dis, azwidth, alpha\n")
        nl = "\n"
        for i in range(len(off)):
            j = i + len(on)
            f.write(f"{runID}, {events[j]}, {cpu_s[j]}, {cpu_ns[j]}, {tacks[j]}, {az_list[j]}, {el_list[j]}, {size[j]}, {frac_2[j]}, {off[i][0]}, {off[i][1]}, {off[i][2]}, {off[i][3]}, {off[i][4]}, {off[i][5]}, {off[i][6]}, {off[i][7]}, {off[i][8]}, {off[i][9]}{nl}")


'''
#!/home/bmode/anaconda2/envs/python_3/bin/python

import sys
sys.path.append("/home/bmode/analysis/")  
import argparse
import heapq

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from itertools import combinations
from numba import njit
from numba.typed import List
import numpy as np
from tqdm import tqdm

import make_clean_showers as mcs
from apply_gains import apply_gains
import target_io


@njit
def clean_image(raw_charge: np.ndarray, grid_ind: List, mean_noise: np.ndarray) -> np.ndarray:
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
    raw_charge_copy = raw_charge
    raw_charge_copy[688:704] = 0.0
    raw_charge_copy[192:256] = 0.0
    raw_charge_copy[0:64] = 0.0
    raw_charge_copy[512:576] = 0.0
    for i, val in enumerate(raw_charge_copy):
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
                image[i, j] = np.NaN
                continue
            s = np.nansum(charge[i:i+5, j:j+5] * aperture)
            noise = np.nansum(noise_pad[i:i+5, j:j+5] * aperture)
            if s > noise*2:
                image[i, j] = charge[i+2, j+2]
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
    miss = b / np.sqrt(1 + a ** 2)
    dis = np.sqrt(x ** 2 + y ** 2)

    q_coord = (x - charge_coords[0]) * (x / dis) + (y - charge_coords[1]) * (y / dis)
    q = np.nansum(q_coord * charge_coords[2]) / CHARGE
    q2 = np.nansum(q_coord ** 2 * charge_coords[2]) / CHARGE
    azwidth = q2 - q ** 2
    alpha = np.arcsin(miss / dis)
    return [a, b, x, y, width, length, miss, dis, azwidth, alpha]

def find_nearest(array, value):
    return ((np.abs(np.asarray(array) - value)).argmin())

def list_subruns(timestamps):
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
def get_bright_coordinates(peaks: np.ndarray) -> List:
    bright_coordinates = List()
    for i in range(peaks.shape[0]):
        for j in range(peaks.shape[1]):
            if peaks[i, j] > 2.0:
                bright_coordinates.append((i, j))
    return bright_coordinates

@njit
def n_reversed(list_: np.ndarray) -> int:
    i = len(list_)
    while i > 0:
        i -= 1
        yield list_[i]
        
def is_noise(peaks: np.ndarray) -> bool:
    bright_coords = get_bright_coordinates(peaks)
    count = 0
    for pair in combinations(bright_coords, 2):
        x_dist = np.abs(pair[0][0] - pair[1][0])
        y_dist = np.abs(pair[0][1] - pair[1][1])
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2)
        if dist < np.sqrt(2):
            #print(pair)
            count += 1
        if count == 4:
            break
    if count < 4:
        return True
    else:
        return False

def linear(x, a, b):
    return(a*x + b)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", metavar="runID")
    args = parser.parse_args()
    runID = int(args.r)
    print(f"Processing run: {runID}")
    runs = {328555: "on", 328557: "allon", 328564: "off", 
            328565: "on", 328567: "on", 328569: "off", 328572: "off", 
            328573: "on", 328574: "on", 328581: "off", 328583: "off",
            328585: "on", 328592: "off", 328597: "off", 328599: "on",
            328606: "off", 328608: "on", 328610: "on", 328615: "off",
            328617: "on", 328619: "on", 328627: "off", 328629: "allon",
            328630: "allon", 328631: "allon", 328640: "off", 328642: "on",
            328646: "on", 328700: "on", 328717: "off", 328719: "on",
            328732: "on", 328733: "on", 328748: "on", 328750: "on",
            328761: "on", 328763: "on", 328770: "on", 328772: "on",
            328781: "on", 328792: "on", 328794: "on", 328821: "on",
            328846: "on", 328854: "on", 328856: "on", 328865: "on",
            328867: "on"}

    datadir = "/home/bmode/cam_data"
    calfile = f"{datadir}/cal{runID}.r1"
    reader = target_io.WaveformArrayReader(calfile)

    n_pixels = reader.fNPixels
    n_samples = reader.fNSamples
    n_events = reader.fNEvents
    waveforms = np.zeros((n_pixels, n_samples), dtype=np.float32)
    mean_noise = np.load("/home/bmode/analysis/noise_averages.npz")["arr_0"]

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
    
    crab = SkyCoord.from_name("crab")
    if runs[runID] == "off":
        crab_off = SkyCoord(76.1333, 22.0145, unit="deg")
    elif runs[runID] == "on":
        crab_off = SkyCoord(91.1333, 22.0145, unit="deg")
    elif runs[runID] == "allon":
        crab_off = SkyCoord.from_name("crab")
    psct_loc = EarthLocation(lat=31.6716989799*u.deg, lon=-110.951291195*u.deg, height=1268*u.m)
    a_x, b_x = (-0.014436381477488698, 1.093377089794729)
    a_y_1, b_y_1 = (-0.03798099601727016, 0.36721872391895594) # az < 180 deg
    a_y_2, b_y_2 = (-0.11408711008257931, 7.51779699821602) # az > 180 deg
    hillas_parameters = []
    cpu_s = []
    cpu_ns = []
    tacks = []
    events = []
    size = []
    frac_2 = []
    az_list = []
    el_list = []
    print("Reading TACK times...")
    for ev in range(n_events):
        ev = int(ev)
        reader.GetR1Event(ev, waveforms)
        tacks.append(reader.fTACK_time)
    print("Identifying flasher events...")
    timestamps = np.asarray(tacks)
    timestamps -= timestamps[0]
    timestamps = [time / 1_000_000_000 for time in timestamps]
    subruns = list_subruns(timestamps)
    opt_pers = []
    for subrun in subruns:
        opt_pers.append(scan_rates(10.0, subrun))
    flasher_events = get_flasher_events(subruns, opt_pers)
    non_flasher_events = [ev for ev in range(n_events) if ev not in flasher_events]
    tack_diff = tacks[-1] - tacks[0]
    tack_split = tack_diff / 2.0 + tacks[0]
    #on_off_ind = find_nearest(tacks, tack_split)
    tacks = []
    print("Calculating Hillas parameters. This might take a while...")
    for ev in non_flasher_events:
        ev = int(ev)
        reader.GetR1Event(ev, waveforms)
        reader.GetTimeStamp(ev)
        tack = reader.fTACK_time
        peak_position = np.argmax(waveforms, axis=1)
        charges = mcs.calculate_charge(waveforms, peak_position, n_samples)
        waveforms = apply_gains(waveforms)
        peaks = np.zeros((40, 40))
        for i, wf in enumerate(waveforms):
            peaks[grid_ind[i]%40, grid_ind[i]//40] = np.amax(wf[30:])
        charges = apply_gains(charges)
        no_charge = [val for val in charges if val == 0.0]
        if len(no_charge) > 750:
            continue
        noise = is_noise(peaks)
        if noise is True:
            continue
        cpu_s_single = reader.fCPU_s
        if tack <= tack_split:
            if runs[runID] == "on" or runs[runID] == "allon":                
                time = Time(cpu_s_single, format="unix")
                crab_azel = crab.transform_to(AltAz(obstime=time, location=psct_loc))
                az, el = (crab_azel.az.deg, crab_azel.alt.deg)
                delta_x = linear(az, a_x, b_x)
                delta_y = linear(el, a_y_1, b_y_1) if az < 180 else linear(el, a_y_2, b_y_2)
            elif runs[runID] == "off":
                time = Time(cpu_s_single, format="unix")
                crab_azel = crab_off.transform_to(AltAz(obstime=time, location=psct_loc))
                az, el = (crab_azel.az.deg, crab_azel.alt.deg)
                delta_x = linear(az, a_x, b_x)
                delta_y = linear(el, a_y_1, b_y_1) if az < 180 else linear(el, a_y_2, b_y_2)
        elif tack > tack_split:
            if runs[runID] == "on" or runs[runID] == "allon":
                time = Time(cpu_s_single, format="unix")
                crab_azel = crab_off.transform_to(AltAz(obstime=time, location=psct_loc))
                az, el = (crab_azel.az.deg, crab_azel.alt.deg)
                delta_x = linear(az, a_x, b_x)
                delta_y = linear(el, a_y_1, b_y_1) if az < 180 else linear(el, a_y_2, b_y_2)
            elif runs[runID] == "off":
                time = Time(cpu_s_single, format="unix")
                crab_azel = crab.transform_to(AltAz(obstime=time, location=psct_loc))
                az, el = (crab_azel.az.deg, crab_azel.alt.deg)
                delta_x = linear(az, a_x, b_x)
                delta_y = linear(el, a_y_1, b_y_1) if az < 180 else linear(el, a_y_2, b_y_2)
        image = clean_image(charges, grid_ind, mean_noise)
        image_size = [val for row in image for val in row if val > 5]
        charge_coords = [[i - 19.5 + 0.31 * ((i // 8) - 2) + delta_x, 
                          j - 19.5 + 0.31 * ((j // 8) - 2) + delta_y,
                          image[j, i]] 
                          for i in range(40) for j in range(40)]
        try:
            h_params = hillas(np.asarray(charge_coords).T)
        except:
            continue
        az_list.append(az)
        el_list.append(el)
        hillas_parameters.append(np.asarray(h_params))
        events.append(ev)
        size.append(np.nansum(image.flatten()))
        bright_2 = np.sum(heapq.nlargest(2, image.flatten()))
        frac_2.append(bright_2 / np.nansum(image.flatten()))
        cpu_s.append(reader.fCPU_s)
        cpu_ns.append(reader.fCPU_ns)
        tacks.append(tack)
    hillas_parameters = np.asarray(hillas_parameters)
    tack_diff = tacks[-1] - tacks[0]
    tack_split = tack_diff / 2.0 + tacks[0]
    on_off_ind = find_nearest(tacks, tack_split)
    if runs[runID] == "off":
        on = hillas_parameters[on_off_ind:]
        off = hillas_parameters[:on_off_ind]
    elif runs[runID] == "on":
        on = hillas_parameters[:on_off_ind]
        off = hillas_parameters[on_off_ind:]
    elif runs[runID] == "allon":
        on = hillas_parameters
        off = []

    with open(f"/data/user/bmode/crab_data/v0.3/on_data_run{runID}.txt", "w") as f:
        f.write("Run, Event, CPU_s, CPU_ns, TACK, Az, El, Size, Frac2, a, b, x, y, width, length, miss, dis, azwidth, alpha\n")
        nl = "\n"
        for i in range(len(on)):
            f.write(f"{runID}, {events[i]}, {cpu_s[i]}, {cpu_ns[i]}, {tacks[i]}, {az_list[i]}, {el_list[i]}, {size[i]}, {frac_2[i]}, {on[i][0]}, {on[i][1]}, {on[i][2]}, {on[i][3]}, {on[i][4]}, {on[i][5]}, {on[i][6]}, {on[i][7]}, {on[i][8]}, {on[i][9]}{nl}")

    with open(f"/data/user/bmode/crab_data/v0.3/off_data_run{runID}.txt", "w") as f:
        f.write("Run, Event, CPU_s, CPU_ns, TACK, Az, El, Size, Frac2, a, b, x, y, width, length, miss, dis, azwidth, alpha\n")
        nl = "\n"
        for i in range(len(off)):
            j = i + len(on)
            f.write(f"{runID}, {events[j]}, {cpu_s[j]}, {cpu_ns[j]}, {tacks[j]}, {az_list[j]}, {el_list[j]}, {size[j]}, {frac_2[j]}, {off[i][0]}, {off[i][1]}, {off[i][2]}, {off[i][3]}, {off[i][4]}, {off[i][5]}, {off[i][6]}, {off[i][7]}, {off[i][8]}, {off[i][9]}{nl}")
'''
