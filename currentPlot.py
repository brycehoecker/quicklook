import target_io
import target_driver
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import datetime
import time
import sys
import os
import run_control
import pickle
import pandas as pd
import matplotlib.dates as mdates
import argparse
import datetime

#### How to Use ####
# python currentPlot.py -i RunNumber


# Create a list of module numbers present in the currents file
def list_modules(data):
    module_numbers = set()
    for sensor in data.keys():
        if sensor == 'timestamp_start':
            continue
        if sensor == 'timestamp_end':
            continue
        module_numbers.add(sensor.split('_')[0])
    return module_numbers


# calculates row and column of pixel in a module from the pixel number
def pixel_row_col_coords(index):
    # Convert bits 1, 3 and 5 to row
    row = 4*((index & 0b100000) > 0) + 2*((index & 0b1000) > 0) + 1*((index & 0b10) > 0)
    # Convert bits 0, 2 and 4 to col
    col = 4*((index & 0b10000) > 0) + 2*((index & 0b100) > 0) + 1*((index & 0b1) > 0)
    return (row, col)


# Assigns current value to appropriate location in module
def arrange_pixels(values_by_pixel):
    values_by_position = np.zeros([8,8])
    for i, val in enumerate(values_by_pixel):
        row, col = pixel_row_col_coords(i)
        values_by_position[row,col] = val
    return values_by_position
    

# Creates a dictionary with key:module number and value:list of currents for a given current reading
def get_currents(data,modules,reading):
     
    # Get currents for each module
    modCurrents = {}
    for mod in modules:
        mod = int(mod)
        currents_by_pixel = np.full(64, np.nan) #Initialize with NANs so that they do not show up in final image
        for pixel in range(npixels):
            value = data[str(mod)+'_pixel'+str(pixel)][reading]
            if value < 2000:
                if value > 0:
                    currents_by_pixel[pixel] = value
        currents_by_position = arrange_pixels(currents_by_pixel)
        
        # rotate modules in odd columns
        loc = modPos[mod]
        if loc[1]%2 == 0: #FIXME should actually be odd columns which rotate
            currents_by_position = np.rot90(currents_by_position, k=2)
        
        # Add currents to dictionary
        modCurrents[mod] = currents_by_position
    
    return modCurrents

#@njit
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
            if s > 3000: #FIXME Cutoff depends on if you are cleaning or not
                image[j, i] = charge[i+2, j+2]

    return image

#@njit
def hillas(charge_coords):
    """Calculates the Hillas parameters for an event."""
    x = 0
    y = 0
    x2 = 0
    y2 = 0
    xy = 0
    CHARGE = 0
    #print(charge_coords.shape)
    CHARGE = np.nansum(charge_coords[2])
    print(f"Size: {CHARGE}")
    x = np.nansum(charge_coords[0] * charge_coords[2])
    print(x)
    y = np.nansum(charge_coords[1] * charge_coords[2])
    x2 = np.nansum(charge_coords[0] ** 2 * charge_coords[2])
    y2 = np.nansum(charge_coords[1] ** 2 * charge_coords[2])
    xy = np.nansum(charge_coords[0] * charge_coords[1] * charge_coords[2])

    x /= CHARGE
    print(x)
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


if __name__ == "__main__":
    nmodules=25
    npixels=64   
  
    # Module positions are for use with pcolormesh - cannot be used with imshow

    # Module Positions (Skyview)
    #modPos = {  4:(0,4),   5:(0,3),   1:(0,2),   3:(0,1), 2:(0,0),
    #          103:(1,4), 125:(1,3), 126:(1,2), 106:(1,1), 9:(1,0),
    #          119:(2,4), 108:(2,3), 110:(2,2), 121:(2,1), 8:(2,0),
    #          115:(3,4), 123:(3,3), 124:(3,2), 112:(3,1), 7:(3,0),
    #          100:(4,4), 111:(4,3), 114:(4,2), 107:(4,1), 6:(4,0)}
 
    # Module Positions (Camera view)
    modPos = {  4:(0,0),   5:(0,1),   1:(0,2),   3:(0,3), 2:(0,4),
              103:(1,0), 125:(1,1), 126:(1,2), 106:(1,3), 9:(1,4),
              119:(2,0), 108:(2,1), 110:(2,2), 121:(2,3), 8:(2,4),
              115:(3,0), 123:(3,1), 124:(3,2), 112:(3,3), 7:(3,4),
              100:(4,0), 111:(4,1), 114:(4,2), 107:(4,3), 6:(4,4)}
 

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="run number", help="Run number for a data run or thresh scan in which currents were recorded")
    parser.add_argument("-p", metavar="Hillas Parameters", help="Calculates and plots hillas parameters")
    args = parser.parse_args()
    ID = args.i
    p = args.p

    dataDir = '/data/local_outputDir/'
    outputDir = '/data/analysis_output/current_plots'

    filename = dataDir + str(ID) + '_currents.txt'
    data = pd.read_csv(filename)
    shape = data.shape
    times = [data['timestamp_start'][i] for i in range(shape[0])]
    times = mdates.num2date(mdates.datestr2num(times))
    date = datetime.date.fromisoformat(data['timestamp_start'][1][:10])

    modules = list_modules(data)
    
    #Get currents and plot
    for reading in range(shape[0]):
        currents = get_currents(data,modules,reading)
        
        #Set up figure
        CurrentFig = plt.figure('Currents Heat Map', (18.,15.))
        gs = gridspec.GridSpec(6,6)
        gs.update(wspace=0.04, hspace=0.04)
        
        max_list = []
        for mod in modules:
            mod=int(mod)
            m = max(modPos[mod])
            max_list.append(m)
        maximum = max(max_list)

        # Create subfigures by module
        for mod in modules:
            mod = int(mod)
            loc = modPos[mod]
            ax = plt.subplot(gs[loc])
            c = ax.pcolormesh(currents[mod], vmin=0, vmax=800) #FIXME input max is handpicked - change as necessary!
            #ax.set_title("Mod {}".format(mod))
            ax.axis('off')
            ax.set_aspect('equal')

        CurrentFig.subplots_adjust(right=0.8,top=0.9,bottom=0.1)
        #CurrentFig.colorbar(c, ax=ax)
        axes = plt.subplot2grid((6,6),(0,5), rowspan=5)
        #plt.colorbar(c, cax=axes)
        cbar = CurrentFig.colorbar(c, cax=axes)
        cbar.set_label('Pixel Current', rotation=270,size=20,labelpad=24)
        #cbar_ax4 = CurrentFig.add_axes([0.85, 0.15, 0.05, 0.7])
        CurrentFig.suptitle('Run {} Date {}'.format(args.i, date), fontsize=24)
        CurrentFig.savefig("{}/{}_CurrentHeatMap_reading{}.png".format(outputDir,ID,reading))
        plt.clf()

