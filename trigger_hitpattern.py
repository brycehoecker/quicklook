import os
import pwd
from typing import List, Tuple

import argparse
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from psct_reader import WaveformArrayReader
import hitpattern_parser as parser


def rm_repeat_triggers(data_tuple: Tuple) -> (List, List):
    """
    Render the trigger hitpattern data usable by removing repeated patterns from timestamp and data list
    
    Parameters
    ----------
    data_tuple: tuple
        Tuple containing trigger hitpatterns and their timestamps, as parsed by hitpattern_parser
    
    Returns
    -------
    trig_times: list
        List of trigger hitpattern timestamps with adjacent redundancies removed
    trig_data: list
        List of trigger hitpattern images with adjacent redundancies removed
    
    """
    
    trig_times, trig_data = data_tuple
    trig_times = list(trig_times)
    i = 0
    while i < len(trig_data) - 1:
        if np.array_equal(trig_data[i], trig_data[i+1]):
            trig_data.pop(i+1)
            trig_times.pop(i+1)
            continue
        i += 1
    trig_data.pop(0)
    trig_times.pop(0)
    trig_times = list(np.asarray(trig_times) - trig_times[0])
    return trig_times, trig_data

def correct_trig_images(trig_data: List) -> List:
    """
    Loop through trigger hitpattern images and apply a reflection about the y-axis for each module column
    
    Parameters
    ----------
    trig_data: list
        List of shape (20, 20) np.ndarray hitpattern images
    
    Returns
    -------
    corrected_trig_data: list
        List of corrected shape (20, 20) hitpattern images
    
    """
    
    corrected_trig_data = []
    for im in trig_data:
        for col in range(5):
            im[:, col*4:col*4 + 4] = np.fliplr(im[:, col*4:col*4 + 4])
        corrected_trig_data.append(im)
    return corrected_trig_data

def match_timestamps(data: np.ndarray, hits: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Find the indices where elements from a smaller array fall within a certain
    Euclidean distance of elements in a smaller array
    
    Parameters
    ----------
    data : np.ndarray
        Array of data timestamps
    hits : np.ndarray
        Array of hitpattern timestamps
    tolerance : float
        Acceptable tolerance such that |data[i] - hit[j]| < tolerance for matched pair (s_i, r_j)
    
    Returns
    -------
    indices : list
        List of index pairs, each pair has smallest spacing within tolerance, i.e. min{(s_i, r_j) < tolerance}
        
    """
    
    i = 0 # index for data
    j = 0 # index for hits
    toler = 0 # flag for marking the beginning of a segment within tolerance
    matched_events = []
    while i < len(data) and j < len(hits):
        valid = (hits[j] - data[i]) > 0 # check if current positions are valid
        if not valid:
            if toler == 1: # add previously recorded best pair if there is one
                matched_events.append(pair)
                toler = 0 # reset tolerable segment flag
            j += 1 # iterate the hits array
            continue
        tolerable = (hits[j] - data[i]) <= tolerance # check if current positions are within tolerance
        if not tolerable: # if valid but not within tolerance, iterate data array
            i += 1
            continue
        else:
            pair = (i, j) # if valid and within tolerance, record pair
            i += 1 # iterate data array
            toler = 1 # flag the beginning of a segment within tolerance
    return matched_events # if upon iterating data array, still valid, and still tolerable, best pair is updated

def plot_hitmap_image(image: np.ndarray, index: int, run: int, keep: bool=False, directory: str=None):
    """
    Plot an image of a hitmap. The image should be corrected before running through here.
    
    Parameters
    ----------
    image: np.ndarray
        2D numpy array of a hitpattern image. Should have shape (20, 20).
    index: int
        Index in hitpattern array, used for title and filename.
    run: int
        Run number.
    keep: bool
        Should the image be saved? Defaults to saving in the current directory.
    directory: str
        Path to image storage directory.
    
    """
    
    fig = plt.gcf()
    if fig.get_size_inches().all() == np.array([18., 15.]).all():
        plt.close(fig)
        fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.04, hspace=0.04)

    for i in range(5):
        for j in range(5):
            sub_image = image[i*4:i*4+4, j*4:j*4+4]
            ax = plt.subplot(gs[i, j])
            sub_image = sub_image[::-1, :]
            c = ax.pcolormesh(sub_image, vmin=0, vmax=1, cmap="viridis")
            ax.axis("off")
            ax.set_aspect("equal")

    fig.subplots_adjust(right=0.8, top=0.9, bottom=0.1)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(c, cax=cbar_ax)
    cbar.set_label("Trigger Hitmap", rotation=270,
                   size=24, labelpad=24)
    cbar_ax.tick_params(labelsize=20)
    fig.suptitle(f"Hitpattern Index {index}", fontsize=30)
    if keep == True:
        if directory == None:
            fig.savefig(f"hitpattern{run}_index{index}.png")
        else:
            fig.savefig(f"{directory}/hitpattern{run}_index{index}.png")

if __name__ == "__main__":
    sns.set_style("whitegrid")
    mpl.rc("figure", figsize=(16, 9))
    mpl.rc("axes", titlesize=24, labelsize=22)
    mpl.rc("legend", fontsize=18)
    mpl.rc("xtick", labelsize=16)
    mpl.rc("ytick", labelsize=16)
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-r", "--run", help="run number")
    argparser.add_argument("-i", "--index", help="index number")
    args = argparser.parse_args()
    run = int(args.run)
    index = int(args.index)
    username = pwd.getpwuid(os.getuid()).pw_name
    if username == "ctauser":
        DATADIR = "/data/local_outputDir"
        SAVEDIR = f"/data/analysis_output/hitpattern_images/run{run}"
    else:
        DATADIR = "/data/wipac/CTA/target5and7data/runs_320000_through_329999"
        SAVEDIR = f"/data/wipac/CTA/web/analysis_output/hitpattern_images/run{run}"
    try:
        os.mkdir(SAVEDIR)
    except:
        pass
    hitfile = f"{DATADIR}/hitpattern{run}.txt"
    trig_times, trig_data = rm_repeat_triggers(parser.parse_file(hitfile))
    trig_data = correct_trig_images(trig_data)
    image = trig_data[index]
    plot_hitmap_image(image, index, run, keep=True, directory=SAVEDIR)
    