import os
import pwd

import argparse
import matplotlib as mpl
mpl.use("Agg")
from tqdm import tqdm

from psct_reader import WaveformArrayReader as Reader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run", help="Run number")
    parser.add_argument("-e", "--event", help="Starting event number")
    parser.add_argument("--raw", action="store_true", help="Don't apply gain calibration")
    parser.add_argument("--stop", help="Stopping event number")
    parser.add_argument("--step", help="step size")
    args = parser.parse_args()
    run = args.run
    event = args.event
    raw = args.raw
    stop = args.stop
    step = args.step
    username = pwd.getpwuid(os.getuid()).pw_name
    if username == "ctauser":
        DATADIR = "/data/local_outputDir"
        SAVEDIR = "/data/analysis_output/camera_images"
    else:
        DATADIR = "/data/wipac/CTA/target5and7data/runs_320000_through_329999"
        SAVEDIR = "/data/wipac/CTA/web/analysis_output/camera_images"
    try:
        os.mkdir(f"{SAVEDIR}/run{run}")
    except:
        pass
    reader = Reader(f"{DATADIR}/cal{run}.r1")
    if stop == None:
        event_list = [event]
    elif step == None:
        event_list = range(event, stop+1)
    else:
        event_list = range(event, stop+1, step)
    if raw is True:
        reader.gains = False
    print("Creating camera images:")
    for ev in tqdm(event_list):
        reader.get_event(ev)
        reader.plot_camera_image(clean=False, keep=True, directory=f"{SAVEDIR}/run{run}")

