from datetime import datetime
import os
import pwd
import sys
import re
from typing import List

import pandas as pd


def read_directory(dropboxdir) -> List:
    try:
        dropboxdir_files = os.listdir(dropboxdir)
    except Exception as ex:
        print("Unable to access dropbox directory.")
        sys.exit(ex)
    drops = []
    for line in dropboxdir_files:
        if re.search("drop$", line):
            drops.append(str(line))
    if drops == []:
        sys.exit("No new drops.")
    print(drops)
    return drops


def process_files(df: pd.DataFrame):
    for row in df.itertuples():
        ped = 328587
        print(row)
        run = int(row[1])
        event = int(row[2])
        raw = bool(row[3] == " True" or row[3] == 1)
        image = bool(row[4] == " True" or row[4] == 1)
        movie = bool(row[5] == " True" or row[5] == 1)
        heatmap = bool(row[6] == " True" or row[6] == 1)
        if image == True:
            try:
                if raw == True:
                    os.system(f"{PYTHONDIR}/python {SCRIPTDIR}/SCT_camera_plot.v2.py"
                              " "
                              f"-r {run} -e {event} --raw")
                else:
                    os.system(f"{PYTHONDIR}/python {SCRIPTDIR}/SCT_camera_plot.v2.py"
                              " "
                              f"-r {run} -e {event}")
            except Exception:
                print(f"Unable to make camera image of run {run}, "
                      f"event {event}.")
                pass
        if movie == True:
            try:
                os.system(f"{PYTHONDIR}/python {SCRIPTDIR}/SCT_make_movie.py"
                          " " f"-i {run} -p {ped} -e {event}")
            except Exception:
                print(f"Unable to make camera movie of run {run}, "
                      f"event {event}.")
                pass
        if heatmap == True:
            try:
                print(f"Creating interactive heatmap of run {run}, "
                      f"event {event} at {str(datetime.today())}")
                if raw == True:
                    os.system(f"{PYTHONDIR}/python {SCRIPTDIR}/SCT_make_heatmap.v2.py"
                              " " f"{run} {event} 0 0")
                else:
                    os.system(f"{PYTHONDIR}/python {SCRIPTDIR}/SCT_make_heatmap.v2.py"
                              " " f"{run} {event} 0 1")
            except Exception as ex:
                print(f"Unable to make interactive heatmap of run {run}, "
                      f"event {event}.")
                print(ex)
        else:
            print("Did not trigger any options!")
            print(heatmap)
            print(heatmap == True)

def read_files(drops: List) -> pd.DataFrame:
    df = pd.concat((pd.read_csv(f"{DROPBOXDIR}/{file_}") for file_ in drops))
    return df


def move_files(drops: List):
    try:
        for file_ in drops:
            os.rename(f"{DROPBOXDIR}/{file_}", f"{DROPBOXDIR}/archive/{file_}")
    except Exception as ex:
        sys.exit(ex)


if __name__ == "__main__":
    username = pwd.getpwuid(os.getuid()).pw_name
    if username == "ctauser":
        DROPBOXDIR = "/data/dropbox"
        PYTHONDIR = "/data/software/anaconda2/envs/sctcamsoft/bin"
        SCRIPTDIR = "/home/ctauser/CameraSoftware/trunk/analysis"
    else:
        DROPBOXDIR = "/data/wipac/CTA/dropbox"
        PYTHONDIR = "/home/bmode/anaconda2/envs/python_3/bin"
        SCRIPTDIR = "/home/bmode/analysis"
    print("Reading directory...")
    drops = read_directory(DROPBOXDIR)
    print("Reading files...")
    df = read_files(drops)
    print("Moving files to archive...")
    move_files(drops)
    print("Processing files...")
    process_files(df)
