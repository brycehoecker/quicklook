import re

import numpy as np
import pandas as pd

rx_dict = {
    "time": re.compile(r"Current time: (?P<time>.*)\n"),
    "hitpattern": re.compile(r"(?P<hitpattern>\d+ \d+ \d+ \d+ \d+)\n"),
    }

def _parse_line(line):
    """
    Parse each line in the hitpattern file

    Parameters
    ----------
    line : str
        line from the file

    Returns
    -------
    key : str
        dictionary key for match
    match : re.MatchObject
        object returned by regular expression matching

    """

    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    return None, None

def parse_file(filepath):
    """
    Parse trigger hitpattern as taken by pSCT Raspberry Pi

    Parameters
    ----------
    filepath : str
        Filepath for hitpattern file to be parsed

    Returns
    -------
    data : pd.DataFrame
        DataFrame containing times and associated hitpatterns

    """
    data = []
    time = []
    hit = []
    count = 0
    with open(filepath, "r") as file:
        line = file.readline()
        while line:
            key, match = _parse_line(line)
            if key == "time":
                time.append(pd.to_datetime(match.group("time"), format="%m/%d/%y %H:%M:%S.%f %Z").timestamp())
            if key == "hitpattern":
                row = match.group("hitpattern")
                hit.append([int(s) for s in "".join(row.split())])
                count += 1
            if count == 20:
                count = 0
                trigger_image = np.asarray(hit, dtype=np.float32)
                data.append(trigger_image)
                hit = []
            line = file.readline()
    return time, data

if __name__ == "__main__":
    pass
