#!/usr/bin/env python
from collections import namedtuple
from distutils.util import strtobool
import logging
logging.basicConfig(level=logging.DEBUG)
import sys
import time
from typing import List, Tuple							
#from astroplan import download_IERS_A
#download_IERS_A()
#from astropy.coordinates import AltAz, EarthLocation, SkyCoord
#from astropy.table import Table
#from astropy.time import Time
#from astropy import units as u
#from astropy.utils import iers
#iers.Conf.iers_auto_url.set('ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all')
#import numpy as np
#from numba import njit, prange
import pandas as pd
#from scipy.sparse import csr_matrix
#from scipy.sparse import lil_matrix
#from scipy.spatial import cKDTree as KDTree
#from datetime import datetime
#import heapq
#from itertools import combinations
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#import matplotlib.lines as lines
#from matplotlib.patches import Ellipse

#ABOVE IMPORTS ARE THEORETICALLY WORKING

#BELOW IMPORTS ARE NOT WORKING RIGHT NOW
import utils
from utils import TrackingError
from astroplan import Observer
from apply_gains import apply_gains


try:
    import target_io
except Exception as ex:
    print("TargetIO must be installed in order to use the pSCT data reader.")
    sys.exit()
