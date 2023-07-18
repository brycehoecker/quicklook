from collections import namedtuple
from distutils.util import strtobool
import logging
#logging.basicConfig(level=logging.DEBUG)
import sys
import time

from astroplan import Observer
#from astroplan import download_IERS_A
#download_IERS_A()
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
#from astropy.utils import iers
#iers.Conf.iers_auto_url.set('ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all')
import numpy as np
from numba import njit, prange
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree as KDTree

import utils
from utils import TrackingError
from apply_gains import apply_gains

try:
    import target_io
except Exception as ex:
    print("TargetIO must be installed in order to use the pSCT data reader.")
    sys.exit()

class DataFormatError(Exception): pass

class RawDataException(Exception): pass

class HillasError(Exception): pass


class WaveformArrayReader:
    """
    Class built on top of target_io.WaveformArrayReader to provide access to
    low level pSCT event data. Interfacing with the data at the TargetIO level
    is fraught with pathological differences in syntax, and the data files do
    not automatically contain derived information. This module greatly
    simplifies pSCT data analysis.

    :param tracking_file: str = None, path to a file containing telescope tracking information corresponding to the data run
    :param observing_mode: str = "ON/OFF", corresponds to particular observing mode, although currently only ON/OFF is supported
    :param source: astropy.SkyCoord or None, if there was a source tracked, include a SkyCoord object for it
    """

    def __init__(self, file_, tracking_file=None, observing_mode="ON/OFF", source=None):
        try:
            if type(file_) != str:
                raise TypeError
            self.file_ = file_
            self.__data_format = self.file_.split(".")[-1]
            self.__run = int(self.file_.split("/")[-1].split(".")[0].lstrip("runcal"))
            logging.debug(f"Data is a .{self.__data_format} file")
            if not (self.__data_format == "r1" or self.__data_format == "fits"):
                raise DataFormatError
            self.__reader = target_io.WaveformArrayReader(file_)
            self.__tracking = pd.read_csv(tracking_file) if tracking_file is not None else None
            self.__pointing = {}
            self.observing_location = EarthLocation(lat="31d40m30.4s", lon="-110d57m7.2s", height=1268*u.m)
            self.observer = Observer(location=self.observing_location, name="VERITAS")
            self.__current_pointing_time = 0
            self.__ev = 0
            self.__n_pixels = self.__reader.fNPixels
            self.__n_samples = self.__reader.fNSamples
            self.__n_events = self.__reader.fNEvents
            self.__cpu_s = 0
            self.__cpu_ns = 0
            self.__tack = 0
            self.__block = -1
            self.__block_phase = -1
            if self.__data_format == "r1":
                self.__wfs = np.zeros((self.__n_pixels, self.__n_samples), dtype=np.float32)
            else:
                self.__wfs = np.zeros((self.__n_pixels, self.__n_samples), dtype=np.ushort)
            self.__peak_positions = None
            self.__grid_ind = utils.get_grid_ind()
            self.__mean_noise = utils.get_mean_noise()
            self.__noise_thresh = 1.5
            self.__clean_thresh = 2.0
            self.__asic_island_cleaning = False
            self.__largest_island = False
            self.__correct_splits = False
            self.__hillas_for_plot = False
            self.__gains = True
            self.__dead_pixels_zero = utils.get_dead_pixels()
            self.__dead_pixels_nan = np.asarray([1. if val == 1. else np.NaN
                                                 for val in self.__dead_pixels_zero])
            self.__charges = np.zeros(self.__n_pixels)
            self.__clean_image = np.zeros((40, 40))
            self.__pix_x = np.repeat(np.arange(40), 40)
            self.__pix_y = np.tile(np.arange(40), 40)
            self.__pixel_centers = np.column_stack([self.__pix_x, self.__pix_y])
            self.__kdtree = KDTree(self.__pixel_centers)
            self.__neighbors = self.__get_neighbors(self.__kdtree)
            self.__nan = False
            self.__hillas = None
            self.__source = source # If the source is supplied, should be astropy SkyCoord in ICRS coordinates
            if observing_mode == "ON/OFF":
                self.__is_ON = False
                self.__is_OFF = False
            if tracking_file is not None and source is not None:
                self.get_event(0)
                self.__current_region = None
                self.pointing
                self.__update_culmination_time()


        except TypeError as ex:
            print("Error: Must input a valid string for the file name!")
            print(ex)
            sys.exit()
        except DataFormatError as ex:
            print("Data file must be either a .fits file or a .r1 file to proceed.")
            sys.exit()

    def __get_neighbors(self, kdtree):
        max_neighbors = 8 # this is correct for the diagonal case that we want
        norm = 2
        radius = 1.95
        neighbors = lil_matrix((1600, 1600), dtype=bool)
        for i, pixel in enumerate(kdtree.data): # walking through the list of pixels
            distances, neighbor_candidates = kdtree.query(pixel, k=max_neighbors+1, p=norm) # this gives a good guess
            distances = distances[1:] # remove self-reference
            neighbor_candidates = neighbor_candidates[1:]
            inside_max_distance = distances < radius * np.min(distances) # collect the ones that are actually neighbors
            neighbors[i, neighbor_candidates[inside_max_distance]] = True # then record that
        return neighbors.tocsr()

    def __update_region(self):
        if self.__source is None:
            logging.warning("Pointing cannot be ON or OFF source without a source target.")
            return None
        if not isinstance(self.__source, SkyCoord):
            logging.warning("The source must be an astropy SkyCoord.")
            raise RuntimeError
        elif self.__is_slewing:
            self.__is_ON = False
            self.__is_OFF = False
            self.__current_region = None
        elif (np.abs(self.__source.ra.deg - self.__pointing["ra"]) < 1.
                and np.abs(self.__source.dec.deg - self.__pointing["dec"]) < 1.):
            self.__is_ON = True
            self.__is_OFF = False
            if self.__current_region != "ON":
                self.__update_culmination_time()
                self.__current_region = "ON"
        elif (np.abs(self.__source.ra.deg - self.__pointing["ra"]) >= 1.
                or np.abs(self.__source.dec.deg - self.__pointing["dec"]) >= 1.):
            self.__is_ON = False
            self.__is_OFF = True
            if self.__current_region != "OFF":
                self.__update_culmination_time()
                self.__current_region = "OFF"
        else:
            logging.error("Something impossible happened.")
            raise RuntimeError

        if not self.__is_slewing and self.__slew_flag:
            self.__update_culmination_time()

    def __update_culmination_time(self):
        ra = self.__pointing["ra"]
        dec = self.__pointing["dec"]
        target = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")
        observing_time = Time(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.__current_pointing_time)),
                              scale="utc")
        culmination = self.observer.target_meridian_transit_time(observing_time, target, which="nearest")
        self.__culmination_time = culmination.unix
        logging.debug("Culmination time has been updated to reflect new pointing target.")

    @property
    def data_format(self):
        """
        Returns the file extension string corresponding to the given data
        format

        :return: data_format
        :rtype: str
        """
        return self.__data_format

    @property
    def culmination_time(self):
        """
        Returns the culmination time for the night of the given data run if
        it exists. Otherwise logs an error that should not ever occur.

        :return: culmination_time
        :rtype: float
        """

        if self.__tracking is not None and self.__source is not None:
            return self.__culmination_time
        else:
            logging.error("No.")

    @property
    def run(self):
        """
        Returns the run number.

        :return: run
        :rtype: int
        """

        return self.__run

    @property
    def n_pixels(self):
        """
        Return the number of pixels.

        :return: n_pixels
        :rtype: int
        """

        return self.__n_pixels

    @property
    def n_samples(self):
        """
        Return the number of samples

        :return: n_samples
        :rtype: int
        """

        return self.__n_samples

    @property
    def n_events(self):
        """
        Return the number of events

        :return: n_events
        :rtype: int
        """

        return self.__n_events

    @property
    def ev(self):
        """
        Return the current event number.

        :return: ev
        :rtype: int
        """

        return self.__ev

    @property
    def wfs(self):
        """
        Return the waveforms, if they an event is loaded.

        :return: wfs
        :rtype: np.ndarray
        """

        if np.array_equal(self.__wfs, np.zeros((self.__n_pixels, self.__n_samples), dtype=np.float32)):
            logging.warning("No event selected, waveform data not available.")
            return None
        if self.__gains is True:
            return np.float32(apply_gains(self.__wfs))
        else:
            return self.__wfs

    @property
    def cpu_s(self):
        """
        Return the CPU time to the second, if the event is loaded, in Unix time.

        :return: cpu_s
        :rtype: int
        """

        if self.__cpu_s != 0:
            return self.__cpu_s
        else:
            logging.warning("No event selected, CPU s time data not available.")
            return None

    @property
    def cpu_ns(self):
        """
        Return the subsecond cpu time, if the event is loaded, in Unix time.

        :return: cpu_ns
        :rtype: int
        """

        if self.__cpu_ns != 0:
            return self.__cpu_ns
        else:
            logging.warning("No event selected, CPU ns time data not available.")
            return None

    @property
    def tack(self):
        """
        Return the TACK time, if the event is loaded, in backplane clock time.

        :return: tack
        :rtype: int
        """

        if self.__tack != 0:
            return self.__tack
        else:
            logging.warning("No event selected, TACK time data not available.")
            return None

    @property
    def block(self):
        """
        Return the block, if the event is loaded.

        :return: block
        :rtype: int
        """

        if self.__block != -1:
            return self.__block
        else:
            logging.warning("No event selected, block data not available.")
            return None

    @property
    def block_phase(self):
        """
        Return the combined block and phase information, if the event is loaded.

        :return: block_phase
        :rtype: int
        """

        if self.__block_phase != -1:
            return self.__block_phase
        else:
            logging.warning("No event selected, block phase data not available.")
            return None

    @property
    def pointing(self):
        """
        Calculate the pointing information for the current pointing time. Also
        helps track slewing information and ON/OFF source status.
        """
        self.__slew_flag = False
        if self.__tracking is None:
            logging.error("Positioner data is not available without valid positioner log.")
            raise RuntimeError
        elif self.__cpu_s == 0:
            logging.warning("No event selected, telescope pointing not available.")
            return None
        elif np.abs(self.__cpu_s - self.__current_pointing_time) < 1.0:
            return self.__pointing
        else:
            try:
                self.__current_pointing_time = self.__cpu_s
                utc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(self.__current_pointing_time))
                self.__pointing["az"] = utils.pointing_from_utc(utc_time,
                                                                "current_position_az",
                                                                self.__tracking)
                self.__pointing["el"] = utils.pointing_from_utc(utc_time,
                                                                "current_position_el",
                                                                self.__tracking)
                self.__pointing["ra"] = utils.pointing_from_utc(utc_time,
                                                                "current_RA",
                                                                self.__tracking)
                self.__pointing["dec"] = utils.pointing_from_utc(utc_time,
                                                                "current_Dec",
                                                                self.__tracking)
                self.__pointing["on_source"] = utils.pointing_from_utc(utc_time, "is_on_source", self.__tracking)
                self.__is_slewing = not self.__pointing["on_source"]
                self.__update_region()
                if self.__is_slewing and self.__slew_flag is False:
                    self.__slew_flag = True
                return self.__pointing
            except TrackingError:
                logging.error("Unable to find relevant tracking data. Might not exist. Exiting...")
                sys.exit()

    @property
    def is_slewing(self):
        """
        Returns whether the telescope was slewing at the time of the current event.

        :return: is_slewing
        :rtype: bool
        """

        return self.__is_slewing

    @property
    def is_ON(self):
        """
        Returns whether the telescope is currently tracking the ON source
        position.

        :return: is_ON
        :rtype: bool
        """

        return self.__is_ON

    @property
    def is_OFF(self):
        """
        Returns whether the telescope is currently tracking the OFF source
        position.

        :return: is_OFF
        :rtype: bool
        """

        return self.__is_OFF

    @property
    def charges(self):
        """
        Calculates the integrated charge for the entire event using utils
        module, if an event is selected. Will log a warning and return None
        if called on with raw data file ("fits"). Applies gains if this
        class property is set to True.

        :return: charges
        :rtype: np.ndarray
        """

        if np.array_equal(self.__wfs, np.zeros((self.__n_pixels, self.__n_samples), dtype=np.float32)):
            logging.warning("No event selected, waveform data not available for charge calculation.")
            return None
        elif self.__data_format == "fits":
            logging.warning("Raw data type incompatible with charge calculation. Use calibrated data.")
            return None
        elif np.array_equal(self.__charges, np.zeros(self.__n_pixels)):
            self.__charges = utils.calculate_charge(self.__wfs, self.__peak_positions, self.__n_samples)
        if self.__gains is True:
            return apply_gains(self.__charges)
        else:
            return self.__charges

    @property
    def image(self):
        """
        Populates the raw data image in a 40 by 40 numpy array. Not currently
        designed to allow for a less than full camera run. Will calculate
        charges if they have not already been calculated.

        :return: image
        :rtype: np.ndarray
        """

        if np.array_equal(self.__image, np.zeros((40, 40))):
            for i, val in enumerate(self.charges):
                self.__image[self.__grid_ind[i]//40, self.__grid_ind[i]%40] = val
        return self.__image


    @property
    def clean_image(self):
        """
        Cleans the image of the current event using utils, based on the values
        of the asic_island_cleaning and largest_island properties. Returns the
        cleaned 40 by 40 numpy array.

        :return: clean_image
        :rtype: np.ndarray
        """

        if np.array_equal(self.__clean_image, np.zeros((40, 40))):
            if self.__asic_island_cleaning is True:
                self.__clean_image = utils.clean_image(self.charges, self.__grid_ind, self.__mean_noise,
                                                    self.__dead_pixels_zero, nan=self.__nan, thresh=self.__clean_thresh)
            elif self.__largest_island is True:
                self.__clean_image = utils.clean_image(self.charges, self.__grid_ind, self.__mean_noise,
                                                    self.__dead_pixels_zero, nan=False, thresh=self.__clean_thresh)
                mask = (self.__clean_image != np.zeros((40, 40))).flatten()
                #self.__clean_image = utils.asic_island_cleaning(self.__clean_image)
                labels = utils.largest_island(mask, self.__neighbors.indices, self.__neighbors.indptr)
                largest_island = labels == np.argmax(np.bincount(labels[labels > 0]))
                self.__clean_image *= largest_island.reshape((40, 40))
            else:
                self.__clean_image = utils.clean_image(self.charges, self.__grid_ind, self.__mean_noise,
                                                    self.__dead_pixels_zero, nan=self.__nan, thresh=self.__clean_thresh)
        return self.__clean_image

    @property
    def hillas(self):
        """
        Calculates the Hillas parameterization of the current event and
        returns a psct_reader.Hillas object.

        :return: hillas
        :rtype: Hillas
        """

        if self.__hillas is None:
            self.__hillas = Hillas()
            delta_x = utils.get_delta_x(self.__cpu_s, self.__culmination_time)
            delta_y = utils.get_delta_y(self.__cpu_s, self.__culmination_time)
            if np.nansum(self.clean_image) == 0. or len(self.clean_image[self.clean_image > 0]) <= 2:
                raise HillasError
            hillas_list = utils.hillas(utils.get_charge_coords(self.clean_image,
                                                               delta_x,
                                                               delta_y,
                                                               for_plot=self.__hillas_for_plot))
            self.__hillas.a = hillas_list[0]
            self.__hillas.b = hillas_list[1]
            self.__hillas.x = hillas_list[2]
            self.__hillas.y = hillas_list[3]
            self.__hillas.width = hillas_list[4]
            self.__hillas.length = hillas_list[5]
            self.__hillas.miss = hillas_list[6]
            self.__hillas.dis = hillas_list[7]
            self.__hillas.azwidth = hillas_list[8]
            self.__hillas.alpha = hillas_list[9]
            self.__hillas.psi = hillas_list[10]
            self.__hillas.size = np.nansum(self.clean_image)
            return self.__hillas
        else:
            return self.__hillas

    @property
    def peak_positions(self):
        """
        Return the current peak positions.

        :return: peak_positions
        :rtype: np.ndarray
        """

        return self.__peak_positions

    @property
    def noise_thresh(self):
        """
        Returns the current value of noise threshold in p.e.

        :return: noise_thresh
        :rtype: float
        """

        return self.__noise_thresh

    @noise_thresh.setter
    def noise_thresh(self, noise_thresh):
        """
        Sets the value of the noise_threshold. Must be greater than zero.
        """

        assert noise_thresh > 0.
        self.__noise_thresh = noise_thresh

    @property
    def clean_thresh(self):
        """
        Returns the current value of cleaning threshold multiplier.

        :return: clean_thresh
        :rtype: float
        """

        return self.__clean_thresh

    @clean_thresh.setter
    def clean_thresh(self, clean_thresh):
        assert clean_thresh > 0.
        self.__clean_thresh = clean_thresh

    @property
    def hillas_for_plot(self):
        """
        Returns the boolean for whether Hillas figures are overlaid on
        image plots. This does not generate particularly accurate overlays yet.

        :return: hillas_for_plot
        :rtype: bool
        """

        return self.__hillas_for_plot

    @hillas_for_plot.setter
    def hillas_for_plot(self, hillas_for_plot):
        assert hillas_for_plot == True or hillas_for_plot == False
        self.__hillas_for_plot = hillas_for_plot

    @property
    def gains(self):
        """
        Returns whether or not gain calibration is applied to charges and
        waveforms, which propagates into Hillas calculations and image plots.

        :return: gains
        :rtype: bool
        """

        return self.__gains

    @gains.setter
    def gains(self, gains):
        assert gains is True or gains is False
        self.__gains = gains

    @property
    def asic_island_cleaning(self):
        """
        Deprecated (I think). Returns whether ASIC island cleaning is
        performed.

        :return: asic_island_cleaning
        :rtype: bool
        """

        return self.__asic_island_cleaning

    @asic_island_cleaning.setter
    def asic_island_cleaning(self, asic_island_cleaning):
        assert asic_island_cleaning == True or asic_island_cleaning == False
        self.__asic_island_cleaning = asic_island_cleaning

    @property
    def nan(self):
        """
        Determines whether dead pixels are replaced with np.NaN values.

        :return: nan
        :rtype: bool
        """

        return self.__nan

    @nan.setter
    def nan(self, nan):
        assert nan is True or nan is False
        self.__nan = nan

    @property
    def largest_island(self):
        """
        Determines whether only the largest signal island is retained during
        image cleaning.

        :return: largest_island
        :rtype: bool
        """

        return self.__largest_island

    @largest_island.setter
    def largest_island(self, largest_island):
        assert largest_island is True or largest_island is False
        self.__largest_island = largest_island

    @property
    def correct_splits(self):
        """
        Determines whether waveform splits are detected, and then afterward
        corrected.

        :return: correct_splits
        :rtype: bool
        """

        return self.__correct_splits

    @correct_splits.setter
    def correct_splits(self, correct_splits):
        assert correct_splits is True or correct_splits is False
        self.__correct_splits = correct_splits

    @property
    def is_noise(self):
        """
        Calculates whether an event is likely to be triggered on electronics
        noise from the FEE modules.

        :return: is_noise
        :rtype: bool
        """

        if np.array_equal(self.__wfs, np.zeros((self.__n_pixels, self.__n_samples), dtype=np.float32)):
            logging.warning("No event selected, waveform data not available for peak identification.")
            return None
        logging.debug("Currently excluding first 32 ns for use with utils.is_noise")
        peaks = np.amax(self.wfs[:, 32:], axis=1)
        peak_image = np.zeros((40, 40))
        for i, val in enumerate(peaks):
            peak_image[self.__grid_ind[i]//40, self.__grid_ind[i]%40] = val
        return utils.is_noise(peak_image, thresh=self.__noise_thresh)

    def get_event(self, ev, interpolate=True):
        """
        Gets the event from the data file using target_io.WaveformArrayReader.
        Distinguishes between raw and calibrated data files. Empties out
        information from previously loaded event. Interpolates and corrects
        splits only for calibrated data.

        :param ev: int, event number to get
        :param interpolate: bool = True, whether or not to interpolate for removed spikes that have been tagged
        """

        ev = int(ev)
        self.__ev = ev
        if self.__data_format == "r1":
            self.__reader.GetR1Event(self.__ev, self.__wfs)
            self.__charges = np.zeros(self.__n_pixels)
            self.__image = np.zeros((40, 40))
            self.__clean_image = np.zeros((40, 40))
            self.__hillas = None
            self.__reader.GetBlock(self.__ev)
            self.__block = self.__reader.fBlock
            self.__block_phase = self.__reader.fPhase
        else:
            self.__reader.GetR0Event(ev, self.__wfs)
            self.__charges = np.zeros(self.__n_pixels)
            self.__reader.GetTimeStamp(ev)
            self.__cpu_s = self.__reader.fCPU_s
            self.__cpu_ns = self.__reader.fCPU_ns
            self.__tack = self.__reader.fTACK_time
            self.__reader.GetBlock(self.__ev)
            self.__block = self.__reader.fBlock
            self.__block_phase = self.__reader.fPhase
            self.__peak_positions = np.argmax(self.__wfs, axis=1)
            raise RawDataException
        self.__reader.GetTimeStamp(ev)
        self.__cpu_s = self.__reader.fCPU_s
        self.__cpu_ns = self.__reader.fCPU_ns
        self.__tack = self.__reader.fTACK_time
        if interpolate:
            self.__wfs = np.float32(self.interp_waveforms(self.__wfs, self.__n_samples))
        self.__peak_positions = np.argmax(self.__wfs, axis=1)
        if self.correct_splits == True:
            for i, pix in enumerate(self.__wfs):
                if utils.idef_split(pix, self.__block_phase):
                    phase_points = [i for i in range(128) if (i + self.__block_phase) % 32 == 0]
                    pix = utils.correct_split(pix, phase_points)
                    self.__wfs[i] = pix


    @staticmethod
    @njit
    def interp_waveforms(waveforms: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Numba-based static method for performing waveform interpolation.
        Returns the corrected waveforms

        :param waveforms: np.ndarray, event waveform array
        :param n_samples: int, number of samples, needs to be given as an argument because this is a static method
        :return: wfs_interp
        :rtype: np.ndarray
        """

        wfs = np.zeros(waveforms.shape, dtype=np.float64)
        count = 0
        for pix in prange(waveforms.shape[0]):
            for sam in prange(waveforms.shape[1]):
                val = waveforms[pix, sam]
                if val < -699.:
                    wfs[pix, sam] = np.NaN
                    count += 1
                else:
                    wfs[pix, sam] = val
        if count == 0:
            return wfs
        inds = np.arange(n_samples)
        wfs_interp = np.zeros(wfs.shape)
        for pix in prange(wfs.shape[0]):
            wf = wfs[pix, :]
            good = np.where(np.isfinite(wf))
            if len(wf[good]) == 0:
                wfs_interp[pix, :] = np.zeros(n_samples)
                continue
            f = np.interp(inds.astype(np.float64), inds[good].astype(np.float64), wf[good])
            wf_interp = np.where(np.isfinite(wf), wf, f[inds])
            wfs_interp[pix, :] = wf_interp
        return wfs_interp

    def plot_camera_image(self, clean=True, nan=True, thresh=2., keep=False, directory=None):
        """
        Plot a camera image using matplotlib. Allows options for clean or raw
        camera images, whether to replace dead pixels with np.Nan, what
        threshold to use for image cleaning, whether to save the plot, and
        whether to specify a different directory.

        :param clean: bool = True, plot raw or cleaned camera image
        :param nan: bool = True, plot dead pixels with np.Nan, not implemented
        :param thresh: float = 2., set cleaning threshold, not implemented
        :param keep: bool = False, determine whether to keep plot
        :param directory: str or None, provide a different save directory path
        """

        if clean:
            utils.plot_camera_image(self.clean_image, self.__run, self.__ev, keep=keep, directory=directory)
        else:
            utils.plot_camera_image(self.image, self.__run, self.__ev, keep=keep, directory=directory)

    def plot_camera_image_hillas(self, keep=False, directory=None):
        """
        Plot a cleaned camera image of the current event with the Hillas
        parameterization overlaid. Ensures that Hillas parameters are
        calculated with the plot in mind. Might not work correctly if
        Hillas parameters are calculated prior to calling this function, if
        hillas_for_plot is False. Same keep and directory keyword arguments
        available as for plot_camera_image.

        :param keep: bool = False, determine whether to keep plot
        :param directory: str or None, provide a different save directory path
        """
        self.hillas_for_plot = True
        utils.plot_camera_image_hillas(self.clean_image, self.__run, self.__ev, utils.get_timestr(self.cpu_s),
                                       [self.hillas.x, self.hillas.y, self.hillas.width, self.hillas.length],
                                       keep=keep, directory=directory)
        self.hillas_for_plot = False


class Hillas:
    """
    Class that functions as a data container for Hillas parameters.
    Initializes all parameters to zero and allows for setting them. This was
    chosen to allow WaveformArrayReader to access Hillas parameters using
    property dot notation.
    """

    __slots__ = ("__a", "__b", "__x", "__y", "__width", "__length",
                 "__miss", "__dis", "__azwidth", "__alpha", "__size", "__psi")

    def __init__(self):
        self.__a = 0
        self.__b = 0
        self.__x = 0
        self.__y = 0
        self.__width = 0
        self.__length = 0
        self.__miss = 0
        self.__dis = 0
        self.__azwidth = 0
        self.__alpha = 0
        self.__psi = 0
        self.__size = 0

    @property
    def a(self):
        return self.__a

    @a.setter
    def a(self, a):
        self.__a = a

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, b):
        self.__b = b

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        self.__x = x

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self, y):
        self.__y = y

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, width):
        self.__width = width

    @property
    def length(self):
        return self.__length

    @length.setter
    def length(self, length):
        self.__length = length

    @property
    def miss(self):
        return self.__miss

    @miss.setter
    def miss(self, miss):
        self.__miss = miss

    @property
    def dis(self):
        return self.__dis

    @dis.setter
    def dis(self, dis):
        self.__dis = dis

    @property
    def azwidth(self):
        return self.__azwidth

    @azwidth.setter
    def azwidth(self, azwidth):
        self.__azwidth = azwidth

    @property
    def alpha(self):
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha):
        self.__alpha = alpha

    @property
    def psi(self):
        return self.__psi

    @psi.setter
    def psi(self, psi):
        self.__psi = psi

    @property
    def size(self):
        return self.__size

    @size.setter
    def size(self, size):
        self.__size = size

