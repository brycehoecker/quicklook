import argparse
from typing import List

from astropy.io import fits as pyfits

from psct_reader import WaveformArrayReader as Reader


def reduce_fits(infile: str, outfile: str, event_list: List):
    """
    Principle function for reducing a pSCT FITS file. 

    infile: str; should be a pSCT .fits or .r1 filename
    outfile: str; should be named red######.fits for reduced .fits
                  files or red######.r1 for reduced .r1 files
    event_list: List; list of events from infile to include in outfile
    """
    
    print("Reducing FITS file!\n(This might take a while...)")
    with pyfits.open(infile) as infits:  # if something breaks, our original file closes
        infits[0].writeto(outfile, checksum=False)  # checksum False to avoid new cards
        data_ext = infits[1]
        event_list = list(event_list)  # in case someone tries to use a numpy array
        pyfits.append(outfile,
                      data_ext.data[event_list],
                      header=data_ext.header,
                      checksum=False,
                      verify=False)
        with pyfits.open(outfile) as outfits:
            verify_fits_headers(infits, outfits)  # doing this in context managers in case they don't match
    
    inreader = Reader(infile)
    outreader = Reader(outfile)
    verify_fits_data(inreader, outreader, event_list[0], event_list[-1])
    print("FITS file reduced successfully!")

def verify_fits_headers(infits: pyfits.HDUList, outfits: pyfits.HDUList):
    print("Checking FITS headers")
    assert(infits[0].header == outfits[0].header)
    assert(infits[1].header["NAXIS1"] == outfits[1].header["NAXIS1"])
    print("FITS headers verified!")

def verify_fits_data(inreader: Reader, outreader: Reader, first_ev: int, last_ev: int):
    print("Checking data integrity")
    assert(inreader.n_pixels == outreader.n_pixels)
    assert(inreader.n_samples == outreader.n_samples)
    inreader.get_event(first_ev)
    outreader.get_event(0)
    assert(np.array_equal(inreader.wfs, outreader.wfs))
    inreader.get_event(last_ev)
    outreader.get_event(outreader.n_events - 1)
    assert(np.array_equal(inreader.wfs, outreader.wfs))
    print("Data integrity verified!")

def parse_simple_file(event_file: str) -> List:
    event_list = []
    with open(event_file) as f:
        print(f"Parsing events from {event_file}")
        for line in f:
            event_list.append(int(line))
    return sorted(event_list)  # not sure what would happen if it wasn't sorted...



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help="path to input file")
    parser.add_argument("-o", "--outfile", help="path to output file")
    parser.add_argument("-e", "--events", help="path to event list file")
    args = parser.parse_args()
    print(args)
    infile = args.infile
    outfile = args.outfile
    event_file = args.events
    event_list = parse_simple_file(event_file)
    reduce_fits(infile, outfile, event_list)


