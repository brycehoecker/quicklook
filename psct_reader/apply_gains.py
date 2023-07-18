import h5py
import numpy as np

import target_io

datadir = "/data/wipac/CTA/target5and7data/runs_320000_through_329999"
datadir = "."
gain_file = f"{datadir}/gain_calibration.hdf5"
f = h5py.File(gain_file, "r")

gain = f["Gains"][()]
f.close()

def apply_gains(charges):
    if charges.shape == (1536, 128):
        return (charges / gain[:, None])
    else:
        return (charges / gain)

if __name__=="__main__":
    from apply_gains import apply_gains
    runID = 327583
    datadir = "/data/wipac/CTA/target5and7data/runs_320000_through_329999"
    calfile = f"{datadir}/cal{runID}.r1"
    reader = target_io.WaveformArrayReader(calfile)
    n_samples = reader.fNSamples
    n_pixels = reader.fNPixels
    n_events = reader.fNEvents
    event = 1015
    waveforms = np.zeros((n_pixels, n_samples), dtype=np.float32)
    reader.GetR1Event(event, waveforms)
    peak_position = np.argmax(waveforms, axis=1)
    charge = mcs.calculate_charge(waveforms, peak_position, n_samples)
    charge = apply_gains(charge)
    print(f"Charge: {charge[149:152]}")

    #gain = list(np.sort(gain))
    print("Start:")
    print(gain[149:152])

