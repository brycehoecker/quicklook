import numpy as np

import target_io
import target_calib

def load_ped(pedfile: str, n_modules: int=24) -> np.ndarray:
    calibrator = target_calib.Calibrator(pedfile)
    ped = calibrator.GetPedLookup()
    return np.asarray(ped).reshape((int(n_modules)*64, 512, 159))

def get_ped_wf(ped: np.ndarray, block: int, block_phase: int) -> np.ndarray:
    return ped[:, block, block_phase:block_phase + 128]
