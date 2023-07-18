from sct_toolkit_new import pedestal
from sct_toolkit_new import waveform
import os
import sys

pedID = None

if len(sys.argv) == 2:
    runID = sys.argv[1]
elif len(sys.argv) == 3:
    runID = sys.argv[1]
    pedID = sys.argv[2]
else:
    raise SystemExit

module_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]

if pedID is not None:
    ped_name = 'pedestal_database_{}.h5'.format(pedID)
    ped = pedestal()
    wf = waveform()
    try:
        ped.make_pedestal_database(ped_name=ped_name, run_number=pedID, modules=module_list, filepath='/data/wipac/CTA/target5and7data/run{}.fits'.format(pedID))
    except:
        pass
    #wf.write_events(run_number=runID, modules=module_list, ped_name=ped_name, outdir='/data/h5_output/')
    wf.write_events(run_number=runID, modules=module_list, ped_name=ped_name, outdir='/data/user/bmode/h5_output/', filepath='/data/wipac/CTA/target5and7data/run{}.fits'.format(runID))
elif pedID is None:
    ped_name = 'pedestal_database_{}.h5'.format(326709)
    wf = waveform()
    wf.write_events(run_number=runID, modules=module_list, ped_name=ped_name, outdir='/data/h5_output/')

