import target_io
import target_driver
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import datetime
import time
import sys
import os
import run_control
import pickle

#### How to Use ####
# python CameraImage.py RunNumber EventNumber EndEvent StepSize 
# RunNumber and EventNumber are required
# If you include EndEvent then all events between EventNumber and EndEvent will be plotted
# If you include StepSize then every StepSize (ie third, fourth, 10th, etc)  event between EventNumber and EndEvent will be plotted

'''
#Setting event list from inputs
argList = sys.argv
runID=int(argList[1])
start_event=int(argList[2])

end_event = start_event
step=1
if(len(argList)>3):
    end_event = int(argList[3])
    if(len(argList)>4):
        step=int(argList[4])

eventList = [i for i in range(start_event,end_event+1,step)]
'''
argList = sys.argv
runID=int(argList[1])
eventList = []
for i in range(len(argList)-2):
    eventList.append(int(argList[i+2])) #FIXME
    eventList.append(int(argList[i+2])+1)
print(eventList)

# Setting output directory
#homedir = os.environ['HOME']
savedir = "/data/analysis_output/camera_images/run{}/".format(runID)
try:
    os.mkdir(savedir)
except:
    pass

# 4 asics, 16 ch per asic
nasic=4
nch=16

# max value of color bar
maxZ = 1200

chPerPacket = 32

hostname = run_control.getHostName()
#indirname = run_control.getDataDirname(hostname)
#filename = indirname+"runs_300000_through_309999/run{}.fits".format(runID)
indirname = "/data/local_outputDir/"
filename = indirname+"run{}.fits".format(runID)
print("")
print("Reading file: ", filename)

# modules need to be ordered in the same way that they were ordered in the data taking loop
#modList=[2,3,6,118,125,126,101,119,108,121,110,115,123,124,112,100,111,114,107,1,4,9]
#modList = [9,6,3,1,4,125, 126, 106, 119,108,121,115,123,124,112,100,111,114,107,2]
modList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]
#modList = [1, 3, 4, 5, 100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]
#modList = [119, 108,121,8, 115, 123, 124, 112, 7, 100, 111, 114, 107, 6]

#modList = [100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]
#modList = [112] #[118,125,126,119,108,121,110,128,123,124,112,100,111,114,107]
nModules = len(modList)
#print(nModules, modList)

#modPos = {     3:5, 9:6, 2:9,
#               118:11, 125:12, 126:13, 106:14, 1:15,
#               119:17, 108:18, 121:19, 110:20, 4:21,
#               115:23, 123:24, 124:25, 112:26, 6:27,
#               100:28, 111:29, 114:30, 107:31,
#               101:14} #101 was formerly in slot 14 before it broke

modPos = {      4:5, 5:6, 1:7, 3:8, 2:9,
                103:11, 125:12, 126:13, 106:14, 9:15,
                119:17, 108:18, 110:19, 121:20, 8:21,
                115:23, 123:24, 124:25, 112:26, 7:27,
                100:28, 111:29, 114:30, 107:31, 6:32,
                101:14} #101 was formerly in slot 14 before it broke

posGrid =       {5:(1,1), 6:(1,2), 7:(1,3), 8:(1,4), 9:(1,5),
                11:(2,1), 12:(2,2), 13:(2,3), 14:(2,4), 15:(2,5),
                17:(3,1), 18:(3,2), 19:(3,3), 20:(3,4), 21:(3,5),
                23:(4,1), 24:(4,2), 25:(4,3), 26:(4,4), 27:(4,5),
                28:(5,1), 29:(5,2), 30:(5,3), 31:(5,4), 32:(5,5)}

"""
This method will calculate index reassignments
that trace out the Z-order curve that the pixels
form in the focal plane.
See: http://stackoverflow.com/questions/42473535/arrange-a-numpy-array-to-represent-physical-arrangement-with-2d-color-plot
"""
def row_col_coords(index):
    # Convert bits 1, 3 and 5 to row
    row = 4*((index & 0b100000) > 0) + 2*((index & 0b1000) > 0) + 1*((index & 0b10) > 0)
    # Convert bits 0, 2 and 4 to col
    col = 4*((index & 0b10000) > 0) + 2*((index & 0b100) > 0) + 1*((index & 0b1) > 0)
    return (row, col)

# calculating the actual index reassignments
row, col = row_col_coords(np.arange(64))

# set up reader, find number of events, packet size, and number of samples
reader = target_io.EventFileReader(filename)
nEvents = reader.GetNEvents()
#print("number of events", nEvents
#if nEvents > 500:
#       nEvents = 5
#eventList = [i for i in range(nEvents)] # BM: I did a bad thing maybe? Fight me. (Directed at self.)
rawdata = reader.GetEventPacket(0,0)
packet = target_driver.DataPacket()
packet.Assign(rawdata, reader.GetPacketSize())
#print("The Packet size is: ", reader.GetPacketSize()
wf = packet.GetWaveform(0)
nSamples = 4*32  #wf.GetSamples()
#print("Number of samples: ", nSamples

ampl = np.zeros([nEvents,nModules,nasic,nch,nSamples])
heatArray = np.zeros([nModules,nasic, nch])
physHeatArr = np.zeros([nModules,8,8])
physRedAmplArr = np.zeros([nModules,8,8,nSamples])

for choose_event in eventList:
    print("Now plotting event: ", choose_event)
    for modInd in range(len(modList)):
        for asic in range(nasic):
            for ch in range(nch):
                ievt = choose_event
                rawdata = reader.GetEventPacket(ievt, int((((nasic*modInd+asic)*nch)+ch)/chPerPacket))  #FIXME
                packet = target_driver.DataPacket()
                packet.Assign(rawdata, reader.GetPacketSize())
                header = target_driver.EventHeader()
                reader.GetEventHeader(ievt, header);        #FIXME
                wf = packet.GetWaveform((asic*nch+ch)%chPerPacket)
                for sample in range(nSamples):
                    ampl[0, modInd, asic, ch, sample] = wf.GetADC(sample)

    ampl = np.asarray(ampl)

    red_ampl = ampl[0]

    #print(np.shape(ampl[0]))
    #print(np.shape(red_ampl))
    baseline = np.mean(red_ampl[:,:,:,1:15],axis=3)
    peak = np.amax(red_ampl[:,:,:,20:],axis=3)
    diff = peak-baseline
    allsamp_diff = red_ampl - np.tile(np.expand_dims(baseline,axis=3),nSamples)
    #print(np.shape(diff))
    #print(diff[0])
    #print(diff[1])
    #print(diff[2])
    for modInd in range(len(diff)):
        if modInd in range(9):
            diff[modInd,:,:] /= 2.0
            allsamp_diff[modInd,:,:,:] /= 2.0
    
    #maxZ = np.amax(diff)
    # reshapes the heatArray from (len(modList),4,16) to (len(modList),64)
    # this allows us to use the index reassignment function
    # also taking the average of the heat array
    heatArray = diff.reshape((len(modList),64))
    redAmplArray = allsamp_diff.reshape((nModules,nasic*nch,nSamples))

    # apply index reassignment
    # phys array will appear upside down in array form
    # but pcolor plots index 0 from bottom up
    physHeatArr[:,row,col] = heatArray
    #physHeatArr = np.mean(physHeatArr, axis=0)
    physRedAmplArr[:,row,col,:] = redAmplArray

    def calcLoc(modInd):
        # determine the location of the module in the gridspace
        reflectList = [4, 3, 2, 1, 0]
        loc = tuple(np.subtract(posGrid[modPos[modList[modInd]]],(1,1)))
        locReflect = tuple([loc[0],reflectList[loc[1]]])
        return loc, locReflect


    # set up plotting
    heatReflectFig = plt.figure('Heat Map Skyview', (18.,15.))

    gs = gridspec.GridSpec(5,5)
    gs.update(wspace=0.04, hspace=0.04)

    for modInd in range(nModules):
        loc, locReflect = calcLoc(modInd)

        # modules in odd columns are rotated by 180 degrees
        # these are even columns here, because gridspec uses 0-based indexing
        if loc[1]%2==0:
            physHeatArr[modInd,:,:]=np.rot90(physHeatArr[modInd,:,:],k=2)

        # deal with heat map
        plt.figure('Pixel Heat Map')
        ax = plt.subplot(gs[loc])
        c = ax.pcolor(physHeatArr[modInd,:,:], vmin=0, vmax=maxZ)
        # take off axes
        ax.axis('off')
        ax.set_aspect('equal')

        # deal with skyview heat map
        plt.figure('Heat Map Skyview')
        ax4= plt.subplot(gs[locReflect])
        c4 = ax4.pcolor(physHeatArr[modInd,:,::-1], vmin=0, vmax=maxZ)
        # take off axes
        ax4.axis('off')
        ax4.set_aspect('equal')

    heatReflectFig.subplots_adjust(right=0.8,top=0.9,bottom=0.1)
    cbar_ax4 = heatReflectFig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar4 = heatReflectFig.colorbar(c4, cax=cbar_ax4)
    cbar4.set_label('Max ADC - baseline (ADC counts)', rotation=270,size=20,labelpad=24)
    cbar_ax4.tick_params(labelsize=16)
    heatReflectFig.savefig("{}/{}_ev{}_diff_pxlSkyHeatMap.png".format(savedir,runID, choose_event))
    plt.clf()

print("Event images saved to: ", savedir)
print("")
