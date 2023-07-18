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

# 4 asics, 16 ch per asic
nasic=4
nch=16

# max value of color bar
maxZ = 600

chPerPacket = 32

argList = sys.argv

runID=int(argList[1])

homedir = os.environ['HOME']

savedir = "/data/analysis_output/camera_movies/".format(homedir)

hostname = run_control.getHostName()
#indirname = run_control.getDataDirname(hostname)
#filename = indirname+"runs_300000_through_309999/run{}.fits".format(runID)
indirname = "/data/local_outputDir/"
filename = indirname+"run{}.fits".format(runID)
print "Reading file: ", filename

# modules need to be ordered in the same way that they were ordered in the data taking loop
#modList=[2,3,6,118,125,126,101,119,108,121,110,115,123,124,112,100,111,114,107,1,4,9]
#modList = [9,6,3,1,4,125, 126, 106, 119,108,121,115,123,124,112,100,111,114,107,2]
modList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]
#modList = [1, 3, 4, 5, 100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]
#modList = [119, 108,121,8, 115, 123, 124, 112, 7, 100, 111, 114, 107, 6]
#modList = [115]

if(len(argList)>2):
    choose_event = int(argList[2])
    if(len(argList)>3):
        modList=[]
        for i in range(3,len(argList)):
            modList.append(int(argList[i]))



#eventList = [1068,1098,1157,1188,1705,1706,1718,1723,1728,1734,1748,1749,1751,1754,1757,1801,1826, 1804,1808,1814,1818,1821,1844,1862,1867,1871,1887,1890,1904,1906,1907,1910, 1912, 1915, 1923, 1925, 1927, 1931, 1934, 1938, 1942, 1947, 1948, 1958, 1959, 1962, 1966, 1974]
eventList=[choose_event]
#modList = [100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]
#modList = [112] #[118,125,126,119,108,121,110,128,123,124,112,100,111,114,107]
nModules = len(modList)
print nModules, modList

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
print "number of events", nEvents
if nEvents > 500:
    nEvents = 5
rawdata = reader.GetEventPacket(0,0)
packet = target_driver.DataPacket()
packet.Assign(rawdata, reader.GetPacketSize())
print "The Packet size is: ", reader.GetPacketSize()
wf = packet.GetWaveform(0)
nSamples = 4*32  #wf.GetSamples()
print "Number of samples: ", nSamples

ampl = np.zeros([nEvents,nModules,nasic,nch,nSamples])
heatArray = np.zeros([nModules,nasic, nch])
physHeatArr = np.zeros([nModules,8,8])
physRedAmplArr = np.zeros([nModules,8,8,nSamples])

for choose_event in eventList:
    for modInd in range(len(modList)):
        for asic in range(nasic):
            for ch in range(nch):
                ievt = choose_event
                rawdata = reader.GetEventPacket(ievt, (((nasic*modInd+asic)*nch)+ch)/chPerPacket)  #FIXME
                packet = target_driver.DataPacket()
                packet.Assign(rawdata, reader.GetPacketSize())
                header = target_driver.EventHeader()
                reader.GetEventHeader(ievt, header);        #FIXME
                wf = packet.GetWaveform((asic*nch+ch)%chPerPacket)
                for sample in range(nSamples):
                    """
                    if(ievt<1102):
                        if( (modInd==19 and asic==1) or (modInd==16 and asic==0) or (modInd==9 and asic==2) ):      ##121 a,114,100
                            ampl[0, modInd, asic, ch, sample] = 0
                        else:
                            ampl[0, modInd, asic, ch, sample] = wf.GetADC(sample)
                    elif(ievt<1700):
                        if( (modInd==19 and asic==1) or (modInd==9 and asic==2) ):  ##121 a,114,100
                            ampl[0, modInd, asic, ch, sample] = 0
                        else:
                            ampl[0, modInd, asic, ch, sample] = wf.GetADC(sample)

                    else:
                        if( (modInd==9 and asic==2) ):      ##121 a,114,100
                            ampl[0, modInd, asic, ch, sample] = 0
                        else:
                            ampl[0, modInd, asic, ch, sample] = wf.GetADC(sample)
                    """
                    ampl[0, modInd, asic, ch, sample] = wf.GetADC(sample)

    with open('/data/fitslists/ampl_{}_ev{}.dat'.format(runID, choose_event),'wb') as f:
        pickle.dump(ampl,f)

    ampl = np.asarray(ampl)

    red_ampl = ampl[0]

    baseline = np.mean(red_ampl[:,:,:,1:15],axis=3)
    peak = np.amax(red_ampl[:,:,:,20:],axis=3)
    diff = peak-baseline
    allsamp_diff = red_ampl - np.tile(np.expand_dims(baseline,axis=3),nSamples)

    for modInd in range(len(diff)):
        if modInd in range(9):
            diff[modInd,:,:] /= 2.0
            allsamp_diff[modInd,:,:,:] /= 2.0


    # reshapes the heatArray from (len(modList),4,16) to (len(modList),64)
    # this allows us to use the index reassignment function
    # also taking the average of the heat array
    heatArray = diff.reshape((len(modList),64))
    redAmplArray = allsamp_diff.reshape((nModules,nasic*nch,nSamples))

    # apply index reassignment
    # phys array will appear upside down in array form
    # but pcolor plots index 0 from bottom up
    physHeatArr[:,row,col] = heatArray
    physRedAmplArr[:,row,col,:] = redAmplArray

    maxZ = physRedAmplArr[:,:,:,:].max()
    print "The maximum is going to be:", maxZ

    def calcLoc(modInd):
        # determine the location of the module in the gridspace
        reflectList = [4, 3, 2, 1, 0]
        loc = tuple(np.subtract(posGrid[modPos[modList[modInd]]],(1,1)))
        locReflect = tuple([loc[0],reflectList[loc[1]]])
        return loc, locReflect

    try:
        os.mkdir("{}{}_ev{}_skyview".format(savedir,runID,choose_event))
        #os.mkdir("{}/movie/{}_ev{}_camview".format(savedir,runID,choose_event))
    except:
        print("Directories already exist")
    for sample in range(40,91):   #nSamples):

        physHeatArr[:,:,:] = physRedAmplArr[:,:,:,sample]

        # set up plotting
        heatFig = plt.figure('Pixel Heat Map', (18.,15.))
        heatReflectFig = plt.figure('Heat Map Skyview', (18.,15.))

        if full:
            gs = gridspec.GridSpec(5,5)
        else:
            gs = gridspec.GridSpec(4,4)
        gs.update(wspace=0.04, hspace=0.04, bottom=0.1, right=0.8)
        gs2 = gridspec.GridSpec(1,5)
        gs2.update(bottom=0.05, top=0.08, right=0.86)

        for modInd in range(nModules):
            loc, locReflect = calcLoc(modInd)

            # modules in odd columns are rotated by 180 degrees
            # these are even columns here, because gridspec uses 0-based indexing
            if loc[1]%2==0:
                physHeatArr[modInd,:,:]=np.rot90(physHeatArr[modInd,:,:],k=2)
            """
            # deal with heat map
            plt.figure('Pixel Heat Map')
            ax = plt.subplot(gs[loc])
            c = ax.pcolor(physHeatArr[modInd,:,:], vmin=0, vmax=maxZ, cmap=viridius)
            # take off axes
            ax.axis('off')
            ax.set_aspect('equal')
            """
            # deal with skyview heat map
            plt.figure('Heat Map Skyview')
            ax4= plt.subplot(gs[locReflect])
            c4 = ax4.pcolor(physHeatArr[modInd,:,::-1], vmin=0, vmax=600)    ##, cmap=plt.get_cmap('viridis'))
            # take off axes
            ax4.axis('off')
            ax4.set_aspect('equal')

        ax_bot = plt.subplot(gs2[:])
        ax_bot.barh(0,sample-40, align='edge', height=1)
        ax_bot.set_xlabel("time in ns", fontsize=24)
        ax_bot.set_yticks([])
        ax_bot.tick_params(axis='x', labelsize=20)
        ax_bot.set_ylim(0,1)
        ax_bot.set_xlim(0,50)
        #ax_bot.annotate("{} ns".format(sample), xy=(113,0.1), fontsize=25)

        """
        heatFig.suptitle("Prototype Schwarzschild Couder Telescope first light,\n January 23, 2019", fontsize=30, y=0.97)
        #plt.title("{} ns".format(sample), fontsize=20, x=0.5, y=0.05)
        heatFig.subplots_adjust(right=0.8,top=0.9,bottom=0.1)
        cbar_ax = heatFig.add_axes([0.81, 0.1, 0.05, 0.8])
        cbar = heatFig.colorbar(c, cax=cbar_ax)
        cbar.set_label('Amplitude (ADC counts)', rotation=270,size=24,labelpad=28)
        cbar_ax.tick_params(labelsize=20)
        heatFig.savefig("{}/movie/{}_ev{}_camview/{}_diff.png".format(savedir,runID, choose_event,sample))
        """

        heatReflectFig.suptitle("Prototype Schwarzschild Couder Telescope first light\n January 23, 2019", fontsize=30, y=0.97)
        #heatReflectFig.subplots_adjust(right=0.75,top=0.9,bottom=0.15)
        cbar_ax4 = heatReflectFig.add_axes([0.81, 0.1, 0.05, 0.8])
        cbar4 = heatReflectFig.colorbar(c4, cax=cbar_ax4)
        cbar4.set_label('Amplitude (ADC counts)', rotation=270,size=24,labelpad=30)
        cbar_ax4.tick_params(labelsize=20)
        heatReflectFig.savefig("{}{}_ev{}_skyview/{}_diff.png".format(savedir,runID,choose_event,sample))
