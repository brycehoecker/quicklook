# Usage
# $ python make_interactive_heatmap.py ##### $$$$$ #
# Where '######' corresponds to a run number in a run######.h5 database created with SCT toolkit
# Where '$$$$$' corresponds to an event number present in the corresponding run

import numpy as np
import sys, os
import urllib, base64
import StringIO
from sct_toolkit_new import pedestal, waveform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool, BasicTicker, LinearColorMapper, LogTicker, ColorBar
from tqdm import tqdm

#print("Currently set to use Miles example runs. Change back to correct mod_nums and fpm_pos to use Leslie runs.")
#Grab values from the arglist
argList = sys.argv
runID=int(argList[1]) #argList[1] means the second thing that was typed into the terminal. argList[0] would be "make_interactive_heatmap.py"
if len(argList)>2:
	eventID=int(argList[2]) #argList[2] means the third thing that was typed into the terminal. Here it is optional to specify an individual event
	one_event=True
else:
	one_event=False

#filename = "/data/wipac/CTA/target5and7data/run{}.h5".format(runID)
filename = "/data/h5_output/run{}.h5".format(runID) # calls file from folder run_files with name run######.h5 where ###### is the run ID. This file is made with mk_wave and can be pedestal subtracted or not.
print("Reading file: {}".format(filename))
wf = waveform(filename) # calls the script waveform from sct_toolkit folder. waveform takes in raw or calibrated waveform data and applies pedistal subtraction.
print(wf)
camera_facing = False # If true, view is from the front of the camera. If false it is from the back.


## define required mappings to create image
print("Generating camera image pixel mappings")

# mod numbers and fpm numbers are written in order from left to right, bottom to top.
#mod_nums = [118, 125, 126, 119, 108, 121, 110, 128, 123, 124, 112, 100, 111, 114, 107]
mod_nums = [100,111,114,107,6,115,123,124,112,7,119,108,110,121,8,103,125,126,106,9,4,5,1,3,2] #FIXME
fpm_nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24] #FIXME

# create 5x5 coordinates
#fpm_pos = np.mgrid[0:5,0:5]
fpm_pos = np.mgrid[0:5, 0:5]
fpm_pos = zip(fpm_pos[0].flatten(),fpm_pos[1].flatten())

# associate modules to FPMs
mod_to_fpm = dict(zip(mod_nums,fpm_nums))
fpm_to_pos = dict(zip(fpm_nums,fpm_pos))

# Create channel mapping for a single module
ch_nums = np.array([[21,20,17,16,5,4,1,0],
                    [23,22,19,18,7,6,3,2],
                    [29,28,25,24,13,12,9,8],
                    [31,30,27,26,15,14,11,10],
                    [53,52,49,48,37,36,33,32],
                    [55,54,51,50,39,38,35,34],
                    [61,60,57,56,45,44,41,40],
                    [63,62,59,58,47,46,43,42]])



# Display image from the front if camera_facing=True
if camera_facing: #Put not in front of camera_facing because it seems like for some reason, you need camera_facing to be True and False both to make all of the variables that get used outside of these control conditions. Really odd. Not what the question is though.
    ch_nums = ch_nums[:,::-1] # mirrors the ch_nums matrix
rot_ch_nums = np.rot90(ch_nums,k=2) # rotates ch_nums by 180 degrees
ch_to_pos = dict(zip(ch_nums.reshape(-1),np.arange(64))) # creates dictionary linking channel number to the position (where position is counted left to right, top to bottom) for unrotated modules
rot_ch_to_pos = dict(zip(rot_ch_nums.reshape(-1),np.arange(64))) # creates dictionary linking channel number to the position (where position is counted left to right, top to bottom) for rotated modules




# define arrays to hold data for each pixel
num_columns = 5 # define number of columns which are in the camera
total_cells = num_columns*num_columns*64 #4*4*64 # total number of channels/pixels in the image
indices = np.arange(total_cells).reshape(-1,int(np.sqrt(total_cells))) # creates square matrix of channel indices for all channels in all modules (40 X 40 grid of channels)
pixels = np.zeros(total_cells) # initialize list of zeros with length equal to number of pixels/channels
pixels_waveforms = np.zeros((total_cells,wf.get_n_samples())) # initialize list of zeros with #total_cells rows and #waveforms columns
mod_desc = np.zeros(total_cells,dtype=int) # initialize list of zeros #total_cells long
ch_desc = np.zeros(total_cells,dtype=int)
asic_desc = np.zeros(total_cells,dtype=int)



# loop through each mod, asic, channel and assign data to correct pixel
print("Assigning data to correct location in pixel map")
#os.environ['DISPLAY'] = ':0.0'
for mod in wf.get_module_list(): #wf.get_module_list() returns a list of the modules which were included during data-taking

    # assign grid position
    i, j = fpm_to_pos[mod_to_fpm[mod]] # for each module number 'mod', get its FPM number and using this get its coordinates (as set above)

    # assign proper channel mapping w/ 180 rotation every other column
    ch_map = dict() # initialize channel map dictionary
    if j%2==0: # for even columns rotate the channels
        ch_map = rot_ch_to_pos # dictionary of channel number to position number
    else: # for odd columns do not rotate the channels
        ch_map = ch_to_pos

    # change order if sky view
    if not camera_facing:
        j = num_columns-1-j # flips column number if we are camera_facing
        pix_ind = np.array(indices[(8*i):8*(i+1),(8*j):8*(j+1)]).reshape(-1) # creates list of channel indices for the relevant module

    for asic in wf.get_asic_list(): # get_asic_list() returns a list of the ASICs which were included during data-taking
        for ch in wf.get_channel_list(): # get_channel_list() returns a list of the channels which were included during data-taking
            charge = np.array(wf.get_branch('Module{}/Asic{}/Channel{}/charge'.format(
                                       mod,asic,ch))) # get list of charges in ADC*ns with 16ns integration window
            waveform = np.array(
                           wf.get_branch('Module{}/Asic{}/Channel{}/waveform'.format(
                                       mod,asic,ch))) # get list of calibrated (ped-subtracted) waveform. #FIXME change waveform to cal_waveform if using pedistal subtraction

            # assign location in pixel grid
            grid_ind = int(pix_ind[ch_map[asic*16+ch]]) # assign current channel to its master channel number for future reference (in grid of channels/pixels)

            # add data and description labels
	    # Modification if only selecting one event at a time! #FIXME
            if one_event==True:
                #pixels[grid_ind] = charge[eventID] # add the charge of just the specified eventi
                pixels[grid_ind] = max(waveform[eventID]) - min(waveform[eventID])
                pixels_waveforms[grid_ind,:] = waveform[eventID] # add the waveform of just the specified event
                if mod in range(1, 10):
                    pixels[grid_ind] /= 2.0
                    pixels_waveforms[grid_ind,:] /=2.0
            else:
                #pixels[grid_ind] = np.mean(charge) # add mean of all charges (from all waveforms) to appropriate location in pixels list
                pixels_waveforms[grid_ind,:] = np.mean(waveform,axis=0) # add mean of all waveforms to appropriate location in pixels_waveforms list
                mod_desc[grid_ind] = int(mod) # add module number
                ch_desc[grid_ind] = int(ch) # add chanel number
                asic_desc[grid_ind] = int(asic) # add asic number


# Create a static image of charges
#for n in range(128):
try:
    #os.mkdir('/data/analysis_output/camera_movies/run{}_ev{}_movie'.format(runID, eventID))
    os.mkdir('/data/analysis_output/camera_movies/run{}_ev{}_movie'.format(runID, eventID))
except:
    print("Directory already exists.")
for n in tqdm(range(128)):
    charge_plot = []
    for row in indices:
        temp = []
        for val in row:
            #temp.append(pixels_waveforms[val][n])
            temp.append(pixels_waveforms[val][n])
        charge_plot.append(temp)
    #fig = plt.imshow(charge_plot, cmap='viridis', vmin=550, vmax=950,  interpolation='nearest', origin=[0,0])
    fig = plt.imshow(charge_plot, cmap='viridis', vmin=np.amin(pixels), vmax=np.amax(pixels),  interpolation='nearest', origin=[0,0])
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    clb = plt.colorbar()
    clb.set_label('ADC Counts')
    plt.title('Run ' + str(runID) + ', Event ' + str(eventID) + ', ' + str(n) + ' ns')
    #plt.savefig('camera_figures/Waveform_run' + str(runID) + '_event' + str(eventID) + '_time' + str(n) + '.png')
    #plt.savefig('/data/analysis_output/camera_movies/run{}_ev{}_movie/BaseSubWaveform_run'.format(runID, eventID) + str(runID) + '_event' + str(eventID) + '_time' + str(n) + '.png')
    plt.savefig('/data/analysis_output/camera_movies/run{}_ev{}_movie/BaseSubWaveform_run{}_event{}_time{}.png'.format(runID, eventID, runID, eventID, n))
    plt.close()


'''
# convert waveforms to base-64 strings
image_list = [] # initializes list of images
for i in xrange(total_cells): # iterates through each pixel in the camera
    sys.stdout.write("\rCreating plots: {}/{}".format(i+1,total_cells)) # prints 'creating plots #/####' where # is the current plot and #### is the total number of plots (ie the total number of cells/channels/pixels)
    sys.stdout.flush() # forces all outputs to be written
    plt.plot(pixels_waveforms[i]) # plots the mean-calibrated waveform for each pixel (where pixel is specified by i)
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude (ADC Counts)')
    fig = plt.gcf() # gets a reference to the current figure

    imgdata = StringIO.StringIO() # creates file-like object imgdata
    fig.savefig(imgdata, format='png') # saves the figure above to imgdata
    imgdata.seek(0)  # rewind the data # sets the file's current position back to 0
    image = 'data:image/png;base64,' + urllib.quote(base64.b64encode(imgdata.buf)) # save file of figure to image
    image_list.append(image) # add image to the image list
    fig.clf() # Closes current figure so that on the next loop the figures do not build on eachother.

print("Generating html")
output_file("interactive_heatmap.html",title='Interactive Heatmap') # creates file interactive_heatmap.html which will be filled shortly

color_mapper = LinearColorMapper(palette='Viridis256', low=0, high=np.amax(pixels)) # select color map for charge. Lowest number is 0 and highest is the max mean charge of any waveform in pixels
Y, X = np.mgrid[0.05:3.95:40j,0.05:3.95:40j] # creates a mesh grid which starts at 0.05, ends at 3.95, and has dimensions 40x40. This creates a grid with the same number of elements as there are channels in the entire camera. With each channel given a width of 0.1 in the html. This is the actual color picture part (not the hover plots)
source = ColumnDataSource(data=dict(
    x = X.reshape(-1), # key[x], X.reshape flattens the grid.
    y = Y.reshape(-1), # key[y], Y.reshape flattens the grid. Together each (x,y) pair gives the coordinates in the figure.
    desc=mod_desc.astype(str),
    ch_desc=ch_desc.astype(str),
    asic_desc = asic_desc.astype(str),
    charge = pixels.astype(int),
    imgs=image_list)) # List of Modules, Channels, ASICs, Charges, and Images which match the length of the data list

hover = HoverTool( tooltips="""
    <div>
        <div>
            <img
                src=@imgs height="40%" alt="@imgs" width="30%"
                style="float: left; margin: 0px 2px 2px 0px; position: fixed; left: 700px; top: 80px
;"
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 15px; font-weight: bold;">Module @desc</span>
            <!---span style="font-size: 15px; color: #966;"> [$index]</span--->
        </div>
        <div>
            <span style="font-size: 15px; font-weight: bold;">ASIC @asic_desc, Channel @ch_desc</span>
        </div>
        <div>
            <span style="font-size: 15px;">Charge: @charge ADC &middot ns</span>
        </div>
    </div>
    """
)
# tooltips gives the HTML specifications needed to produce a webpage
# First section covers where the focal plane image is located in the HTML-it is a fixed position
# Second section gives the specifications for the hover pop-up for each pixel in HTML (like the words used, font, etc)

hm = figure(title="Camera Image", tools=[hover], toolbar_location="below",
           toolbar_sticky=False, x_range=(0,4), y_range=(0,4)) #X_range and y_range are set to correspond to the mesh grid above. Ranging from 0 to 4 for each.
hm.image(image=[pixels.reshape(-1,int(np.sqrt(total_cells)))],
          color_mapper=color_mapper,dh=[4], dw=[4], x=[0], y=[0]) #resets shape of pixels to square and sets this as image. sets color map to that provided above. dh=height=4 dw=width=4 to match the x_range/y_range and mesh grid above. x and y are coordinates to locate image anchors - set to the origin.
hm.rect('x', 'y', fill_alpha=0, line_alpha=0, width=.1, height=.1, source=source) #Creates rectangles on top of each pixel which can be recognized by the hover. The details of their orientation come from "source" above.
hm.axis.visible = False #hides axes we use to position images.
color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), title='Charge',
                     label_standoff=12, border_line_color=None, location=(0,0)) #creates color bar on the side of the image.
hm.add_layout(color_bar, 'right') #places the colorbar on the right
show(hm) #Shows the final image
'''
