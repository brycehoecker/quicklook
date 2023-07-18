import os
import sys

import base64
from bokeh.models import HoverTool, BasicTicker, LinearColorMapper, LogTicker, ColorBar, GlyphRenderer, Rect
from bokeh.plotting import figure, output_file, show, save, ColumnDataSource
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#from io import StringIO
from io import BytesIO
import urllib
from tqdm import tqdm

import analysis_quicklook
from apply_gains import apply_gains
import target_io

def make_heatmap_cal(runID, pedID,  event_index, datadir, charge_heatmap=True, gain=True):
    modlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 103, 106, 107, 
               108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]
    #savedir = f"{datadir}/run{runID}"
    savedir = f"/data/wipac/CTA/web/analysis_output/interactive_heatmaps/run{runID}"
    try:
        os.mkdir(savedir)
    except Exception as ex:
        print(ex)
        pass
    quicklook = analysis_quicklook.Quicklooker(pedID, runID, modlist, datadir, savedir)
    
    print("Generating camera image pixel mappings")

    waveforms = np.zeros((quicklook.n_pixels, quicklook.n_samples), dtype=np.float32)
    quicklook.calreader.GetR1Event(int(event_index), waveforms)
    ev_analysis = analysis_quicklook.EventAnalyzer(waveforms, quicklook.n_samples, quicklook.n_pixels)
    charge = ev_analysis.charge
    #charge = apply_gains(charge)
    waveforms = apply_gains(waveforms)

    mod_nums = [100,111,114,107,6,115,123,124,112,7,119,108,110,121,8,103,125,126,106,9,4,5,1,3,2] #FIXME
    fpm_nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24] #FIXME

    # create 5x5 coordinates
    fpm_pos = np.mgrid[0:5,0:5]
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
    rot_ch_nums = np.rot90(ch_nums, k=2)
    ch_to_pos = dict(zip(ch_nums.reshape(-1), np.arange(64)))
    rot_ch_to_pos = dict(zip(rot_ch_nums.reshape(-1), np.arange(64)))
    num_columns = 5
    total_cells = 5 * 5 * 64
    indices = np.arange(total_cells).reshape(-1, int(np.sqrt(total_cells)))
    pixels = np.zeros(total_cells)
    pixels_waveforms = np.zeros((total_cells, quicklook.n_samples))
    mod_desc = np.zeros(total_cells, dtype=int)
    ch_desc = np.zeros(total_cells, dtype=int)
    asic_desc = np.zeros(total_cells, dtype=int)
    pixel_desc = np.zeros(total_cells, dtype=int)
    index_desc = np.zeros(total_cells, dtype=int)
    for index, mod in enumerate(modlist):
        i, j = fpm_to_pos[mod_to_fpm[mod]]
        ch_map = dict()
        if j % 2 == 0:
            ch_map = rot_ch_to_pos
        else:
            ch_map = ch_to_pos
        j = num_columns - 1 - j
        pix_ind = np.array(indices[(8*i):8*(i+1), (8*j):8*(j+1)]).reshape(-1)
        for asic in range(4):
            for ch in range(16):
                grid_ind = int(pix_ind[ch_map[asic * 16 + ch]])
                pixels_waveforms[grid_ind, :] = waveforms[64 * index + asic * 16 + ch, :]
                if charge_heatmap is True:
                    pixels[grid_ind] = charge[64 * index + asic * 16 + ch]
                else:
                    pixels[grid_ind] = ev_analysis.peak_to_peak[64 * index + asic * 16 + ch]
                mod_desc[grid_ind] = int(mod)
                ch_desc[grid_ind] = int(ch)
                asic_desc[grid_ind] = int(asic)
                pixel_desc[grid_ind] = int(index * 64 + asic * 16 + ch)
                if mod < 110:
                    index_desc[grid_ind] = int(index * 64 + asic * 16 + ch)
                elif mod > 110:
                    index_desc[grid_ind] = int(index * 64 + asic * 16 + ch - 64)

    image_list = []
    for i in tqdm(range(total_cells)):
        plt.plot(pixels_waveforms[i])
        plt.xlabel("Time (ns)", fontsize=14)
        plt.ylabel("Amplitude (ADC Counts / Gain)", fontsize=14)
        plt.tick_params(labelsize=10)
        fig = plt.gcf()
        #imgdata = StringIO()
        imgdata = BytesIO()
        fig.savefig(imgdata, format="png")
        imgdata.seek(0)
        image = "data:image/png;base64," + urllib.parse.quote(base64.b64encode(imgdata.getbuffer()))
        image_list.append(image)
        fig.clf()
    print("Generating html...")
    if charge_heatmap is True:
        output_filename = f"{savedir}/run{runID}_ev{event_index}_Interactive_Heatmap_charge_pe.html" 
    else:
        output_filename = f"{savedir}/run{runID}_ev{event_index}_Interactive_Heatmap_peaktopeak.html"
    output_file(output_filename, title=f"Run: {runID} Event: {event_index}")
    color_mapper = LinearColorMapper(palette="Viridis256", low=0, high=np.amax(pixels))
    Y, X = np.mgrid[0.05:3.95:40j, 0.05:3.95:40j]
    source = ColumnDataSource(data=dict(
        x = X.reshape(-1),
        y = Y.reshape(-1),
        desc = mod_desc.astype(str),
        ch_desc = ch_desc.astype(str),
        asic_desc = asic_desc.astype(str),
        pixel_desc = pixel_desc.astype(str),
        index_desc = index_desc.astype(str),
        charge = pixels.astype(int),
        imgs=image_list))
    if charge_heatmap is not True: 
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
                        <span style="font-size: 15px;">Peak to peak: @charge ADC </span>
                    </div>
                </div>
                """
            )
    else:
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
                        <span style="font-size: 15px; font-weight: bold;"> Pixel @pixel_desc, Index @index_desc</span>
                    </div>
                    <div>
                        <span style="font-size: 15px;">Charge: @charge Photoelectrons</span>
                    </div>
                </div>
                """
            )
    if charge_heatmap is True:
        name = "Charge (Photoelectrons)"
    else:
        name = "Peak to Peak (ADC)"
    hm = figure(title=f"Run: {runID} Event: {event_index} {name}", tools=[hover], toolbar_location="below", toolbar_sticky=False, x_range=(0, 4), y_range=(0, 4))
    hm.image(image=[pixels.reshape(-1, int(np.sqrt(total_cells)))],
                color_mapper=color_mapper, dh=[4], dw=[4], x=[0], y=[0])
    recta = hm.rect('x', 'y', fill_alpha=0, line_alpha=0, width=.1, height=.1, source=source)
    grs = recta.select(dict(type=GlyphRenderer))
    for glyph in grs:
        if isinstance(glyph.glyph, Rect):
            rect_renderer = glyph
    hover.renderers = [rect_renderer]
    hm.axis.visible = False
    if charge_heatmap is True:
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), title="Charge", label_standoff=12, border_line_color=None, location=(0,0))
    else:
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), title="Peak to Peak", label_standoff=12, border_line_color=None, location=(0,0))
    hm.add_layout(color_bar, 'right')
    os.environ['DISPLAY'] = ':0.0'
    save(hm)

def make_heatmap_raw(runID, pedID,  event_index, datadir, charge_heatmap=True):
    modlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 103, 106, 107, 108, 111, 112, 114, 115, 119, 121, 123, 124, 125, 126]
    savedir = f"{datadir}/run{runID}"
    quicklook = analysis_quicklook.Quicklooker(pedID, runID, modlist, datadir, savedir)
    
    print("Generating camera image pixel mappings")

    waveforms = np.zeros((quicklook.n_pixels, quicklook.n_samples), dtype=np.ushort) # FIXME
    #quicklook.calreader.GetR1Event(int(event_index), waveforms) FIXME
    reader = target_io.WaveformArrayReader(f"{datadir}/run{runID}.fits") # FIXME
    reader.GetR0Event(int(event_index), waveforms) # FIXME
    ev_analysis = analysis_quicklook.EventAnalyzer(waveforms, quicklook.n_samples, quicklook.n_pixels)
    charge = ev_analysis.charge
    #waveforms = apply_gains(waveforms) FIXME

    mod_nums = [100,111,114,107,6,115,123,124,112,7,119,108,110,121,8,103,125,126,106,9,4,5,1,3,2] #FIXME
    fpm_nums = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24] #FIXME

    # create 5x5 coordinates
    fpm_pos = np.mgrid[0:5,0:5]
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
    rot_ch_nums = np.rot90(ch_nums, k=2)
    ch_to_pos = dict(zip(ch_nums.reshape(-1), np.arange(64)))
    rot_ch_to_pos = dict(zip(rot_ch_nums.reshape(-1), np.arange(64)))
    num_columns = 5
    total_cells = 5 * 5 * 64
    indices = np.arange(total_cells).reshape(-1, int(np.sqrt(total_cells)))
    pixels = np.zeros(total_cells)
    pixels_waveforms = np.zeros((total_cells, quicklook.n_samples))
    mod_desc = np.zeros(total_cells, dtype=int)
    ch_desc = np.zeros(total_cells, dtype=int)
    asic_desc = np.zeros(total_cells, dtype=int)
    for index, mod in enumerate(modlist):
        i, j = fpm_to_pos[mod_to_fpm[mod]]
        ch_map = dict()
        if j % 2 == 0:
            ch_map = rot_ch_to_pos
        else:
            ch_map = ch_to_pos
        j = num_columns - 1 - j
        pix_ind = np.array(indices[(8*i):8*(i+1), (8*j):8*(j+1)]).reshape(-1)
        for asic in range(4):
            for ch in range(16):
                grid_ind = int(pix_ind[ch_map[asic * 16 + ch]])
                pixels_waveforms[grid_ind, :] = waveforms[64 * index + asic * 16 + ch, :]
                if charge_heatmap is True:
                    pixels[grid_ind] = charge[64 * index + asic * 16 + ch]
                else:
                    pixels[grid_ind] = ev_analysis.peak_to_peak[64 * index + asic * 16 + ch]
                mod_desc[grid_ind] = int(mod)
                ch_desc[grid_ind] = int(ch)
                asic_desc[grid_ind] = int(asic)

    image_list = []
    for i in tqdm(range(total_cells)):
        plt.plot(pixels_waveforms[i])
        plt.xlabel("Time (ns)", fontsize=14)
        plt.ylabel("Amplitude (ADC Counts)", fontsize=14)
        plt.tick_params(labelsize=10)
        fig = plt.gcf()
        #imgdata = StringIO()
        imgdata = BytesIO()
        fig.savefig(imgdata, format="png")
        imgdata.seek(0)
        image = "data:image/png;base64," + urllib.parse.quote(base64.b64encode(imgdata.getbuffer()))
        image_list.append(image)
        fig.clf()
    print("Generating html...")
    if charge_heatmap is True:
        output_filename = f"{savedir}/run{runID}_ev{event_index}_Interactive_Heatmap_charge_raw.html" #FIXME: _charge_pe
    else:
        output_filename = f"{savedir}/run{runID}_ev{event_index}_Interactive_Heatmap_peaktopeak.html"
    output_file(output_filename, title=f"Run: {runID} Event: {event_index}")
    color_mapper = LinearColorMapper(palette="Viridis256", low=0, high=np.amax(pixels))
    Y, X = np.mgrid[0.05:3.95:40j, 0.05:3.95:40j]
    source = ColumnDataSource(data=dict(
        x = X.reshape(-1),
        y = Y.reshape(-1),
        desc = mod_desc.astype(str),
        ch_desc = ch_desc.astype(str),
        asic_desc = asic_desc.astype(str),
        charge = pixels.astype(int),
        imgs=image_list))
    if charge_heatmap is not True: 
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
                        <span style="font-size: 15px;">Peak to peak: @charge ADC </span>
                    </div>
                </div>
                """
            )
    else:
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
                        <span style="font-size: 15px;">Charge: @charge ADC ns</span>
                    </div>
                </div>
                """
            )
    if charge_heatmap is True:
        name = "Charge (ADC Counts)"
    else:
        name = "Peak to Peak (ADC)"
    hm = figure(title=f"Run: {runID} Event: {event_index} {name}", tools=[hover], toolbar_location="below", toolbar_sticky=False, x_range=(0, 4), y_range=(0, 4))
    hm.image(image=[pixels.reshape(-1, int(np.sqrt(total_cells)))],
                color_mapper=color_mapper, dh=[4], dw=[4], x=[0], y=[0])
    recta = hm.rect('x', 'y', fill_alpha=0, line_alpha=0, width=.1, height=.1, source=source)
    grs = recta.select(dict(type=GlyphRenderer))
    for glyph in grs:
        if isinstance(glyph.glyph, Rect):
            rect_renderer = glyph
    hover.renderers = [rect_renderer]
    hm.axis.visible = False
    if charge_heatmap is True:
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), title="Charge", label_standoff=12, border_line_color=None, location=(0,0))
    else:
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), title="Peak to Peak", label_standoff=12, border_line_color=None, location=(0,0))
    hm.add_layout(color_bar, 'right')
    os.environ['DISPLAY'] = ':0.0'
    save(hm)

if __name__ == "__main__":
    arg_list = sys.argv
    runID = int(arg_list[1])
    pedID = 328587
    eventID = int(arg_list[2])
    raw = False
    username = pwd.getpwuid(os.getuid()).pw_name
    if username == "ctauser":
        DATADIR = "/data/local_outputDir"
    else:
        DATADIR = "/data/wipac/CTA/target5and7data/runs_320000_through_329999"
    if len(arg_list) > 3:
        raw = bool(int(arg_list[3]))
        if len(arg_list) > 4:
            gain = bool(int(arg_list[4]))

    charge_heatmap = True
    if raw is False:
        make_heatmap_cal(runID, pedID, eventID, DATADIR, gain=gain)
    else:
        make_heatmap_raw(runID, pedID, eventID, DATADIR)
    print("Done!")


"""WIP: Need to replace almost all instances of quicklook.n_pixels = 1536 with total_cells = 5 * 5 * 64 = 1600"""

