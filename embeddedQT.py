import sys
import time
import numpy as np

from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from psct_reader import WaveformArrayReader as Reader
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt


class ApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QWidget()
        self.setCentralWidget(self._main)
        layout = QGridLayout(self._main)  # (row , col)
        
        imageArea =QStackedWidget()
        layout.addWidget(imageArea,1,1,1,2)
        
        
                               
        self.eventNumber = QLineEdit()
        layout.addWidget(QLabel('Event Number:'),0,0)        
        layout.addWidget(self.eventNumber,0,1,1,0)
        
        self.directoryPath = QLineEdit()
        layout.addWidget(QLabel('Directory:'),0,2)
        layout.addWidget(self.directoryPath,0,3)

        self.createButton = QPushButton('Create')
        layout.addWidget(self.createButton,0,4)
        
        self.previousButton = QPushButton('Previous')
        layout.addWidget(self.previousButton,3,0)
        
        self.nextButton = QPushButton('Next')
        layout.addWidget(self.nextButton,3,1)


        reader = Reader('/home/bryce/CTA/Bryces_CTA_project/data/wipac/CTA/target5and7data/runs_320000_through_329999/cal327583.r1')
        reader.get_event(0)
        image = reader.image #"image" contains all FITS data 
        run = reader.run
        ev = reader.ev
        
        maxZ = np.nanmax(image)
        fig = plt.gcf()                                                           #fig is created, gcf calls .figure()
        
        if fig.get_size_inches().all() == np.array([18., 15.]).all():
            plt.close(fig)
            fig = plt.figure(figsize=(18, 15))                                    #fig is instanciated as 18 Width x 15 Length (inches)
            
            static_canvas = FigureCanvas(fig)                                     #pyQT widget is created, FigureCanvas has a 18x15 object on it

                                                                    
        #layout.addWidget(static_canvas,1,1,1,2)
        imageArea.addWidget(static_canvas)
        
        
        gs = gridspec.GridSpec(5, 5)                                              #creates 5x5 grid that can accept subplots
        gs.update(wspace=0.04, hspace=0.04)
        
      
        iD = 0        
        grid = np.ndarray(shape=(5,5))     
        for i in range(5):
           for j in range(5):
               sub_image = image[i*8:i*8+8, j*8:j*8+8]                            #takes portion of reader.image and creates sub_image
               ax = plt.subplot(gs[4 - i, j], picker = True, label = iD)          #creates the axis with label = iD 
               
               c = ax.pcolormesh(sub_image, vmin=0, vmax=maxZ, cmap="viridis")    #blends axis and sub_image 
               ax.axis("off")
               ax.set_aspect("equal")
                
               grid[4-i, j] = iD 
               iD += 1

                
        layout.addWidget(NavigationToolbar(static_canvas, self))
        
        fig.subplots_adjust(right=0.71, left=.285, top=0.9, bottom=0.1)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_label("Charge (Photoelectrons)", rotation=270, size=24, labelpad=24)
        cbar_ax.tick_params(labelsize=20)
        fig.suptitle(f"Run {run} Event {ev}", fontsize=30)
        
        def onclick(event):                                 # event attributes : name, canvas, guiEvent, xy cord in pixels,
            
            
            artist = event.artist        
            iD = artist.get_label()
            #gs = artist.get_gridspec()


            new_fig = plt.figure(figsize=(18, 15))
            new_canvas = FigureCanvas(new_fig)
            
            #layout.addWidget(new_canvas,1,1,1,2)
            imageArea.addWidget(new_canvas)
            imageArea.setCurrentWidget(1)

            sub_image = image[0:8, 0:8]                             # iD is going to be decoded into coordinates (i,j)
            new_gs = gridspec.GridSpec(1,1)
            new_gs.update(wspace=0.04, hspace=0.04)
            ax = plt.subplot(new_gs[0,0])   
            c = ax.pcolormesh(sub_image, vmin=0, vmax=maxZ, cmap="viridis")
            ax.axis("off")
            ax.set_aspect("equal")

            
            layout.addWidget(NavigationToolbar(new_canvas, self))
            cbar_ax = new_fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = new_fig.colorbar(c, cax=cbar_ax)
            cbar.set_label("Charge (Photoelectrons)", rotation=270, size=24, labelpad=24)
            cbar_ax.tick_params(labelsize=20)
            new_fig.suptitle(f"Run {run} Event {ev}", fontsize=30)
            
            
                       
            
                                                
            
            
        fig.canvas.mpl_connect("pick_event",onclick,)
        



if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    
##    qapp = QApplication.instance()
##    if not qapp:
##        qapp = QApplication(sys.argv)
##
##
##    
##
##    mainWidget = QtWidgets.QStackedWidget()
##        
##    mainWindow = ApplicationWindow()
##       # firstWindow = firstWindow()
##        
##    mainWidget.addWidget(mainWindow)
##      #  mainWidget.add(firstWindow)    
##    mainWindow.show()
##    mainWindow.activateWindow()
##    mainWindow.raise_()
##    qapp.exec()
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
        mainWidget = QtWidgets.QStackedWidget()

    app = ApplicationWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
