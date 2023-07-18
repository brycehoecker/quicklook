import sys
import time
import numpy as np

from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as tB
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from psct_reader import WaveformArrayReader as Reader

#import mainProgram as mPro

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt

    

#TO DO

#make window full screen at launch                                      COMPLETED
#decode ID number into i, j coordinates to print specific image         COMPLETED 
#running log of pixel statistics : charge to 2 sig figs                 
#forward and backward bytearray                                         COMPLETED ish                                           
#create function setters & getters

#Make it look pretty and neat


#ApplicationWindow Variables
# reader - calls initReader function that initializes WaveFormArrayReader object 
# image - image data from WaveFormArrayReader
# run - run number from WaveFormArrayReader
# ev - event number from WaveFormArrayReader
#
# layout - QGridLayout object that holds all objects
# iD - used to create labels for each subplot coordinates[4-i,j] in order to access by iD number
# grid - 5x5 array filled with iD numbers in order of creation to compare coordinates with iD number
# dataLog - QListWidget that displays the charge information when a pixel is clicked
# 
# firstWindow - QMainWindow object that creates the expanded view of a subplot when clicked 
# createImage - function that creates the big 5x5 image composed of subplots
#

def initReader(self):
    self.reader.get_event(self.eventNumber)
    self.image = self.reader.image                                                    #"image" contains all FITS data 
    self.run = self.reader.run
    self.ev = self.reader.ev   

class dataLog(QListWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(100)
        

    def setText(self,item):
        self.addItem(item)
        self.scrollToBottom()
        
class mainWindow(QMainWindow):           #MAIN WINDOW 
    def __init__(self):
        super().__init__()
        self._main = QWidget()
        self.setCentralWidget(self._main)
        self.layout = QGridLayout(self._main)  # (row , col)
#        self.firstwindow = firstWindow()
        
        self.iD = 0                                                      #creates image area for enlarged image 
        self.grid = np.ndarray(shape = (5,5))

        self.reader = Reader('/home/bryce/CTA/Bryces_CTA_project/data/wipac/CTA/target5and7data/runs_320000_through_329999/cal327583.r1')            #DIRECTORY PATH FOR CAL FILE
        self.eventNumber = 237310
        self.createImage()

        self.Vlayout = QVBoxLayout()
        self.layout.addLayout(self.Vlayout,0,4,0,4)
        
        self.createButton = QPushButton('Create')                
        self.createButton.clicked.connect(self.createImage)        
        self.Vlayout.addWidget(self.createButton)
        
        self.previousButton = QPushButton('Previous')
        self.previousButton.clicked.connect(self.prevImage)        
        self.Vlayout.addWidget(self.previousButton)
        
        self.nextButton = QPushButton('Next')
        self.nextButton.clicked.connect(self.nextImage)
        self.Vlayout.addWidget(self.nextButton)

        self.dataLog = dataLog()
        self.layout.addWidget(self.dataLog,4,0,1,3)       #row , col , row span , col span 
        self.Vlayout.setContentsMargins(10,100,10,600)    #left, top, right, bottom

    def nextImage(self):
        if (self.eventNumber + 5 < self.reader.n_events):
            self.eventNumber += 5
            self.createImage()
        
    def prevImage(self):
        if (self.eventNumber >= 5 ):
            self.eventNumber -= 5
            self.createImage()
        
    def createImage(self):
        initReader(self)
        maxZ = np.nanmax(self.image)
        fig = plt.gcf()                                                           #fig is created, gcf calls .figure()        
        if fig.get_size_inches().all() == np.array([18., 15.]).all():
            plt.close(fig)
            fig = plt.figure(figsize=(18, 15))                                    #fig is instanciated as 18 Width x 15 Length (inches)
            self.static_canvas = FigureCanvas(fig)                                     #pyQT widget is created, FigureCanvas has a 18x15 object on it             

        gs = gridspec.GridSpec(5, 5)                                              #creates 5x5 grid that can accept subplots
        gs.update(wspace=0.04, hspace=0.04)
       
        for i in range(5):
            for j in range(5):
                sub_image = self.image[i*8:i*8+8, j*8:j*8+8]                            #takes portion of reader.image and creates sub_image
                ax = plt.subplot(gs[4 - i, j], picker = True, label = self.iD)          #creates the axis with label = iD            
                c = ax.pcolormesh(sub_image, vmin=0, vmax=maxZ, cmap="viridis")    #blends axis and sub_image 
                ax.axis("off")
                ax.set_aspect("equal")

                self.grid[4-i,j] = self.iD
                self.iD += 1
               
        fig.subplots_adjust(right=0.71, left=.285, top=0.9, bottom=0.1)
        self.cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        self.cbar = fig.colorbar(c, cax=self.cbar_ax)
        self.cbar.set_label("Charge (Photoelectrons)", rotation=270, size=24, labelpad=24)
        self.cbar_ax.tick_params(labelsize=15)
        fig.suptitle(f"Run {self.run} Event {self.ev}", fontsize=30)
        fig.canvas.mpl_connect("pick_event",self.nextWindow)
        
        self.layout.addWidget(self.static_canvas,1,1)#,1,2)
        self.toolBar = NavigationToolbar(self.static_canvas, self)
      #  self.layout.addWidget(self.toolBar,2,1,1,2)


        

    def nextWindow(self,event):
        self.module = event.artist.get_label()
        self.charge = self.toolBar._mouse_event_to_message(event.mouseevent)
        self.dataLog.setText(f"Module: {self.module} , Charge: {self.charge[15:19]}" )
               
####       for item, value in cbarax.items():
####           print (item, ":", value, "\n")
######       
##
##        self.iD = float(event.artist.get_label())
####        print("id: ", self.iD, "\n")
####        print(self.grid)
##        for i in range(5):
##            for j in range(5):
##                compare = self.grid[i,j]
##                if (compare == self.iD):
####                    print("found: ", i , j )
##                    firstWindow.firstWindow.i = 4 - i
##                    self.firstWindow.j = j
##                    self.firstWindow.iD = self.iD
##                    break
##       # self.firstWindow.createImage(self)
##       # mainWidget.addWidget(self.firstWindow)
##   
####        mainWidget.setCurrentIndex(mainWidget.currentIndex()+1)
##        
##class firstWindow(QMainWindow):     #ENLARGED IMAGE WINDOW
##    def __init__(self):
##        super().__init__()
##        self._main = QWidget()
##        self.setCentralWidget(self._main)
##        self.layout = QGridLayout(self._main)  # (row , col)
##        self.reader = None
##
##
##        self.previousButton = QPushButton('Previous')
##        self.previousButton.clicked.connect(self.prevWindow)        
##        self.layout.addWidget(self.previousButton,3,0)
##     
##        self.nextButton = QPushButton('Next')
####      self.nextButton.clicked.connect( function that goes to next square )
##        self.layout.addWidget(self.nextButton,3,1)
##
##       # self.dataLog = mainWindow.dataLog
##        
##        self.layout.addWidget(self.dataLog,4,4,4,1)
##
##        
##        self.iD = None
##        self.i = None                        # coordinates for ID numbers
##        self.j = None                        # 
##    
##        
##    def createImage(self,mainWindow):
##             
##        new_fig = plt.figure(figsize=(18, 15))
##        self.new_canvas = FigureCanvas(new_fig)
##     
##        sub_image = mainWindow.image[self.i*8:self.i*8+8, self.j*8:self.j*8+8]              # iD is going to be decoded into coordinates (i,j)
##        new_gs = gridspec.GridSpec(1,1)
##        new_gs.update(wspace=0.04, hspace=0.04)
##
##        maxZ = np.nanmax(mainWindow.image)
##        ax = plt.subplot(new_gs[0,0])   
##        c = ax.pcolormesh(sub_image, vmin=0, vmax=maxZ, cmap="viridis")
##        ax.axis("off")
##        ax.set_aspect("equal")
##                             
##        cbar_ax = new_fig.add_axes([0.85, 0.15, 0.05, 0.7])
##        cbar = new_fig.colorbar(c, cax=cbar_ax)
##        cbar.set_label("Charge (Photoelectrons)", rotation=270, size=24, labelpad=24)
##        cbar_ax.tick_params(labelsize=20)
##        new_fig.suptitle(f"Run {mainWindow.run} Event {mainWindow.ev} ", fontsize=30)
##        
##        self.layout.addWidget(self.new_canvas,1,1,1,2)                  
##        self.layout.addWidget(NavigationToolbar(self.new_canvas, self),2,1,1,2)
##        
##    def prevWindow(self):
##        mainWidget.setCurrentIndex(mainWidget.currentIndex()-1)

      




if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    
    qapp = QApplication.instance()
    if not qapp:
        qapp = QApplication(sys.argv)

    mainWidget = QtWidgets.QStackedWidget()
    
    mainWindow = mainWindow()
##    firstWindow = fWin.firstWindow(mainWindow)

    
    mainWidget.addWidget(mainWindow)
 ##   mainWidget.addWidget(firstWindow)
    
    mainWidget.showMaximized()
    mainWidget.activateWindow()
    mainWidget.raise_()
    qapp.exec()

