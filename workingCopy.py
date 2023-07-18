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


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QGridLayout(self._main)  # (row , col)

        self.eventNumber = QLineEdit()
       # self.eventNumber.setMaxLength(6)
       # self.eventNumber.resize(10,20)
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
        fig = plt.gcf()
        if fig.get_size_inches().all() == np.array([18., 15.]).all():
            plt.close(fig)
            fig = plt.figure(figsize=(18, 15))
            
            static_canvas = FigureCanvas(fig)

        layout.addWidget(static_canvas,1,1,1,2)

        gs = gridspec.GridSpec(5, 5)                                #creates 5x5 grid that can accept subplots
        gs.update(wspace=0.04, hspace=0.04)
        
        
        grid = np.ndarray(shape=(5,5))
        iD = 0        
              
  
        for i in range(5):
            for j in range(5):
                sub_image = image[i*8:i*8+8, j*8:j*8+8]             #takes portion of reader.image and creates sub_image
                ax = plt.subplot(gs[4 - i, j], label = iD)          #creates a 
                #try to set gs[4 - i, j] = to something
                #ax.set_xlabel(4-i)
                #ax.set_ylabel(j)
                c = ax.pcolormesh(sub_image, vmin=0, vmax=maxZ, cmap="viridis")
                ax.axis("off")
                ax.set_aspect("equal")
                #print(ax.get_xlabel)
                grid[4- i, j] = iD
                iD += 1
                
                

                
        layout.addWidget(NavigationToolbar(static_canvas, self))
        
        fig.subplots_adjust(right=0.71, left=.285, top=0.9, bottom=0.1)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_label("Charge (Photoelectrons)", rotation=270,
                       size=24, labelpad=24)
        cbar_ax.tick_params(labelsize=20)
        fig.suptitle(f"Run {run} Event {ev}", fontsize=30)
        
        def onclick(event):
            if event.inaxes :
                x = event.inaxes.get_gid
                fig = event.inaxes.get_gridspec()
                fig = event.inaxes.get_subplotspec()
               # fig.
              #  str(x)
               # fig.get_subplot_params
                print(x)
               
                               
        fig.canvas.mpl_connect("button_press_event",onclick)
        

    #    self._dynamic_ax = dynamic_canvas.figure.subplots()
     #   t = np.linspace(0, 10, 101)
        # Set up a Line2D.
      #  self._line, = self._dynamic_ax.plot(t, np.sin(t + time.time()))
    #    self._timer = dynamic_canvas.new_timer(50)
     #   self._timer.add_callback(self._update_canvas)
     #   self._timer.start()

   # def _update_canvas(self):
    #    t = np.linspace(0, 10, 101)
        # Shift the sinusoid as a function of time.
     #   self._line.set_data(t, np.sin(t + time.time()))
    #    self._line.figure.canvas.draw()


if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = ApplicationWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
