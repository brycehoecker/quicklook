import psct_reader
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys 
from psct_reader import WaveformArrayReader as Reader
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt


class mainWindow(QWidget):
    def __init__(self):
        super().__init__()
        mainLayout = QVBoxLayout()

        self.eventNumber = QLineEdit()
        mainLayout.addWidget(QLabel('Event Number:'))
        mainLayout.addWidget(QLabel(self.eventNumber))

        self.directoryPath = QLineEdit()
        mainLayout.addWidget(QLabel('Directory:'))
        mainLayout.addWidget(QLabel(self.directoryPath))

        #print(directoryPath.text())
        #print(eventNumber.text)

        self.createButton = QPushButton('Create')
        mainLayout.addWidget(self.createButton)

        self.createButton.clicked.connect(self.getImage)
        self.setLayout(mainLayout)


    def getImage(self):
        image = createImage()

        
class createImage(QWidget):
    def __init__(self):
        super().__init__()
        #
        #
        #
        global counter
        counter += 1
        reader = Reader('/home/bryce/CTA/Bryces_CTA_project/data/wipac/CTA/target5and7data/runs_320000_through_329999/cal327583.r1')
        
        reader.get_event(counter)
        image = reader.image
        run = reader.run
        ev = reader.ev

        maxZ = np.nanmax(image)
        fig = plt.gcf()
        if fig.get_size_inches().all() == np.array([18., 15.]).all():
            plt.close(fig)
            fig = plt.figure(figsize=(18, 15))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.02, hspace=0.02)
        for i in range(5):
            for j in range(5):
                sub_image = image[i*8:i*8+8, j*8:j*8+8]
                ax = plt.subplot(gs[4-i, j])
                c = ax.pcolormesh(sub_image, vmin=0, vmax=maxZ, cmap="viridis")
                ax.axis("off")
                ax.set_aspect("equal")
        fig.subplots_adjust(right=0.7, left=0.29, top=0.9, bottom=0.0)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_label("Charge (Photoelectrons)", rotation=270, size=22, labelpad=24)
        cbar_ax.tick_params(labelsize=20)
        fig.suptitle(f"Run {run} Event {ev}", fontsize=30)

        self.resize(1800, 800)
        fig.show()

global counter
counter = 0
app = QApplication(sys.argv)
main = mainWindow()
main.show()
sys.exit(app.exec_())

        
