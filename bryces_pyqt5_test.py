import psct_reader
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys 
from psct_reader import WaveformArrayReader as Reader
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt


class mainWindow(QWidget):                                                  #
    def __init__(self):                                                     #
        super().__init__()                                                  #
        mainLayout = QVBoxLayout()                                          # Currently QVBoxLayout instead of QFormLayout
        #mainLayout = QFormLayout()
        self.eventNumber = QLineEdit()                                      #
        mainLayout.addWidget(QLabel('Event Number:'))                       #
        mainLayout.addWidget(QLabel(self.eventNumber))                      #

        self.directoryPath = QLineEdit()                                    #
        mainLayout.addWidget(QLabel('Directory:'))                          #
        mainLayout.addWidget(QLabel(self.directoryPath))                    #

        #print(directoryPath.text())
        #print(eventNumber.text)
        
        #mainLayout = QFormLayout()
        self.btn = QPushButton("Choose from list")
        self.btn.clicked.connect(self.getItem)
        
        self.le = QLineEdit()
        mainLayout.addWidget(self.btn,self.le)
        self.btn1 = QPushButton("get name")
        self.btn1.clicked.connect(self.gettext)

        self.user_number1 = QLineEdit()
        mainLayout.addWidget(self.btn1,self.le1)
        self.btn2 = QPushButton("Enter an Numberr:")
        self.btn2.clicked.connect(self.getint)

        self.le2 = QLineEdit()
        mainLayout.addWidget(self.btn2,self.le2)
        self.setLayout(layout)
        self.setWindowTitle("Input Dialog demo")

    def getItem(self):
        items = ("0", "1", "2", "3 etc...")

        item, ok = QInputDialog.getItem(self, "select input dialog", "list of languages", items, 0, False)
        if ok and item:
            self.le.setText(item)
    def gettext(self):
        text, ok = QInputDialog.getText(self, 'Text Input Dialog', 'Enter your name:')

        if ok:
            self.le1.setText(str(text))
            
    def getint(self):
        num,ok = QInputDialog.getInt(self,"integer input dualog","enter a number")

        if ok:
            self.le2.setText(str(num))


        self.createButton = QPushButton('Create')                           #
        mainLayout.addWidget(self.createButton)                             #

        self.createButton.clicked.connect(self.getImage)                    #
        self.setLayout(mainLayout)                                          #


    def getImage(self):                                                     #
        image = createImage()                                               #


        
class createImage(QWidget):
    def __init__(self):
        super().__init__()


        #   Should make reader.get_event(12345) change based on user inputs
        #
        #

        reader = Reader('/home/bryce/CTA/Bryces_CTA_project/data/wipac/CTA/target5and7data/runs_320000_through_329999/cal327583.r1')
        reader.get_event(btn2)
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

    def getDouble(self):
        num,ok = QInputDialog.getInt(self,"Number input dialog","enter a number")
        
        if ok:
            self.le2.setText(str(num))



app = QApplication(sys.argv)
main = mainWindow()
main.show()
sys.exit(app.exec_())

        
