# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'brycectauitest.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import psct_reader
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, \
     QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel
from psct_reader import WaveformArrayReader as Reader
import numpy as np

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.photo = QtWidgets.QLabel(self.centralwidget)
        self.photo.setGeometry(QtCore.QRect(0, 0, 841, 511))
        self.photo.setText("")
        self.photo.setPixmap(QtGui.QPixmap("Previous.jpg"))
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")
        self.Previous = QtWidgets.QPushButton(self.centralwidget)
        self.Previous.setGeometry(QtCore.QRect(0, 510, 411, 41))
        self.Previous.setObjectName("Previous")
        self.Next = QtWidgets.QPushButton(self.centralwidget)
        self.Next.setGeometry(QtCore.QRect(410, 510, 391, 41))
        self.Next.setObjectName("Next")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.Next.clicked.connect(self.show_Next)
        self.Previous.clicked.connect(self.show_Previous)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Previous.setText(_translate("MainWindow", "Previous"))
        self.Next.setText(_translate("MainWindow", "Next"))

    def show_Next(self):
        self.photo.setPixmap(QtGui.QPixmap(self.getImage)       ##I stopped here trying to get it to change the image displayed

    def show_Previous(self):
        self.photo.setPixmap(QtGui.QPixmap("imgs/Previous.jpg"))

    def getImage(self):
        image = createImage()

        
class createImage(QWidget):
    def __init__(self):
        super().__init__()
        #
        #
        #

        reader = Reader('/home/bryce/CTA/Bryces_CTA_project/data/wipac/CTA/target5and7data/runs_320000_through_329999/cal327583.r1')
        reader.get_event(1)
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

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
