import psct_reader
from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from psct_reader import WaveformArrayReader as Reader
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLineEdit, QPushButton, QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
import sys

class my_window(QMainWindow):
	def __init__(self):
		super(my_window, self).__init__()
		self.setGeometry(100, 0, 1700, 800)                    # (start window horizontal, start window vertical, horizontal, vertical)
		self.setWindowTitle("pSCT PyQt5 testing area")
		self.initUI()

	def initUI(self):
		self.label = QtWidgets.QLabel(self)
		self.label.setText("test label 1")
		self.label.move(50, 50)
		self.button1 = QtWidgets.QPushButton(self)
		self.button1.setText("Click here")
		self.button1.clicked.connect(self.clicked)
		
	def clicked(self):
		self.label.setText("This button was pressed...")
		self.update()

	def update(self):
		self.label.adjustSize()
		

def window():
	app = QApplication(sys.argv)
	win = my_window()
	win.show()
	sys.exit(app.exec())
	
window()
	

