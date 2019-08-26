import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTextEdit, QAction, QFileDialog, QPushButton, QLabel, QVBoxLayout, QHBoxLayout)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import pandas as pd
import numpy as np



class MyApp(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        btn1 = QPushButton(self)
        btn1.setText('Open Log File')
#        
#        
        btn1.clicked.connect(self.showDialog)
        
        self.lbl = QLabel('QFilename', self)
        self.lbl.setAlignment(Qt.AlignCenter)
        
        vbox = QVBoxLayout()
#        vbox.addLayout(hbox)1
        vbox.addStretch(1)
        vbox.addWidget(btn1)
        vbox.addWidget(self.lbl)
        vbox.addStretch(1)
        
        self.setLayout(vbox)
        
        self.setWindowTitle('File Selection')
        self.setGeometry(300, 300, 300, 200)
        self.show()

    def showDialog(self):

        self.fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        print("fname is ", self.fname)
        self.lbl.setText(self.fname[0])
        self.lbl.adjustSize()
        self.makeDF()
#        if fname[0]:
#            f = open(fname[0], 'r')
#
#            with f:
#                data = f.read()
#                self.textEdit.setText(data)

#    def fileSelected(self, text):
#
#        self.lbl.setText(fname)
#        self.lbl.adjustSize()
    def makeDF(self):
        self.df = pd.read_csv(self.fname[0], sep='\t')
        print(self.df)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())