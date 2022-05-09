# Form implementation generated from reading ui file 'detect.ui'
#
# Created by: PyQt6 UI code generator 6.3.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


import sys
import os
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap
import yoloCustomObjectDetection as yolo


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        #logical parameter definition
        self.cap = cv2.VideoCapture(0)
        self.playerTimer = QTimer()
        self.playerTimer.timeout.connect(self.getImage)
        self.detector = yolo.MugDetection(capture_index=0, model_name='yolov5m.pt')
        self.plotFPS = False
        self.plotBox = False


        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(648, 544)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.video_output = QtWidgets.QLabel(self.centralwidget)
        self.video_output.setGeometry(QtCore.QRect(120, 0, 400, 300))
        self.video_output.setStyleSheet("")
        self.video_output.setText("")
        self.video_output.setObjectName("video_output")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(220, 380, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(16)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        self.start_detect = QtWidgets.QPushButton(self.centralwidget)
        self.start_detect.setGeometry(QtCore.QRect(220, 310, 191, 71))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(22)
        self.start_detect.setFont(font)
        self.start_detect.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.DefaultContextMenu)
        self.start_detect.setStyleSheet("")
        self.start_detect.setObjectName("start_detect")
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(220, 420, 111, 31))
        self.checkBox_2.setObjectName("checkBox_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 648, 30))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        #binding buttons and checkboxes
        self.start_detect.clicked.connect(self.startDetect)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.checkBox.setText(_translate("MainWindow", "Plot Box"))
        self.start_detect.setText(_translate("MainWindow", "Start Detect"))
        self.checkBox_2.setText(_translate("MainWindow", "Plot FPS"))

    def startDetect(self):
        self.playerTimer.start(0)

    def getImage(self):
        assert self.cap.isOpened()
        frame = self.detector(self.cap)
        show = cv2.resize(frame,(400,300))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.video_output.setPixmap(QtGui.QPixmap.fromImage(showImage))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    #ui.playerTimer.start(0)
    MainWindow.show()
    sys.exit(app.exec())