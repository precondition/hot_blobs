# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1026, 687)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.header = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.header.sizePolicy().hasHeightForWidth())
        self.header.setSizePolicy(sizePolicy)
        self.header.setWhatsThis("")
        self.header.setAccessibleName("")
        self.header.setWordWrap(True)
        self.header.setOpenExternalLinks(True)
        self.header.setObjectName("header")
        self.verticalLayout.addWidget(self.header)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        self.verticalLayout.addItem(spacerItem)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QtCore.QSize(1002, 602))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setLineWidth(3)
        self.frame.setObjectName("frame")
        self.canvas = Canvas(self.frame)
        self.canvas.setEnabled(True)
        self.canvas.setGeometry(QtCore.QRect(1, 1, 1000, 600))
        self.canvas.setMouseTracking(True)
        self.canvas.setTabletTracking(True)
        self.canvas.setAutoFillBackground(True)
        self.canvas.setObjectName("canvas")
        self.frame_options = QtWidgets.QFrame(self.canvas)
        self.frame_options.setGeometry(QtCore.QRect(740, -1, 261, 78))
        self.frame_options.setAutoFillBackground(True)
        self.frame_options.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_options.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_options.setLineWidth(3)
        self.frame_options.setObjectName("frame_options")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_options)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.options = QtWidgets.QWidget(self.frame_options)
        self.options.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.options.sizePolicy().hasHeightForWidth())
        self.options.setSizePolicy(sizePolicy)
        self.options.setObjectName("options")
        self.gridLayout = QtWidgets.QGridLayout(self.options)
        self.gridLayout.setObjectName("gridLayout")
        self.blurSlider = QtWidgets.QSlider(self.options)
        self.blurSlider.setMinimum(10)
        self.blurSlider.setMaximum(50)
        self.blurSlider.setProperty("value", 15)
        self.blurSlider.setOrientation(QtCore.Qt.Horizontal)
        self.blurSlider.setObjectName("blurSlider")
        self.gridLayout.addWidget(self.blurSlider, 1, 1, 1, 1)
        self.blurLabel = QtWidgets.QLabel(self.options)
        self.blurLabel.setObjectName("blurLabel")
        self.gridLayout.addWidget(self.blurLabel, 1, 0, 1, 1)
        self.radiusSlider = QtWidgets.QSlider(self.options)
        self.radiusSlider.setAccessibleName("")
        self.radiusSlider.setMinimum(10)
        self.radiusSlider.setMaximum(50)
        self.radiusSlider.setProperty("value", 25)
        self.radiusSlider.setOrientation(QtCore.Qt.Horizontal)
        self.radiusSlider.setObjectName("radiusSlider")
        self.gridLayout.addWidget(self.radiusSlider, 0, 1, 1, 1)
        self.radiusLabel = QtWidgets.QLabel(self.options)
        self.radiusLabel.setAccessibleName("")
        self.radiusLabel.setObjectName("radiusLabel")
        self.gridLayout.addWidget(self.radiusLabel, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.options, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.frame)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Simpleheat Demo Replica"))
        self.header.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Heqtmap</span><span style=\" font-size:12pt;\"> is a tiny and fast heatmap library. More on </span><a href=\"https://github.com/precondition/simpleheat\"><span style=\" font-size:12pt; text-decoration: underline; color:#0000ff;\">precondition / hot_blobs</span></a></p></body></html>"))
        self.blurLabel.setText(_translate("MainWindow", "Blur"))
        self.radiusLabel.setText(_translate("MainWindow", "Radius"))
from canvas_class import Canvas
