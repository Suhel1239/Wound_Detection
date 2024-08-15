import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
import sys
from segment.predict import run
# from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
# from utils.plots import plot_one_box
from models.experimental import attempt_load
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from utils.segment.plots import plot_masks
from utils.segment.plots import plot_images_and_masks
import shutil

class Ui_MainWindow(object):
   
   default_normal = "/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/normal.jpg"
   default_seg = "/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/segmented.jpg"
   img_path = "/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/0016.png"
   #  Handling YOLOv7 ************************************************
   model_path = "/Users/suhelkhan/Downloads/yolov7_with_category_30_epochs/weights/best.pt"
   
   file_name = ""
   def change_img(self):
        self.img_path = self.browsefiles()

   def browsefiles(self):
        directory_path = "/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/results_pred/exp"
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
         try:
              shutil.rmtree(directory_path)
         except OSError as e:
              print(f"Error: {directory_path} : {e.strerror}")
         else:
              print(f"Directory '{directory_path}' does not exist.")

        fname = QFileDialog.getOpenFileName()
        self.file_name = fname[0]
        print(type(fname))
        # "/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/0016.png"
        run(weights="/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/yolov7_with_category_30_epochs/weights/best.pt",
          source=fname[0], project="/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/results_pred",)
        self.File_name.setText(fname[0])
        self.img_path = os.path.join(str(fname[0]))
        print(self.img_path)
        splitted =  self.file_name.split("/")
        seg_path  = os.path.join("/Users/suhelkhan/NHRI_woundDetect/PYQT5_Projects/yolov7/seg/results_pred/exp" , splitted[-1])
        self.normal_img.setPixmap(QtGui.QPixmap(self.img_path))
        self.seg_img.setPixmap(QtGui.QPixmap(seg_path))
        print(seg_path)
        return fname
    
   def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.File_name = QtWidgets.QLineEdit(self.centralwidget)
        self.File_name.setGeometry(QtCore.QRect(60, 30, 521, 31))
        self.File_name.setObjectName("File_name")

        # Browse Button ############
        self.browse_buttton = QtWidgets.QPushButton(self.centralwidget)
        self.browse_buttton.setGeometry(QtCore.QRect(590, 30, 100, 32))
        self.browse_buttton.setObjectName("browse_buttton")
        self.browse_buttton.clicked.connect(self.browsefiles)
        
        # For normal image
        self.normal_img = QtWidgets.QLabel(self.centralwidget)
        self.normal_img.setGeometry(QtCore.QRect(60, 200, 311, 291))
        self.normal_img.setText("")
        self.normal_img.setPixmap(QtGui.QPixmap(self.default_normal))
        self.normal_img.setScaledContents(True)
        self.normal_img.setObjectName("normal_img")

        # Segmented image
        self.seg_img = QtWidgets.QLabel(self.centralwidget)
        self.seg_img.setGeometry(QtCore.QRect(460, 200, 301, 291))
        self.seg_img.setText("")
        self.seg_img.setPixmap(QtGui.QPixmap(self.default_seg))
        self.seg_img.setScaledContents(True)
        self.seg_img.setObjectName("seg_img")

        ##### Segment Button
        # self.browse_buttton_2 = QtWidgets.QPushButton(self.centralwidget)
        # self.browse_buttton_2.setGeometry(QtCore.QRect(350, 120, 100, 32))
        # self.browse_buttton_2.setObjectName("browse_buttton_2")
        # self.browse_buttton_2.clicked.connect(self.change_img)

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

   def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.browse_buttton.setText(_translate("MainWindow", "Browse"))

        # self.browse_buttton_2.setText(_translate("MainWindow", "Segment"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
