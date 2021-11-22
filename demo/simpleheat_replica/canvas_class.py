import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPainter, QBrush, QPen, QLinearGradient, QColor, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint, QRect

def RGBA_to_QPixmap(matrix):
    # The ARGB32 QImage format requires a matrix in BGRA format for its constructor
    BGRA_matrix = cv2.cvtColor(matrix, cv2.COLOR_RGBA2BGRA)
    return QPixmap(QImage(BGRA_matrix, matrix.shape[1], matrix.shape[0], QImage.Format_ARGB32))

class Canvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

    def set_hm(self, hm):
        self.hm = hm

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)

    def mouseMoveEvent(self, event):
        self.hm.add((event.x(), event.y()))
        self.setPixmap(RGBA_to_QPixmap(self.hm.generate_image()))

    def paintEvent(self, event):
        super().paintEvent(event)

