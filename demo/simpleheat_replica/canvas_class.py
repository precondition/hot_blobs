from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPainter, QBrush, QPen, QLinearGradient, QColor, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint, QRect

class Canvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

    def set_hm(self, hm):
        self.hm = hm

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)

    def mouseMoveEvent(self, event):
        self.hm.add([(event.x(), event.y())])
        self.setPixmap(QPixmap(self.hm.get_image()))

    def paintEvent(self, event):
        super().paintEvent(event)

