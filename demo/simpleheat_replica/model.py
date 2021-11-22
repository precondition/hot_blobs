from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QBrush, QPen, QLinearGradient, QColor, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint, QRect
import sys
sys.path.append("../..")
from canvas_class import RGBA_to_QPixmap
from hot_blobs import Heatmap, qt_image_to_array
from simpleheat_data import data
from ui import Ui_MainWindow


class TheWindow(QMainWindow):
    def __init__(self):
        super(TheWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        hashable_data = {(x, y) : value for x, y, value in data[1:]}
        self.hm = Heatmap(self.ui.canvas.width(), self.ui.canvas.height()).data(hashable_data).max(18)
        self.hm.add({(x, y) : value for x, y, value in data[0:1]})
        self.ui.canvas.set_hm(self.hm)
        self.ui.canvas.setPixmap(RGBA_to_QPixmap(self.hm.generate_image()))
        self.ui.blurSlider.valueChanged[int].connect(self.changeBlur)
        self.ui.radiusSlider.valueChanged[int].connect(self.changeRadius)

        self.radius = 25
        self.blur = 51


    def changeBlur(self, new_blur):
        self.blur = new_blur
        self.hm.stamp(r=self.radius, blur=self.blur)
        self.ui.canvas.setPixmap(RGBA_to_QPixmap(self.hm.generate_image()))

    def changeRadius(self, new_radius):
        self.radius = new_radius
        self.hm.stamp(r=self.radius, blur=self.blur)
        self.ui.canvas.setPixmap(RGBA_to_QPixmap(self.hm.generate_image()))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TheWindow()
    w.show()
    sys.exit(app.exec())
