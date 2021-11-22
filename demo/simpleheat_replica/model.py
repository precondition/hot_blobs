from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QBrush, QPen, QLinearGradient, QColor, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint, QRect
import sys
sys.path.append("../..")
from hot_blobs import Heatmap, qt_image_to_array
from simpleheat_data import data
from ui import Ui_MainWindow

class TheWindow(QMainWindow):
    def __init__(self):
        super(TheWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        hashable_data = {(x, y) : value for x, y, value in data}
        self.hm = Heatmap(self.ui.canvas.width(), self.ui.canvas.height()).data(hashable_data).max(18)
        self.ui.canvas.set_hm(self.hm)
        self.ui.canvas.setPixmap(QPixmap(self.hm.get_image()))
        self.ui.blurSlider.valueChanged[int].connect(self.changeBlur)
        self.ui.radiusSlider.valueChanged[int].connect(self.changeRadius)

        self.radius = 25
        self.blur = 51


    def changeBlur(self, new_blur):
        self.blur = new_blur + (new_blur % 2 == 0)
        self.hm.stamp(r=self.radius, blur=self.blur)
        self.ui.canvas.setPixmap(QPixmap(self.hm.get_image()))

    def changeRadius(self, new_radius):
        self.radius = new_radius
        self.hm.stamp(r=self.radius, blur=self.blur)
        self.ui.canvas.setPixmap(QPixmap(self.hm.get_image()))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TheWindow()
    w.show()
    sys.exit(app.exec())
