from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPainter, QBrush, QPen, QLinearGradient, QColor, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint, QRect
import sys
sys.path.append("../..")
from canvas_class import RGBA_to_QPixmap
from hot_blobs import Heatmap
import numpy as np
from simpleheat_data import data
from ui import Ui_MainWindow


def qt_image_to_array(img: QImage, share_memory=False) -> np.ndarray:
    """
    Creates a numpy array from a QImage.

   If share_memory is True, the numpy array and the QImage is shared.
   Be careful: make sure the numpy array is destroyed before the image,
   otherwise the array will point to unreserved memory!

   NOTE: Despite the format of the image being Format_(A)RGB32,
         each element of the matrix is BGRA.
    """
    assert isinstance(img, QImage), "img must be a QtGui.QImage object but is actually {}".format(type(img))
    assert img.format() == QImage.Format.Format_ARGB32 or img.format() == QImage.Format.Format_RGB32, \
        "img format must be QImage.Format.Format_ARGB32 or QImage.Format.Format_RGB32, got: {}".format(img.format())

    img_size = img.size()
    buffer = img.constBits()
    n_bits_image  = img_size.width() * img_size.height() * img.depth()
    buffer.setsize(n_bits_image//8)

    assert img.depth() == 32, "unexpected image depth: {}".format(img.depth())

    # Note the different width height parameter order!
    arr = np.ndarray(shape  = (img_size.height(), img_size.width(), img.depth()//8),
                     buffer = buffer,
                     dtype  = np.uint8)

    if share_memory:
        return arr
    else:
        return copy.deepcopy(arr)
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
