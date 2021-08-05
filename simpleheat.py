from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPainter, QBrush, QLinearGradient, QColor, QPixmap, QImage
from typing import Dict, Union, Tuple, Sequence
import numpy as np
import copy
from math import ceil
import cv2

def sigma(x: Union[int, float]) -> int:
    return ceil(0.3*((x-1)*0.5 - 1) + 0.8)

def inverse_sigma(x: Union[int, float]) -> int:
    return ceil(1/3*(20*x-7))

def qt_image_to_array(img: QImage, share_memory=False) -> np.ndarray:
    """ Creates a numpy array from a QImage.

        If share_memory is True, the numpy array and the QImage is shared.
        Be careful: make sure the numpy array is destroyed before the image, 
        otherwise the array will point to unreserved memory!!

        NOTE: Despite the format of the image being Format_ARGB32,
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

class SimpleHeat:

    default_gradient: Dict[float, str] = {
        0.4: "blue",
        0.6: "cyan",
        0.7: "lime",
        0.8: "yellow",
        1.0: "red"
    }

    def __init__(self, canvas):
        self._canvas = canvas
        self._width: int = canvas.width
        self._height: int = canvas.height
        if not isinstance(self._width, int):
            self._width: int = canvas.width()
            self._height: int = canvas.height()
        assert isinstance(self._height, int)
        self._radius: int = 25
        self._r2: int = 75
        self._max: int = 1
        self._data: Dict[Tuple[int], int] = {}
        self._stamp: QImage = None


    def data(self, data: Sequence[int]):
        for d in data:
            xy_tuple = tuple(d[:2])
            if len(d) < 3:
                value = 1
            else:
                value = d[2]
            self._data[xy_tuple] = self._data.get(xy_tuple, 0) + value
            if self._max < self._data[xy_tuple]:
                self._max = self._data[xy_tuple]
        print(f"{self._max=}")
        return self

    def max(self, maximum: int):
        self._max = maximum
        return self

    def add(self, data_point: Sequence[int]):
        xy_tuple = tuple(data_point[:2])
        if len(data_point) < 3:
            value = 1
        else:
            value = data_point[2]
        self._data[xy_tuple] = self._data.get(xy_tuple, 0) + value
        if self._max < self._data[xy_tuple]:
            self._max = self._data[xy_tuple]
        return self

    def clear(self):
        self._data = {}
        return self

    def radius(self, r=25, blur=51):
        # create a grayscale blurred circle image that we'll use for drawing points

        # In accordance to the three-sigma rule, all we need to capture 95% of a circle 
        # blurred with a Gaussian filter is a bigger circle whose radius is that of the
        # smaller one plus 2σ of the used Gaussian function.
        # Using 3σ is pointless as the alpha channel is not precise enough to capture 
        # anything (other than pure transparency) beyond the big circle described before.
        # With that in mind, the most optimal way to frame/bound such a blurred circle
        # is thus to create a square for which the sides equal to the diameter of the bigger circle.
        self._r2 = r+sigma(blur)*2
        circle_canvas = QImage(
                2*self._r2,
                2*self._r2,
                QImage.Format.Format_ARGB32)

        circle_canvas.fill(QColor("transparent"))
        painter = QPainter(circle_canvas)
        center = QPoint(self._r2, self._r2)
        focal_point = center
        painter.setBrush(QBrush(Qt.black))
        painter.drawEllipse(center, r, r)
        painter.end()

        image_data: np.ndarray = qt_image_to_array(circle_canvas)
        blur += blur % 2 == 0 # increment blur if it is even
        blurred_image_data: np.ndarray = cv2.GaussianBlur(image_data, (blur,blur), sigma(blur))
        self._stamp: QImage = QImage(
                blurred_image_data,
                blurred_image_data.shape[1],
                blurred_image_data.shape[0],
                QImage.Format_ARGB32)

        return self

    def draw_stamp(self, painter: QPainter, point: QPoint) -> None:
        painter.drawPixmap(point, QPixmap.fromImage(self._stamp))
        # painter.drawPixmap(point, QPixmap("stamp.png"))

    def resize(self) -> None:
        self._width = self._canvas.width()
        self._height = self._canvas.height()

    def gradient(self, grad: Dict[float, str]) -> np.ndarray:
        canvas: QImage = QImage(1, 256, QImage.Format.Format_RGB32)
        painter: QPainter = QPainter(canvas)
        first_point: QPoint = QPoint(0,0)
        second_point: QPoint = QPoint(0,256)
        gradient: QLinearGradient = QLinearGradient(first_point, second_point)
        canvas.width = 1
        canvas.height = 256

        for i in grad:
            gradient.setColorAt(i, QColor(grad[i]))

        brush: QBrush = QBrush(gradient)
        painter.setBrush(brush)
        painter.fillRect(QRect(first_point, second_point), brush)
        painter.end()

        return qt_image_to_array(canvas)

    def get_image(self, min_opacity=0.05) -> QImage:
        if self._stamp is None:
            self.radius()
        canvas = QImage(self._width, self._height, QImage.Format.Format_ARGB32)
        painter = QPainter(canvas)
        for xy_tuple in self._data:
            x: int = xy_tuple[0]
            y: int = xy_tuple[1]
            value: int = self._data[xy_tuple]
            painter.setOpacity(min(max(value/self._max, min_opacity), 1))
            self.draw_stamp(painter, QPoint(x-self._r2, y-self._r2))
        painter.end()

        image_data: np.ndarray = qt_image_to_array(canvas)
        grad: np.ndarray = self.gradient(self.default_gradient)
        self.colorize(image_data, grad)
        img: QImage = QImage(image_data, image_data.shape[1], image_data.shape[0], QImage.Format_ARGB32)
        return img


    def colorize(self, pixels: np.ndarray, gradient: np.ndarray) -> None:
        for i in range(len(pixels)):
            for j in range(len(pixels[i])):
                alpha_value: int = pixels[i][j][3]
                if alpha_value:
                    pixels[i][j][:-1] = gradient[alpha_value][0][:-1]

