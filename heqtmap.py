from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPainter, QBrush, QLinearGradient, QColor, QPixmap, QImage
from typing import Dict, Union, Tuple, Iterable
from collections import Counter
import numpy as np
import copy
from math import ceil
import cv2
from time import time

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

class Heatmap:
    """
    Class for producing a heatmap in the form of a QImage.
    The data used to produce the heatmap is input as an iterable of 
    non-unique (x, y) tuples representating the coordinates of the 
    heatmap image — for info, the point (0,0) is in the top left corner.
    """

    default_gradient: Dict[float, str] = {
        0.4: "blue",
        0.6: "cyan",
        0.7: "lime",
        0.8: "yellow",
        1.0: "red"
    }

    def __init__(self, w, h):
        self._width: int = w
        self._height: int = h
        self._radius: int = 25
        self._r2: int = 75
        self._max: int = 1
        self._min : int = 0
        self._data: Counter = Counter()
        self._stamp: QImage = None
        self._bg_color: QColor = QColor("transparent")


    def set_data(self, data: Iterable[Tuple[int]], compute_max: bool = True) -> 'Heatmap':
        self._data = Counter(data)
        if compute_max:
            self._max = max(self._data.values())
        return self

    def set_max(self, maximum: int) -> 'Heatmap':
        self._max = maximum
        return self

    def get_max(self) -> int:
        return self._max

    def set_min(self, minimum: int) -> 'Heatmap':
        self._min = minimum
        return self

    def get_min(self) -> int:
        return self._min

    def set_bg_color(self, bg_color: QColor) -> 'Heatmap':
        self._bg_color = bg_color
        return self

    def get_bg_color(self) -> QColor:
        return self._bg_color

    def add(self, data: Iterable[Tuple[int]]) -> 'Heatmap':
        print(f"added data is {data}")
        self._data.update(data)
        for data_point in data:
            if self._data[data_point] > self._max:
                self._max = self._data[data_point]
        return self

    def clear(self) -> 'Heatmap':
        self._data = Counter()
        return self

    def set_stamp(self, r: int = 25, blur: int = 51) -> 'Heatmap':
        """
        Create a grayscale blurred circle image that we'll use like
        a rubber stamp for drawing points
        """

        assert isinstance(blur, int) and blur > 0 and blur % 2 == 1, \
                f"The blur factor must be a positive odd integer because it is used to determine the kernel size for the gaussian blur."
        # In accordance to the three-sigma rule, all we need to capture 99.7% of a circle 
        # blurred with a Gaussian filter is a bigger circle whose radius is that of the
        # smaller one plus 3σ of the used Gaussian function.
        # Using a bigger σ value is pointless as the alpha channel is not precise enough to capture 
        # anything (other than pure transparency) beyond the big circle described before.
        # With that in mind, the most optimal way to frame/bound such a blurred circle
        # is thus to create a square for which the sides equal to the diameter of the bigger circle.
        self._r2 = r+sigma(blur)*3
        circle_canvas = QImage(
                2*self._r2,
                2*self._r2,
                QImage.Format.Format_ARGB32)

        circle_canvas.fill(self._bg_color)
        painter = QPainter(circle_canvas)
        center = QPoint(self._r2, self._r2)
        focal_point = center
        painter.setBrush(QBrush(Qt.black))
        painter.drawEllipse(center, r, r)
        painter.end()

        image_data: np.ndarray = qt_image_to_array(circle_canvas)
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

    def resize(self, w: int, h: int) -> None:
        self._width = w
        self._height = h

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

    def get_image(self, min_opacity: float = 0.05) -> QImage:
        if self._stamp is None:
            start = time()
            self.set_stamp()
            end = time()
            print(f"Setting the stamp took {(end - start)*1000:.10f} milliseconds!")
        canvas = QImage(self._width, self._height, QImage.Format.Format_ARGB32)
        canvas.fill(self._bg_color)
        painter = QPainter(canvas)
        start = time()
        for (x, y), value in self._data.items():
            if value < self._min:
                continue
            painter.setOpacity(min(max(value/self._max, min_opacity), 1))
            self.draw_stamp(painter, QPoint(x-self._r2, y-self._r2))
        painter.end()
        end = time()
        print(f'Drawing all the stamps took {(end - start)*1000:.10f} milliseconds!')

        start = time()
        image_data: np.ndarray = qt_image_to_array(canvas)
        end = time()
        print(f'Converting the image to an array took {(end - start)*1000:.10f} milliseconds!')
        start = time()
        grad: np.ndarray = self.gradient(self.default_gradient)
        end = time()
        print(f"{grad=}")
        print(f"Creating the gradient took {(end - start)*1000:.10f} milliseconds!")
        start = time()
        colorised_image_data = self.colorized(image_data, grad)
        end = time()
        print(f"Colorizing the heatmap took {(end - start)*1000:.10f} milliseconds!")
        img: QImage = QImage(colorised_image_data, colorised_image_data.shape[1], colorised_image_data.shape[0], QImage.Format_ARGB32)
        return img


    def colorized(self, pixels: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        # Mask to select only the alpha value of the gradient image which is a 256×1×4 (h×w×depth) image 
        mask = np.array([[[False, False, False, True]]]*256)
        indices = np.indices(gradient.shape)[0]
        # Replace the constant 255 alpha value of the gradient by the index of the row
        gradient[mask] = indices[mask]
        # Get column vector of the alpha values of each pixel in the grayscale heatmap
        opacities = pixels[:, :, -1]
        # A fourth extra axis of length 1 is produced so we index it to remove it 
        # and thus get an array that has the same dimensions as `pixels.shape`
        return gradient[opacities][:, :, 0, :] 

