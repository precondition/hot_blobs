from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPainter, QBrush, QLinearGradient, QColor, QPixmap, QImage
from typing import Dict, Union, Tuple, Iterable
from collections import Counter
import numpy as np
import copy
from math import ceil
import cv2
from time import time

def qt_image_to_array(img: QImage, share_memory=False) -> np.ndarray:
    """
    Creates a numpy array from a QImage.

   If share_memory is True, the numpy array and the QImage is shared.
   Be careful: make sure the numpy array is destroyed before the image,
   otherwise the array will point to unreserved memory!!

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

class PresetGradients:

    default_gradient = {
        0.4: "blue",
        0.6: "cyan",
        0.7: "lime",
        0.8: "yellow",
        1.0: "red"
    }

    sweet_period = {
        0.00: "#3f51b1",
        0.35: "#5a55ae",
        0.40: "#7b5fac",
        0.45: "#8f6aae",
        0.50: "#a86aa4",
        0.62: "#cc6b8e",
        0.75: "#f18271",
        0.87: "#f3a469",
        1.00: "#f7c978"
    }

    fabled_sunset = {
       0.25: "#231557",
       0.40: "#44107A",
       0.67: "#FF1361",
       1.00: "#FFF800"
    }

    dense_water = {
        0.00: "#3ab5b0",
        0.31: "#3d99be",
        1.00: "#56317a"
    }

    sublime = {
        0.0: QColor(13, 15, 140),
        0.4: QColor(13, 15, 140),
        0.7: QColor(217, 70, 239),
        1.0: QColor(135, 4, 23)
    }

    argon = {
        0.0: "#03001e",
        0.4: "#7303c0",
        0.6: "#ec38bc",
        1.0: "#fdeff9"
    }

    king_yna = {
        0.4: "#1a2a6c",
        0.7: "#b21f1f",
        1.0: "#fdbb2d"
    }

    complete_list = [default_gradient, sweet_period, fabled_sunset, dense_water, sublime, argon, king_yna]


class Heqtmap:
    """
    Class for producing a heatmap in the form of a QImage.
    The data used to produce the heatmap is input as an iterable of
    non-unique (x, y) tuples representating the coordinates of the
    heatmap image — for info, the point (0,0) is in the top left corner.
    """

    def __init__(self, w, h):
        self._width: int = w
        self._height: int = h
        self._radius: int = 25
        self._r2: int = 75
        self._max: int = 1
        self._data: Counter = Counter()
        self._stamp: QImage = None
        self._chosen_gradient: Dict[float, Union[str, QColor]] = PresetGradients.default_gradient


    def data(self, data: Iterable[Tuple[int]], compute_max: bool = True) -> 'Heqtmap':
        self._data = Counter(data)
        if compute_max:
            if len(self._data) > 0:
                self._max = max(self._data.values())
        return self

    def add(self, data: Iterable[Tuple[int]]) -> 'Heqtmap':
        if isinstance(data, tuple) and isinstance(data[0], int):
            data = [data]
        for data_point in data:
            assert len(data_point) == 2, f"The data point \"{data_point}\" is not a point in 2D space! Did you forget to enclose your (x, y) tuple in a list when feeding it into this method?"
            if self._data[data_point] > self._max:
                self._max = self._data[data_point]
        self._data.update(data)
        return self

    def clear(self) -> 'Heqtmap':
        self._data = Counter()
        return self

    def max(self, maximum: int) -> 'Heqtmap':
        assert maximum > 0, "The maximum must be a strictly positive value!"
        self._max = maximum
        return self

    def _sigma(self, x: Union[int, float]) -> int:
        """
        This is the sigma function used by OpenCV to compute the Gaussian standard
        deviation from the kernel size aka the blur factor in our case.
        Source: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#getgaussiankernel

        The output must be an integer and to do that `ceil` is used to round up and thus avoid 0.
        """
        return ceil(0.3*((x-1)*0.5 - 1) + 0.8)

    def stamp(self, r: int = 25, blur: int = 51) -> 'Heqtmap':
        """
        Create a grayscale blurred circle image that we'll use like
        a rubber stamp for drawing points
        """
        assert r > 0, "The radius must be a strictly positive value!"
        assert isinstance(blur, int) and blur > 0 and blur % 2 == 1, \
                f"The blur factor must be a strictly positive odd integer."
        # In accordance to the three-sigma rule, all we need to capture 99.7% of a circle
        # blurred with a Gaussian filter is a bigger circle whose radius is that of the
        # smaller one plus 3σ of the used Gaussian function.
        # Using a bigger σ value is pointless as the alpha channel is not precise enough to capture
        # anything (other than pure transparency) beyond the big circle described before.
        # With that in mind, the most optimal way to frame/bound such a blurred circle
        # is thus to create a square for which the sides equal to the diameter of the bigger circle.
        self._r2 = r+self._sigma(blur)*3
        circle_canvas = QImage(
                2*self._r2,
                2*self._r2,
                QImage.Format.Format_ARGB32)
        # It's important to use a format that supports transparency

        circle_canvas.fill(QColor("transparent"))
        painter = QPainter(circle_canvas)
        center = QPoint(self._r2, self._r2)

        painter.setBrush(QBrush(Qt.black))
        painter.drawEllipse(center, r, r)
        painter.end()

        # TODO: Converting back and forth just to apply a blur seems a little wasteful.
        image_data: np.ndarray = qt_image_to_array(circle_canvas)
        blurred_image_data: np.ndarray = cv2.GaussianBlur(image_data, (blur,blur), self._sigma(blur))
        self._stamp: QImage = QImage(
                blurred_image_data,
                blurred_image_data.shape[1],
                blurred_image_data.shape[0],
                QImage.Format_ARGB32)
        return self

    def resize(self, w: int, h: int) -> 'Heqtmap':
        self._width = w
        self._height = h
        return self

    def gradient(self, grad: Dict[float, Union[str, QColor]]) -> 'Heqtmap':
        self._chosen_gradient = grad
        return self

    def _gen_gradient(self, grad: Dict[float, str] = PresetGradients.default_gradient) -> np.ndarray:
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

    def _colorized(self, pixels: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        assert gradient.shape == (256, 1, 4), f"The gradient must be a 256x1x4 numpy array! The provided array is {gradient.shape}."
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

    def get_image(self, min_opacity: float = 0.05) -> QImage:
        assert 0 <= min_opacity and min_opacity <= 1, "The minimum opacity must be a float in the range [0;1]."
        if self._stamp is None:
            start = time()
            self.stamp()
            end = time()
            print(f"Setting the stamp took {(end - start)*1000:.10f} milliseconds!")
        canvas = QImage(self._width, self._height, QImage.Format.Format_ARGB32)
        canvas.fill(QColor("transparent"))
        painter = QPainter(canvas)
        start = time()
        for (x, y), value in self._data.items():
            # Draw a black stamp at x, y whose opacity is set to the normalized value
            painter.setOpacity(min(max(value/self._max, min_opacity), 1))
            painter.drawImage(QPoint(x-self._r2, y-self._r2), self._stamp)
        painter.end()
        end = time()
        print(f'Drawing all the stamps took {(end - start)*1000:.10f} milliseconds!')

        start = time()
        image_data: np.ndarray = qt_image_to_array(canvas)
        end = time()
        print(f'Converting the image to an array took {(end - start)*1000:.10f} milliseconds!')
        start = time()
        grad: np.ndarray = self._gen_gradient(self._chosen_gradient)
        end = time()
        print(f"Creating the gradient took {(end - start)*1000:.10f} milliseconds!")
        start = time()
        colorized_image_data = self._colorized(image_data, grad)
        end = time()
        print(f"Colorizing the heatmap took {(end - start)*1000:.10f} milliseconds!")
        img: QImage = QImage(colorized_image_data, colorized_image_data.shape[1], colorized_image_data.shape[0], QImage.Format_ARGB32)
        return img

