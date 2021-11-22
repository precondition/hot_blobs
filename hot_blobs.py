from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPainter, QBrush, QLinearGradient, QColor, QPixmap, QImage
from typing import Dict, Union, Tuple, Iterable, Hashable
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
   otherwise the array will point to unreserved memory!

   NOTE: Despite the format of the image being Format_(A)RGB32,
         each element of the matrix is BGRA.
    """
    assert isinstance(img, QImage), "img must be a QtGui.QImage object but is actually {}".format(type(img))

    img_size = img.size()
    buffer = img.constBits()
    n_bits_image = img_size.width() * img_size.height() * img.depth()
    buffer.setsize(n_bits_image//8)

    # Note the different width height parameter order!
    arr = np.ndarray(shape = (img_size.height(), img_size.width(), img.depth()//8),
                 buffer = buffer,
                 dtype  = np.uint8)

    if share_memory:
        return arr
    else:
        return copy.deepcopy(arr)

class PresetGradients:

    # default_gradient = {
        # 0.4: "blue",
        # 0.6: "cyan",
        # 0.7: "lime",
        # 0.8: "yellow",
        # 1.0: "red"
    # }

    default_gradient = {
        0.4: "#0000FF",
        0.6: "#00FFFF",
        0.7: "#00FF00",
        0.8: "#FFFF00",
        1.0: "#FF0000"
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


class Heatmap:
    """
    Class for producing a heatmap in the form of a QImage.
    The data used to produce the heatmap is input as an iterable of
    non-unique (x, y) tuples representing the coordinates of the
    heatmap image — for info, the point (0,0) is in the top left corner.
    """

    def __init__(self, w, h):
        self.width: int = w
        self.height: int = h

        # Use the setter methods to change the value of these fields
        self._radius: int = 25
        self._r2: int = 75
        self._max: int = 1

        self._data: Counter = Counter()
        # h×w×1 array of values between 0 and 1 (initialized to None to detect that no stamp has been generated yet)
        self._stamp: np.ndarray = None
        self._chosen_gradient: Dict[float, Union[str, QColor]] = PresetGradients.default_gradient

    def data(self, data: Iterable[Tuple[int]], compute_max: bool = True) -> 'Heatmap':
        self._data = Counter(data)
        if compute_max:
            if len(self._data) > 0:
                self._max = max(self._data.values())
        return self

    def add(self, data: Iterable[Tuple[int]]) -> 'Heatmap':
        """Adds one or more data points to the heatmap

        Parameters
        ----------
        data : Tuple[int] or Iterable[Tuple[int]]
            The new data point(s) to add. It can be either a single tuple `(x, y)`
            or an iterable of tuples `[(a, b), (c, d), ...]`.

        Returns
        -------
        Object
            Instance of the Heatmap object for function chaining

        Raises
        -----
        AssertionError
            If the data points are not hashable or points in 2D space.
        """

        if isinstance(data, tuple) and isinstance(data[0], int):
            data = [data]
        for data_point in data:
            assert isinstance(data_point, Hashable) and len(data_point) == 2,\ 
            f"The data point \"{data_point}\" is not a point in 2D space!"
            if self._data.get(data_point, 0)+1 > self._max:
                self._max = self._data.get(data_point, 0)+1
        self._data.update(data)
        return self

    def clear(self) -> 'Heatmap':
        """Clears all heatmap data points.

        Returns
        -------
        Object
            Instance of the Heatmap object for function chaining
        """

        self._data = Counter()
        return self

    def max(self, maximum: int) -> 'Heatmap':
        """Sets the maximum value with which to normalise the heatmap data points.

        Params
        ------
        maximum: int
            Strictly positive value to use as maximum.

        Returns
        -------
        Object
            Instance of the Heatmap object for function chaining

        Raises
        ------
        AssertionError
            if `maximum` <= 0
        """

        assert maximum > 0, "The maximum must be a strictly positive value!"
        self._max = maximum
        return self

    @staticmethod
    def _sigma(x: Union[int, float]) -> int:
        """TODO
        This is the sigma function used by OpenCV to compute the Gaussian standard
        deviation from the kernel size aka the blur factor in our case.
        Source: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#getgaussiankernel

        The output must be an integer and to do that `ceil` is used to round up and thus avoid 0.

        Params
        ------
        x: int or float
            blur factor

        Returns
        -------
            int
                Gaussian standard deviation
        """
        return ceil(0.3*((x-1)*0.5 - 1) + 0.8)

    def stamp(self, r: int = 25, blur: int = 51) -> 'Heatmap':
        """Creates a grayscale blurred circle image used like a rubber stamp for 
        drawing points.

        Params
        ------
        r: int, default=25
            Radius of the circle

        blur: int, default=51
            Kernel size for the Gaussian blur. 

        Returns
        -------
        Object
            Instance of the Heatmap object for function chaining

        Raises
        -----
        AssertionError
            if `r` <= 0;
            if blur is not a strictly positive integer.

        Notes
        -----
        The `blur` parameter is used to set the kernel size used for the Gaussian blur and must thus be
        odd integer (in order to have a matrix with a true center) but this method abstracts this
        constraint away by accepting any strictly positive integer, be it even or odd, and simply
        increments any even `blur` value received.
        """
        assert r > 0, "The radius must be a strictly positive value!"
        assert isinstance(blur, int) and blur > 0, f"The blur factor must be a strictly positive integer."
        blur += (blur-1) % 2  # Force blur to be odd

        # In accordance to the three-sigma rule, all we need to capture 99.7% of a circle
        # blurred with a Gaussian filter is a bigger circle whose radius is that of the
        # smaller one plus 3σ of the used Gaussian function.
        # Using a bigger σ value is pointless as the alpha channel is not precise enough to capture
        # anything (other than pure transparency) beyond the big circle described before.
        # With that in mind, the most optimal way to frame/bound such a blurred circle
        # is thus to create a square for which the sides equal to the diameter of the bigger circle.
        self._r2 = round(r)+self._sigma(blur)*3
        empty_circle_canvas = np.zeros([2*self._r2, 2*self._r2], dtype=float)
        # `circle_canvas` has odd dimensions so it doesn't have a true center point
        center = (self._r2, self._r2)
        circle_canvas = cv2.circle(empty_circle_canvas, center, r, 1.0, cv2.FILLED)
        blurred_image_data: np.ndarray = cv2.GaussianBlur(circle_canvas, (blur, blur), self._sigma(blur))
        self._stamp = blurred_image_data
        return self

    def resize(self, w: int, h: int) -> 'Heatmap':
        """Resizes the dimensions of the heatmap.

        Data points that are cropped out after the resize *remain* in the 
        internal dataset of points.

        Params
        ------
        w: int
            New width.

        h: int
            New height.

        Returns
        -------
        Object
            Instance of the Heatmap object for function chaining
        """
        self.width = w
        self.height = h
        return self

    def gradient(self, grad: Dict[float, Union[str, QColor]]) -> 'Heatmap':
        """Sets the color gradient to use in the heatmap to `grad`.

        Params
        ------
        grad: Dict[float, str] or Dict[float, (r, g, b)]
            Dictionary defining the color stops of the linear gradient.
            The keys are the stops (floats between 0 and 1) and the values are the color.

        Returns
        -------
        Object
            Instance of the Heatmap object for function chaining
        """
        self._chosen_gradient = grad
        return self

    def _gen_gradient(self, grad: Dict[float, str] = PresetGradients.default_gradient) -> np.ndarray:
        """Generates a 256×1×4 linear gradient image in RGBA format.

        Params
        ------
        grad: Dict[float, str] or Dict[float, (r, g, b)], default=PresetGradients.default_gradient
            Dictionary defining the color stops of the linear gradient.
            The keys are the stops (floats between 0 and 1) and the values are the color.

        Returns
        -------
        np.ndarray
            256×1×4 numpy array with the alpha channel ([:, :, 3]) all equal to 255.
        """

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

    def _colorized(self, opacities: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Colorizes the black and white heatmap (`opacities`) with the color 
        of the chosen gradient (`gradient`).

        A bijective function maps each possible opacity value to each possible 
        gradient color.

        For example, if a "pixel" in the black and white heatmap picture has a 
        value of 200, it will be transformed into the 200th color of the linear 
        color gradient but its opacity will drop from 255 to 200, the original 
        opacity value of this pixel.
        [200] -> [80, 08, 5, 255] -> [80, 08, 5, 200]

        Params
        ------
        opacities: np.ndarray
            self.w×self.h×1 array of uint8's representing the black and white heatmap.

        grad: Dict[float, str] or Dict[float, (r, g, b)], default=PresetGradients.default_gradient
            Dictionary defining the color stops of the linear gradient.
            The keys are the stops (floats between 0 and 1) and the values are the color.

        Returns
        -------
        np.ndarray
            self.w×self.h×4 array of uint8's representing the colorised heatmap in RGBA format.
        """

        #TODO: Change the alpha value of the gradient in _gen_gradient?
        assert gradient.shape == (256, 1, 4), \
                f"The gradient must be a 256x1x4 numpy array! " \
                "The provided array has dimensions {gradient.shape}."
        # Mask to select only the alpha value of the gradient image which is a 256×1×4 (h×w×depth) image
        mask = np.array([[[False, False, False, True]]]*256)
        indices = np.indices(gradient.shape)[0]
        # Replace the constant 255 alpha value of the gradient by the index of the row
        gradient[mask] = indices[mask]
        # Get column vector of the alpha values of each pixel in the grayscale heatmap
        # A fourth extra axis of length 1 is produced so we index it to remove it
        # and thus get an array that has the same dimensions as the background image.
        return gradient[opacities][:, :, 0, :]

    def _draw_stamp(self, canvas: np.ndarray, x: int, y: int, opacity: float) -> None:
        if y >= canvas.shape[0] or x >= canvas.shape[1]:
            # The stamp is beyond the south or east borders of the canvas
            return
        cropped_canvas: np.ndarray = canvas[max(0, y):y+self._stamp.shape[0], max(0, x):x+self._stamp.shape[1]]
        # -min(0, x) returns either 0 if x >= 0 or |x| if x < 0, so it's always a positive value
        cropped_stamp: np.ndarray = self._stamp[-min(0, y):-min(0, y)+cropped_canvas.shape[0],
                                                -min(0, x):-min(0, x)+cropped_canvas.shape[1]]
        cropped_stamp_trns = cropped_stamp * opacity
        # Alpha composition/blending "OVER" operation
        stamped_cropped_canvas: np.ndarray = cropped_canvas + cropped_stamp_trns * (1 - cropped_canvas)
        # Copying the stamp overlaid onto the canvas into the RoI
        canvas[max(0, y):y+self._stamp.shape[0], max(0, x):x+self._stamp.shape[1]] = stamped_cropped_canvas

    #TODO: rename to "draw_image"
    def get_image(self, min_opacity: float = 0.05) -> QImage:
        """Generates the final heatmap image.

        This is the main method that generates the stamp (template of the hot blobs) if it hasn't already been generated,
        draws all data points onto a transparent canvas and colorises these data points to produces the final result.

        EVERYTHING is redrawn every time this method is called, there is no "differential" drawing in place or fetching
        the previous result if no new data has been fed in since the last call.

        Params
        ------
        min_opacity: float
            Float between 0 and 1 (inclusive) to set the minimum opacity of the overall heatmap overlay.

        Returns
        -------
        QImage
            ARGB32 QImage of the final heatmap overlay.
        """
        assert 0 <= min_opacity and min_opacity <= 1, "The minimum opacity must be a float in the range [0;1]."
        if self._stamp is None:
            start = time()
            self.stamp()
            end = time()
            print(f"Setting the stamp took {(end - start)*1000:.10f} milliseconds!")
        canvas = np.zeros([self.height, self.width], dtype=float)
        start = time()
        for (x, y), value in self._data.items():
            # Draw a black stamp at x, y whose opacity is set to the normalized value
            opacity = min(max(value/self._max, min_opacity), 1)
            self._draw_stamp(canvas, x-self._r2, y-self._r2, opacity)
        end = time()
        print(f'Drawing all the stamps took {(end - start)*1000:.10f} milliseconds!')
        # Convert from float64 matrix with values in the domain [0;1] 
        # to a uint8 matrix with values in the domain [0;255].
        # This is a faster way of doing `(canvas*255).astype(np.uint8)`
        image_data: np.ndarray = cv2.convertScaleAbs(src=canvas, alpha=255)
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
