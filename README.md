# Hot Blobs

A Python library for drawing heatmap overlays with the same look and feel as
[simpleheat](https://github.com/mourner/simpleheat) and [heatmap.js](https://github.com/pa7/heatmap.js).

![Glamor shot]()

A "heatmap" can take many forms but this library focuses only on a single kind. See the illustrated table below.
Geographic zoomable maps are not supported.

![Horizontal strip of different heatmap types with a red cross in the corner of unsupported heatmap types]()

## Reference

### Constructor

| Constructor | Description |
|-------------|-------------|
| `Heatmap(w: int, h: int)` | Constructs an empty `Heatmap` of dimensions `w`×`h` |

### Public Methods

| Return type | Method | Description |
|-------------|--------|-------------|
| hot_blobs.Heatmap | `data(data: Iterable[Tuple[int]], compute_max: bool = True)` | Initialise the heatmap data with `data` |
| hot_blobs.Heatmap | `add(data: Iterable[Tuple[int, int]]` | Adds one or more data points to the heatmap |
| hot_blobs.Heatmap | `clear()` | Clears all heatmap data points. |
| hot_blobs.Heatmap | `max(maximum: int)` | Sets the maximum value with which to normalise the heatmap data points. |
| hot_blobs.Heatmap | `stamp(r: int = 25, blur: int = 51)` | Creates a grayscale blurred circle image used like a rubber stamp for drawing points. |
| hot_blobs.Heatmap | `resize(w: int, h: int)` | Resizes the dimensions of the heatmap. |
| hot_blobs.Heatmap | `gradient(grad: Dict[float, Union[str, QColor]])` | Sets the color gradient to use in the heatmap to `grad`. |
| np.ndarray | `generate_image(min_opacity: float = 0.05)` | Generates the final heatmap image in the form of a `width`×`height`×4 numpy array in RGBA format. |

Each method contains a detailed docstring that you can consult anytime by typing `help(Heatmap.method_name)`.

The reason why all those "setter" methods return an instance of the Heatmap object is for purposes of function chaining.
This means that these two code blocks are perfectly equivalent:

```py
from hot_blobs import Heatmap
hm = Heatmap(600, 300)
hm.data(my_data)
hm.max(20)
hm.stamp(r=10, blur=63)
im = hm.generate_image(0.1)
```

```py
from hot_blobs import Heatmap
im = Heatmap(600, 300).data(my_data).max(20).stamp(r=10, blur=63).generate_image(0.1)
```

### Basic example

```py
import hot_blobs
from PIL import Image # Use any desired library to convert the RGBA matrix into an image

# You can either feed in a collection of non-unique 2D tuples
# or a dictionary that associates a value to each unique tuple.
observations = [(400, 300), (400, 300), (400, 300), (10, 20), (10, 20), (750, 530)]
observations = {(400, 300): 3, (10, 20): 2, (750, 530): 1}
heatmap_obj = hot_blobs.Heatmap(w=800, h=600).data(observations)
heatmap_matrix = heatmap_obj.generate_image()
Image.fromarray(heatmap_matrix).save("basic_heatmap_example.png")
```

See the `demo/` directory for more examples.


<!-- WIP
## Prior Art

This is not the first nor the only Python heatmap library, but I developed it because none of the available offerings suited me.

First, let's cite the №1 result you get when searching for "heatmap python" online: `seaborn`. While `seaborn` is a fine data-visualisation library that produces pretty heatmaps, they're not the kind of heatmap that you would overlay over a background picture.

-->