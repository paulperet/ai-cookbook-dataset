# Object Detection and Bounding Boxes

In previous sections, we focused on image classification, where we assumed a single major object per image. However, real-world images often contain multiple objects. Object detection is the computer vision task that involves identifying *what* objects are present (classification) and *where* they are located (localization). This technique is critical for applications like autonomous driving, robotics, and security systems.

In this guide, we'll introduce the fundamental concept of representing an object's location: the **bounding box**.

## Prerequisites

First, let's import the necessary libraries and set up our environment.

```python
%matplotlib inline
from d2l import torch as d2l
import torch
```

## Loading a Sample Image

We'll use a sample image containing a dog and a cat to demonstrate bounding boxes.

```python
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## Understanding Bounding Boxes

A bounding box is a rectangular region that encloses an object. There are two common ways to represent one:

1.  **Corner Coordinates (x1, y1, x2, y2):** The (x, y) coordinates of the top-left and bottom-right corners of the rectangle.
2.  **Center Coordinates (cx, cy, w, h):** The (x, y) coordinates of the box's center, along with its width and height.

We need functions to convert between these two representations. The input `boxes` is expected to be a 2D tensor with shape `(n, 4)`, where `n` is the number of boxes.

```python
#@save
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

## Defining Bounding Boxes for Our Image

Let's define the bounding boxes for the dog and cat in our image using the corner coordinate format. The coordinate origin `(0, 0)` is at the top-left corner of the image, with the x-axis increasing to the right and the y-axis increasing downwards.

```python
# bbox is short for bounding box
dog_bbox = [60.0, 45.0, 378.0, 516.0]  # [x1, y1, x2, y2]
cat_bbox = [400.0, 112.0, 655.0, 493.0]
```

We can verify our conversion functions work correctly by performing a round-trip conversion.

```python
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

**Output:**
```
tensor([[True, True, True, True],
        [True, True, True, True]])
```

## Visualizing the Bounding Boxes

To draw the boxes on our image, we need a helper function to convert our bounding box format into the format used by Matplotlib's `Rectangle` patch.

```python
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert from (x1, y1, x2, y2) to Matplotlib format: ((x1, y1), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

Now, let's add the bounding boxes to our image. We'll use blue for the dog and red for the cat.

```python
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

You should see the dog and cat neatly enclosed within their respective colored rectangles.

## Summary

*   **Object detection** involves identifying objects and their spatial locations within an image.
*   An object's location is typically represented by a rectangular **bounding box**.
*   The two primary bounding box representations are **corner coordinates** `(x1, y1, x2, y2)` and **center coordinates** `(cx, cy, w, h)`. We can convert between them using the functions defined above.

## Exercises

1.  Find another image and try to draw a bounding box around an object. Compare the time it takes to label a bounding box versus assigning a category label. Which task is more time-consuming?
2.  Why must the innermost dimension (the last axis) of the input `boxes` tensor for our conversion functions always be 4?