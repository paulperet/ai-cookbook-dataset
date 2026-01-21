# Anchor Boxes for Object Detection: A Practical Guide

## Introduction

In object detection, algorithms need to identify regions in an image that contain objects and adjust their boundaries to match the ground-truth bounding boxes. Different models use different region sampling strategies. This guide introduces **anchor boxes**—multiple bounding boxes with varying scales and aspect ratios centered on each pixel. We'll implement anchor box generation, labeling, and prediction workflows step-by-step.

## Prerequisites

First, let's set up our environment and import necessary libraries. We'll adjust printing precision for cleaner outputs.

```python
# For MXNet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
np.set_printoptions(2)  # Simplify printing accuracy
npx.set_np()
```

```python
# For PyTorch
%matplotlib inline
from d2l import torch as d2l
import torch
torch.set_printoptions(2)  # Simplify printing accuracy
```

## Step 1: Generating Multiple Anchor Boxes

Suppose our input image has height `h` and width `w`. We generate anchor boxes with different shapes centered on each pixel. Let:
- **Scale** `s ∈ (0, 1]`
- **Aspect ratio** `r > 0` (width-to-height ratio)

The width and height of an anchor box are `w * s * √r` and `h * s / √r`, respectively. Given a center position, an anchor box with known width and height is fully determined.

To generate multiple shapes efficiently, we use:
- A series of scales: `s₁, ..., sₙ`
- A series of aspect ratios: `r₁, ..., rₘ`

Instead of using all `n × m` combinations per pixel (which would create `w × h × n × m` boxes), we only consider combinations containing either `s₁` or `r₁`:

```
(s₁, r₁), (s₁, r₂), ..., (s₁, rₘ), (s₂, r₁), (s₃, r₁), ..., (sₙ, r₁)
```

This gives us `n + m - 1` anchor boxes per pixel, totaling `w × h × (n + m - 1)` boxes for the entire image.

### Implementation: The `multibox_prior` Function

Here's the implementation that generates anchor boxes given an input image, scales, and aspect ratios:

```python
# MXNet implementation
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, ctx=device)
    ratio_tensor = d2l.tensor(ratios, ctx=device)
    
    # Offsets to center anchors on pixels
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width
    
    # Generate center points
    center_h = (d2l.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (d2l.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = d2l.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)
    
    # Calculate widths and heights
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    
    # Create anchor manipulations
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2
    
    # Generate output grid
    out_grid = d2l.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)
```

```python
# PyTorch implementation
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    
    # Offsets to center anchors on pixels
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width
    
    # Generate center points
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    
    # Calculate widths and heights
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:]))) \
                   * in_height / in_width
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    
    # Create anchor manipulations
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2
    
    # Generate output grid
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

### Testing Anchor Box Generation

Let's test our function on a sample image:

```python
# Load image and get dimensions
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
print(f"Image dimensions: height={h}, width={w}")

# Create dummy input data
X = torch.rand(size=(1, 3, h, w))  # Batch size 1, 3 channels

# Generate anchor boxes
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
print(f"Anchor box tensor shape: {Y.shape}")
```

The output shape `(1, 2042040, 4)` shows we have:
- Batch size: 1
- Number of anchor boxes: 2,042,040
- 4 coordinates per box (x_min, y_min, x_max, y_max)

### Visualizing Anchor Boxes

Let's examine anchor boxes centered at a specific pixel (250, 250):

```python
# Reshape to access boxes by pixel location
boxes = Y.reshape(h, w, 5, 4)  # 5 boxes per pixel
print(f"Anchor box at (250, 250, 0): {boxes[250, 250, 0, :]}")
```

The coordinates are normalized (0-1). To visualize them, we need a helper function:

```python
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""
    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    
    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

Now let's visualize the 5 anchor boxes centered at pixel (250, 250):

```python
d2l.set_figsize()
bbox_scale = d2l.tensor((w, h, w, h))  # Scale back to original dimensions
fig = d2l.plt.imshow(img)

# Show anchor boxes with their parameters
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

You'll see that the blue anchor box (scale=0.75, aspect ratio=1) nicely surrounds the dog in the image.

## Step 2: Understanding Intersection over Union (IoU)

To quantify how well an anchor box matches a ground-truth bounding box, we use **Intersection over Union (IoU)**, also known as the Jaccard index. For two bounding boxes A and B:

```
IoU(A, B) = Area(A ∩ B) / Area(A ∪ B)
```

IoU ranges from 0 (no overlap) to 1 (perfect overlap).

### Implementing IoU Calculation

```python
# MXNet implementation
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    
    # Calculate intersection coordinates
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    
    # Calculate intersection and union areas
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    
    return inter_areas / union_areas
```

```python
# PyTorch implementation
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    
    # Calculate intersection coordinates
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    
    # Calculate intersection and union areas
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    
    return inter_areas / union_areas
```

## Step 3: Labeling Anchor Boxes for Training

For training, each anchor box needs:
1. **Class label**: The class of the object it contains (or background)
2. **Offset label**: The adjustment needed to match the ground-truth bounding box

### Assigning Ground-Truth Boxes to Anchor Boxes

We need to match each anchor box with the most appropriate ground-truth bounding box. The algorithm:

1. Create an IoU matrix `X` where `x[i,j]` = IoU between anchor box `A_i` and ground-truth box `B_j`
2. Find the largest IoU and assign that ground-truth box to that anchor box
3. Discard the corresponding row and column
4. Repeat until all ground-truth boxes are assigned
5. For remaining anchor boxes, assign ground-truth boxes only if IoU > threshold

```python
# MXNet implementation
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)
    
    # Initialize assignment map
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    
    # Assign based on threshold
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= iou_threshold)[0]
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    
    # Greedy assignment for remaining ground-truth boxes
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    
    return anchors_bbox_map
```

```python
# PyTorch implementation
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)
    
    # Initialize assignment map
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    
    # Assign based on threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    
    # Greedy assignment for remaining ground-truth boxes
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    
    return anchors_bbox_map
```

### Calculating Offset Labels

For an anchor box `A` assigned to ground-truth box `B`, we calculate normalized offsets:

Given centers `(x_a, y_a)` and `(x_b, y_b)`, widths `w_a` and `w_b`, heights `h_a` and `h_b`:

```
offset_x = ((x_b - x_a) / w_a - μ_x) / σ_x
offset_y = ((y_b - y_a) / h_a - μ_y) / σ_y
offset_w = (log(w_b / w_a) - μ_w) / σ_w
offset_h = (log(h_b / h_a) - μ_h) / σ_h
```

Typically: `μ_x = μ_y = μ_w = μ_h = 0`, `σ_x = σ_y = 0.1`, `σ_w = σ_h = 0.2`

```python
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset
```

### Complete Labeling Function

Now let's combine everything into a single function that labels anchor boxes:

```python
# MXNet implementation
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size