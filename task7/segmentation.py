"""Task 6 segmentation functions bundled for standalone Task 7 use."""
import cv2
import numpy as np


def connected_components(binary_image: np.ndarray, connectivity: int = 8,
                         min_area: int = 1):
    """Return a label image and spatially ordered component bounding boxes.

    Nonzero pixels are foreground. Bounding boxes use x, y, width, height.
    Labels retain all components. The returned component list excludes regions
    smaller than min_area. Each entry retains its original component label.
    """
    if binary_image.ndim != 2 or binary_image.size == 0:
        raise ValueError("binary_image must be a nonempty 2D image")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if min_area < 1:
        raise ValueError("min_area must be positive")

    foreground = (binary_image > 0).astype(np.uint8)
    # Lecture 4, Part 1, page 34: first pass, equivalences, second pass.
    height, width = foreground.shape
    labels = np.zeros(foreground.shape, dtype=np.int32)
    parents = [0]

    def find_root(label):
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    previous_neighbors = [(-1, 0), (0, -1)]
    if connectivity == 8:
        previous_neighbors.extend([(-1, -1), (-1, 1)])

    # First pass propagates labels and records equivalence using parent links.
    for row in range(height):
        for column in range(width):
            if foreground[row, column] == 0:
                continue
            neighbors = []
            for row_offset, column_offset in previous_neighbors:
                y = row + row_offset
                x = column + column_offset
                if 0 <= y < height and 0 <= x < width and labels[y, x] > 0:
                    neighbors.append(find_root(int(labels[y, x])))
            if neighbors:
                smallest = min(neighbors)
                labels[row, column] = smallest
                for neighbor in neighbors:
                    parents[find_root(neighbor)] = smallest
            else:
                labels[row, column] = len(parents)
                parents.append(len(parents))

    # Second pass resolves equivalences and measures each final region.
    region_labels = {}
    regions = []
    for row, column in zip(*np.nonzero(foreground)):
        root = find_root(int(labels[row, column]))
        if root not in region_labels:
            region_labels[root] = len(regions) + 1
            regions.append({"label": len(regions) + 1, "x": int(column),
                            "y": int(row), "right": int(column) + 1,
                            "bottom": int(row) + 1, "area": 0})
        final_label = region_labels[root]
        labels[row, column] = final_label
        region = regions[final_label - 1]
        region["x"] = min(region["x"], int(column))
        region["y"] = min(region["y"], int(row))
        region["right"] = max(region["right"], int(column) + 1)
        region["bottom"] = max(region["bottom"], int(row) + 1)
        region["area"] += 1

    components = []
    for region in regions:
        right = region.pop("right")
        bottom = region.pop("bottom")
        region["width"] = right - region["x"]
        region["height"] = bottom - region["y"]
        if region["area"] >= min_area:
            components.append(region)
    components.sort(key=lambda component: (component["x"], component["y"]))
    return labels, components


def prepare_character_binary(gray_image: np.ndarray, smooth_size: int = 3,
                             background_size: int = 31, offset: float = 4,
                             median_size: int = 5):
    """Apply local-mean thresholding to a grayscale text-line crop.

    A foreground pixel exceeds its local mean by at least offset gray levels.
    Returns the smoothed grayscale image, local mean and median-filtered mask.
    """
    if gray_image.ndim != 2 or gray_image.size == 0:
        raise ValueError("gray_image must be a nonempty 2D grayscale image")
    for size in (smooth_size, background_size, median_size):
        if not isinstance(size, (int, np.integer)) or size < 1 or size % 2 == 0:
            raise ValueError("Filter sizes must be positive odd integers")

    smoothed = cv2.blur(gray_image.astype(np.float64), (smooth_size, smooth_size),
                        borderType=cv2.BORDER_REPLICATE)
    local_mean = cv2.blur(smoothed, (background_size, background_size),
                          borderType=cv2.BORDER_REPLICATE)
    binary = (smoothed >= local_mean + offset).astype(np.uint8) * 255
    binary = cv2.medianBlur(binary, median_size)
    return smoothed, local_mean, binary
