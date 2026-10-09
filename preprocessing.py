import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Contrast Enhancement Functions

def contrast_stretching(input_image: np.ndarray, a: float = None, b: float = None, show_transform: bool = False) -> np.ndarray:
    '''
    f(r) = alpha * r + beta
    By default, stretch the histogram to cover the full range of pixel values.
    '''
    if a is None:
        a = np.min(input_image)
    if b is None:
        b = np.max(input_image)    

    if a == b: ## black and white
        stretched_image = np.where(input_image < a, 0, 255).astype(np.uint8)
    else:
        alpha = 255 / (b - a)
        beta = -alpha * a
        stretched_image = np.clip(alpha * input_image + beta, 0, 255).astype(np.uint8)

        if show_transform:
            # Plot the transformation function
            r = np.arange(0, 256)
            s = alpha * r + beta
            s = np.clip(s, 0, 255)

            plt.figure(figsize=(6, 4))
            plt.plot(r, s, color='blue')
            plt.title("Contrast Stretching Transformation Function")
            plt.xlabel("Input Pixel Intensity (r)")
            plt.ylabel("Output Pixel Intensity (s)")
            plt.xlim(-5, 260)
            plt.ylim(-5, 260)
            plt.axvline(x=a, color='red', linestyle='--', label=f'a = {a}')
            plt.axvline(x=b, color='green', linestyle='--', label=f'b = {b}')
            plt.text(a + 5, 240, f'a = {a}', color='red', fontsize=10)
            plt.text(b + 5, 10, f'b = {b}', color='green', fontsize=10)
            # plt.legend(loc='upper left')
            plt.grid()
            plt.show()
    return stretched_image

def thresholding(input_image: np.ndarray, threshold: float = None, show_transform: bool = False) -> np.ndarray:
    '''
    if r < threshold, f(r) = 0
    elif r >= threshold, f(r) = 255
    '''
    if threshold is None:
        threshold = np.mean(input_image)

    thresholded_image = contrast_stretching(input_image, a=threshold, b=threshold)

    if show_transform:
        # Plot the transformation function
        r = np.arange(0, 256)
        s = np.where(r < threshold, 0, 255)

        plt.figure(figsize=(6, 4))
        plt.plot(r, s, color='blue')
        plt.title("Thresholding Transformation Function")
        plt.xlabel("Input Pixel Intensity (r)")
        plt.ylabel("Output Pixel Intensity (s)")
        plt.xlim(-5, 260)
        plt.ylim(-5, 260)
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'threshold = {threshold}')
        plt.text(threshold + 5, 240, f'threshold = {threshold}', color='red', fontsize=10)
        # plt.legend(loc='upper left')
        plt.grid()
        plt.show()

    return thresholded_image

def contrast_stretching_piecewise(input_image: np.ndarray, r1: float = None, r2: float = None, s1: float = None, s2: float = None, show_transform: bool = False) -> np.ndarray:
    '''
    if r < r1, f(r) = m0 * r + c0
    elif r1 <= r <= r2, f(r) = m1 * r + c1
    elif r > r2, f(r) = m2 * r + c2
    '''
    if r1 is None:
        r1 = np.min(input_image)
    if r2 is None:
        r2 = np.max(input_image)
    if s1 is None:
        s1 = 0
    if s2 is None:
        s2 = 255

    # Avoid invalid parameters
    if r1 >= r2 or s1 > s2 or s1 < 0 or s2 > 255:
        return input_image.copy()

    m2 = (255 - s2) / (255 - r2)
    c2 = -m2 * 255 + 255

    m1 = (s2 - s1)/(r2 - r1)
    c1 = -m1 * r1 + s1

    m0 = s1 / r1
    c0 = 0

    if show_transform:
        # Plot the transformation function
        r = np.arange(0, 256)
        s = np.where(r < r1, m0 * r + c0, 
                     np.where(r > r2, m2 * r + c2, 
                              m1 * r + c1))

        plt.figure(figsize=(6, 4))
        plt.plot(r, s, color='blue')
        plt.title("Piecewise Linear Contrast Stretching Transformation Function")
        plt.xlabel("Input Pixel Intensity (r)")
        plt.ylabel("Output Pixel Intensity (s)")
        plt.xlim(-5, 260)
        plt.ylim(-5, 260)
        plt.axvline(x=r1, color='red', linestyle='--', label=f'r1 = {r1}')
        plt.axvline(x=r2, color='green', linestyle='--', label=f'r2 = {r2}')
        plt.text(r1 + 5, 240, f'r1 = {r1}', color='red', fontsize=10)
        plt.text(r2 + 5, 10, f'r2 = {r2}', color='green', fontsize=10)
        plt.axhline(y=s1, color='orange', linestyle='--', label=f's1 = {s1}')
        plt.axhline(y=s2, color='purple', linestyle='--', label=f's2 = {s2}')
        plt.text(10, s1 + 5, f's1 = {s1}', color='orange', fontsize=10)
        plt.text(10, s2 + 5, f's2 = {s2}', color='purple', fontsize=10)
        # plt.legend(loc='upper left')
        plt.grid()
        plt.show()

    stretched_image = np.where(input_image < r1, m0 * input_image + c0, # less than r1
                      np.where(input_image > r2, m2 * input_image + c2, # more than r2
                      m1 * input_image + c1))                           # in between
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)
    return stretched_image

def histogram_equalization(input_image: np.ndarray, show_transform: bool = False) -> np.ndarray:
    output_image = cv2.equalizeHist(input_image)

    if show_transform:
        hist = cv2.calcHist([input_image], [0], None, [256], [0, 256]).flatten()

        # Cumulative distribution function
        cdf = np.cumsum(hist)

        # Transformation function
        cdf_min = cdf[np.nonzero(hist)[0][0]]
        T = np.round((cdf - cdf_min) / (input_image.size - cdf_min) * 255)
        T = np.clip(T, 0, 255)

        # Plot transformation function
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(256), T)
        plt.xlabel("Input Pixel Intensity (r)")
        plt.ylabel("Output Pixel Intensity (s)")
        plt.title("Histogram Equalisation Transformation Function")
        plt.xlim(-5, 260)
        plt.ylim(-5, 260)
        plt.grid()
        plt.show()
    return output_image

def clahe_equalization(input_image: np.ndarray, clip_limit: float = 1.2, tile_grid_size: tuple = (4, 4)) -> np.ndarray:
    ''' CLAHE (Contrast Limited Adaptive Histogram Equalisation)
    Improves local contrast of an image by applying histogram equalization in small regions (tiles) of the image.
    The contrast of each tile is limited to avoid amplifying noise.
      '''
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    output_image = clahe.apply(input_image)
    return output_image

# Filtering Functions

def average_filter(input_image: np.ndarray, kernel_size: int = 5, use_cv2:bool = False) -> np.ndarray:
    '''Apply average filter on the input 2D greyscale image'''
    # cv2 implementation
    if use_cv2:
        return cv2.blur(input_image, (kernel_size, kernel_size), borderType=cv2.BORDER_CONSTANT)
    
    if input_image.ndim != 2:
        raise ValueError("input_image must be a 2D grayscale image")
    if kernel_size%2 == 0 or kernel_size <= 0:
        print("Error. kernel_size should be positive odd integers")
        raise ValueError ("kernel_size should be odd number for simplicity")
    input_size = input_image.shape

    # Calculate padding
    padding = kernel_size // 2
    padded_img = np.pad(input_image, padding, mode="edge")

    output_image = np.empty(input_image.shape, dtype=np.float64)

    # Iterate through pixels
    for i in range(input_size[0]):
        for j in range(input_size[1]):
            # sliding window
            window = padded_img[
                i : i + kernel_size,
                j : j + kernel_size
            ]
            output_image[i,j] = np.mean(window)
    output_image = np.clip(np.rint(output_image), 0, 255).astype(np.uint8)

    return output_image

def median_filter(input_image: np.ndarray, kernel_size: int = 5, use_cv2:bool=False) -> np.ndarray:
    '''Apply median filter on the input 2D greyscale image'''
    ## cv2 implementation
    if use_cv2:
        return cv2.medianBlur(input_image, kernel_size)

    if input_image.ndim != 2:
        raise ValueError("input_image must be a 2D grayscale image")
    if kernel_size%2 == 0 or kernel_size <= 0:
        print("Error. kernel_size should be positive odd integers")
        raise ValueError ("kernel_size should be odd number for simplicity")
    input_size = input_image.shape

    # Calculate padding
    padding = kernel_size // 2
    padded_img = np.pad(input_image, padding, mode="edge")

    output_image = np.empty_like(input_image)
    
    # Iterate through pixels
    for i in range(input_size[0]):
        for j in range(input_size[1]):
            # sliding window
            window = padded_img[
                i : i + kernel_size,
                j : j + kernel_size
            ]
            output_image[i,j] = np.median(window)

    return output_image

def median_filter_t(input_image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    '''Apply median filter on the input 2D greyscale image
    Kernel uses a t shape of length = kernel_size
    e.g.{
    [0,0,1,0,0],
    [0,0,1,0,0],
    [1,1,1,1,1],
    [0,0,1,0,0],
    [0,0,1,0,0],]'''

    if input_image.ndim != 2:
        raise ValueError("input_image must be a 2D grayscale image")
    if kernel_size%2 == 0 or kernel_size <= 0:
        print("Error. kernel_size should be positive odd integers")
        raise ValueError ("kernel_size should be odd number for simplicity")
    input_size = input_image.shape

    # Calculate padding
    padding = kernel_size // 2
    padded_img = np.pad(input_image, padding, mode="edge")

    output_image = np.empty_like(input_image)
    
    t_mask = np.zeros((kernel_size, kernel_size), dtype=bool)
    t_mask[padding, :] = True
    t_mask[:, padding] = True
    # Iterate through pixels
    for i in range(input_size[0]):
        for j in range(input_size[1]):
            # sliding window
            window = padded_img[
                i : i + kernel_size,
                j : j + kernel_size
            ]
            output_image[i,j] = np.median(window[t_mask])
    return output_image

def main():
    img = cv2.imread("Test_Image_2026.bmp", cv2.IMREAD_GRAYSCALE)
    avg_image = average_filter(img, kernel_size=15, use_cv2=True)
    median_image = median_filter(img, kernel_size=15, use_cv2=True)
    median_t_image = median_filter_t(img, kernel_size=19)
    avg_median_t_image = average_filter(img, kernel_size=5, use_cv2=True)
    
    plot_img_hist([img, median_image, median_t_image, avg_median_t_image], title="Median Filtered Image and Histogram", labels=["Original","Median","Median T","Median T Average"])
if __name__ == "__main__":
    main()