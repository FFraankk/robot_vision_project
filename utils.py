import cv2
import numpy as np

def plot_img_hist(image: np.ndarray, title: str = "Image and Histogram") -> None:
    import matplotlib.pyplot as plt
    
    _, axes = plt.subplots(1, 2, figsize=(12, 3))
    
    # Display the image
    axes[0].imshow(image, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title("Image")
    
    # Display the histogram
    axes[1].hist(image.ravel(), bins=256, range=(0, 256), color="black")
    axes[1].set_xlabel("Pixel intensity")
    axes[1].set_ylabel("Pixel count")
    axes[1].set_title("Intensity Histogram")
    
    plt.suptitle(title)
    plt.show()