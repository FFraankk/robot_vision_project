import cv2
import numpy as np

def constrast_stretching(input_image: np.ndarray, a: float = None, b: float = None) -> np.ndarray:
    '''
    f(r) = alpha * r + beta
    '''
    if a is None:
        a = np.min(input_image)
    if b is None:
        b = np.max(input_image)    

    alpha = 255 / (b - a)
    beta = -alpha * a
    
    # Avoid division by zero
    if a - b == 0:
        return input_image.copy()
    
    stretched_image = alpha * input_image + beta
    return stretched_image.astype(np.uint8)