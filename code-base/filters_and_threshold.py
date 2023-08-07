import numpy as np
from scipy import signal as sg
import cv2
from scipy.ndimage import maximum_filter
from run_attention import *


def beny_threshold(image):
    image = image * 255
    image = np.clip(image, 0, 255).astype(np.uint8)

    low_kernel = np.full((7, 7), 1 / 49).astype(np.float32)

    hing_kernel = np.full((3, 3), -1 / 9).astype(np.float32)
    hing_kernel[1, 1] = 8 / 9

    mask1 = cv2.inRange(image, (210, 100, 100), (255, 170, 160))
    mask2 = cv2.inRange(image, (156, 80, 60), (185, 115, 90))
    mask3 = cv2.inRange(image, (100, 50, 30), (150, 60, 35))
    mask4 = cv2.inRange(image, (85, 30, 30), (210, 50, 40))

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)
    mask = cv2.bitwise_or(mask, mask4)

    result = cv2.bitwise_and(image, image, mask=mask)

    red_image = result[:, :, 0]
    green_image = result[:, :, 1]
    blue_image = result[:, :, 2]

    maximum_filter(red_image, 5, output=red_image)

    red_image = sg.convolve2d(red_image, low_kernel, mode='same')
    green_image = sg.convolve2d(green_image, low_kernel, mode='same')
    blue_image = sg.convolve2d(blue_image, low_kernel, mode='same')

    final_image = np.dstack((red_image, green_image, blue_image)).clip(0, 255).astype(np.uint8)

    return final_image.astype(np.float32) / 255


"""
def red_thresholding(image):

    # red masks
    mask1 = cv2.inRange(image, (118, 44, 35), (180, 125, 85))
    mask2 = cv2.inRange(image, (85, 37, 30), (120, 85, 60))

    # green masks
    mask9 = cv2.inRange(image, (163, 250, 245), (189, 255, 255))

    mask = cv2.bitwise_or(mask1, mask2)
    # mask = cv2.bitwise_or(mask, mask3)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


def low_pass_filter(c_image: np.ndarray):

    kernel = np.full((3, 3), 1/9)

    red_image = c_image[:, :, 0]
    green_image = c_image[:, :, 1]
    blue_image = c_image[:, :, 2]

    red_image_float = red_image.astype(np.float32)
    green_image_float = green_image.astype(np.float32)
    blue_image_float = blue_image.astype(np.float32)

    convolved_red_image = sg.convolve(red_image_float, kernel)
    convolved_green_image = sg.convolve(green_image_float, kernel)
    convolved_blue_image = sg.convolve(blue_image_float, kernel)

    filtered_image = np.dstack((convolved_red_image, convolved_green_image, convolved_blue_image))
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image


def high_pass_filter(c_image: np.ndarray):
    kernel = np.full((3, 3), -1 / 9)
    kernel[1, 1] = 8/9

    red_image = c_image[:, :, 0]
    green_image = c_image[:, :, 1]
    blue_image = c_image[:, :, 2]

    red_image_float = red_image.astype(np.float32)
    green_image_float = green_image.astype(np.float32)
    blue_image_float = blue_image.astype(np.float32)

    convolved_red_image = sg.convolve(red_image_float, kernel)
    convolved_green_image = sg.convolve(green_image_float, kernel)
    convolved_blue_image = sg.convolve(blue_image_float, kernel)

    convolved_red_image = convolved_red_image[:c_image.shape[0], :c_image.shape[1]]
    convolved_green_image = convolved_green_image[:c_image.shape[0], :c_image.shape[1]]
    convolved_blue_image = convolved_blue_image[:c_image.shape[0], :c_image.shape[1]]

    filtered_image = np.dstack((convolved_red_image, convolved_green_image, convolved_blue_image))
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image


def kernel_creator():
    kernel_size = 32
    kernel_image = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    center_x, center_y = kernel_size // 2, kernel_size // 2
    radius = min(center_x, center_y) - 5

    cv2.circle(kernel_image, (center_x, center_y), radius, 255, -1)

    return kernel_image
"""