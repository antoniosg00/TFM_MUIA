import cv2
import numpy as np
import pygame

def compute_borders(circuit_path, finish_path):
    """
    Computes and modifies the circuit and finish line images for edge detection.

    Parameters:
    - circuit_path: Path to the circuit image file.
    - finish_path: Path to the finish line image file.

    Returns:
    - circuit: Processed circuit image (grayscale).
    - finish: Processed finish line image (grayscale).
    - position: Position of the modified box (x, y).
    """
    # Load the circuit and finish image
    circuit = cv2.imread(circuit_path, cv2.IMREAD_GRAYSCALE)
    finish = cv2.imread(finish_path, cv2.IMREAD_GRAYSCALE)
    finish, position = modify_box(cv2.bitwise_not(finish))
    # # Detect edges using Canny
    # c_edges = cv2.Canny(circuit, threshold1=100, threshold2=200)
    # f_edges = cv2.Canny(finish, threshold1=100, threshold2=200)
    # # Make edges thicker using dilation (to avoid radar leaks)
    # kernel = np.ones((3,3), np.uint8)  # Adjust the kernel size for thicker/thinner edges
    # c_edges_dilated = cv2.dilate(c_edges, kernel, iterations=1)  # numpy.ndarray with one channel -> shape = (height, width)
    # f_edges_dilated = cv2.dilate(f_edges, kernel, iterations=1)  # numpy.ndarray with one channel -> shape = (height, width)
    # return c_edges_dilated, f_edges_dilated

    return circuit, finish, position[::-1]  # Original position = (y, x)=(row, col)  ->  (x, y)


def modify_box(matrix):
    """
    Modifies a matrix to enhance detection of a rectangular box.

    Parameters:
    - matrix: Binary image matrix.

    Returns:
    - matrix: Modified matrix with enhanced rectangular box.
    - top_left: Coordinates of the top-left corner of the box (y, x).
    """
    # Find the dimensions of the matrix
    rows = len(matrix)
    cols = len(matrix[0])

    # Find the coordinates of the rectangular box
    top_left = None
    bottom_right = None
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 255:
                if top_left is None:
                    top_left = (i, j)
                bottom_right = (i, j)

    # Calculate the midpoint of the box
    mid_row = (top_left[0] + bottom_right[0]) // 2

    # Modify the top half of the box
    for i in range(top_left[0], mid_row + 1):
        for j in range(top_left[1], bottom_right[1] + 1):
            matrix[i][j] = 100

    return matrix, top_left


def scale_image(img, factor):
    """
    Scales a Pygame image by a given factor.

    Parameters:
    - img: Pygame image object to scale.
    - factor: Scaling factor.

    Returns:
    - scaled_img: Scaled Pygame image.
    """
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)


def blit_rotate_center(win, image, top_left, angle):
    """
    Blits a rotated image onto a Pygame window.

    Parameters:
    - win: Pygame window surface to draw on.
    - image: Pygame image object to rotate and blit.
    - top_left: Top-left coordinates (x, y) of the image.
    - angle: Rotation angle in degrees.
    """
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)
    win.blit(rotated_image, new_rect.topleft)
