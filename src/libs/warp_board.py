import cv2
import numpy as np

def warp_board(image, corners, output_size=800):
    # Destination points for a top-down 800x800 view
    dst_pts = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype=np.float32)

    src_pts = np.array(corners, dtype=np.float32)

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp the image
    warped = cv2.warpPerspective(image, matrix, (output_size, output_size))
    return warped
