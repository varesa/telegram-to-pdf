import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def filter_image(image):
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    THRESHOLD_OFFSET = -20
    cutoff = np.percentile(grayscale.ravel(), 50) + THRESHOLD_OFFSET
    print("cutoff:", cutoff)

    ret, threshold = cv.threshold(grayscale, cutoff, 255, cv.THRESH_BINARY)

    return grayscale, threshold


"""
        edges,
        rho=2,
        theta=np.pi / 180,
        threshold=200,
        lines=np.array([]),
        minLineLength=100,
        maxLineGap=20,
"""


def get_lines(image):
    edges = cv.Canny(
        image=image,
        threshold1=75,
        threshold2=500,
    )
    # plt.imshow(edges, 'binary')
    # plt.show()
    # print(edges)

    lines = cv.HoughLinesP(
        edges,
        rho=2,
        theta=np.pi / 180,
        threshold=100,
        lines=np.array([]),
        minLineLength=50,
        maxLineGap=30,
    )

    return lines


ANGLE_THRESHOLD = 15


def get_line_angle_offset(image):
    lines = get_lines(image)
    angles = []

    im_lines = np.copy(image) * 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(im_lines, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) > 45:
                angle += -90 * np.sign(angle)
            if abs(angle) < ANGLE_THRESHOLD:
                angles.append(angle)

    #plt.imshow(im_lines)
    plt.show()

    return np.average(angles)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    return cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)


def get_bounding_rect(image):
    contours, hierarchy = cv.findContours(
        image=image,
        mode=cv.RETR_TREE,
        method=cv.CHAIN_APPROX_SIMPLE,
    )

    """max_area = None
    max_contour = None
    for contour in contours:
        area = cv.contourArea(contour)
        if not max_area or area > max_area:
            max_area = area
            max_contour = contour

    im_contours = image * 0
    cv.drawContours(
        image=im_contours,
        contours=contours,
        contourIdx=-1,
        color=(255, 0, 0),
        thickness=5,
    )
    
    print(max_contour)

    plt.imshow(im_contours, 'gray')
    plt.show()

    return cv.boundingRect(max_contour)"""

    CONTOUR_THRESHOLD = 200_000

    min_x = max_x = min_y = max_y = None
    for contour in contours:
        if cv.contourArea(contour) < CONTOUR_THRESHOLD:
            continue
        for point in contour:
            x, y = point[0]
            if not min_x or x < min_x:
                min_x = x
            if not max_x or x > max_x:
                max_x = x

            if not min_y or y < min_y:
                min_y = y
            if not max_y or y > max_y:
                max_y = y

    print(min_x, max_x - min_x, min_y, max_y - min_y)
    return min_x, min_y, max_x - min_x, max_y - min_y


def fix_image(input, output):
    im = cv.imread(input)
    im_gs, im_threshold = filter_image(im)

    correction = get_line_angle_offset(im_threshold)
    im_rotated = rotate_image(im, correction)

    im_rotated_gs, im_rotated_threshold = filter_image(im_rotated)

    x, y, w, h = get_bounding_rect(im_rotated_threshold)

    cv.rectangle(im_rotated, (x, y), (x + w, y + h), (255, 0, 0), 3)
    im_cropped = im_rotated[y:y + h, x:x + w]

    plt.imshow(im_rotated)
    plt.show()

    cv.imwrite(output, im_cropped)
