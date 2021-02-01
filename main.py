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
    #plt.imshow(edges, 'binary')
    #plt.show()
    #print(edges)

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
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            if abs(angle) > 45:
                angle += -90 * np.sign(angle)
            if abs(angle) < ANGLE_THRESHOLD:
                angles.append(angle)

    plt.imshow(im_lines)
    plt.show()

    return np.average(angles)


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    return cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)


def main():
    im = cv.imread('/home/esav.fi/esa//Pictures/opencv_test1.jpg')
    im_gs, im_threshold = filter_image(im)

    """
    hist = cv.calcHist(
        images=im_gs,
        channels=[0],
        mask=None,
        histSize=[256],
        ranges=[0, 256],
    )
    print(hist)
    """

    correction = get_line_angle_offset(im_threshold)
    im_rotated = rotate_image(im, correction)

    plt.imshow(im_rotated)
    plt.show()

    plt.imshow(im)
    plt.show()


    im_rotated_gs, im_rotated_threshold = filter_image(im_rotated)

    contours, hierarchy = cv.findContours(
        image=im_rotated_threshold,
        mode=cv.RETR_TREE,
        method=cv.CHAIN_APPROX_SIMPLE,
    )

    max_area = None
    max_contour = None
    for contour in contours:
        area = cv.contourArea(contour)
        if not max_area or area > max_area:
            max_area = area
            max_contour = contour

    """
    im_contours = im_rotated * 0
    cv.drawContours(
        image=im_contours,
        contours=max_contour,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=3,
    )
    
    plt.imshow(im_contours)
    plt.show()
    """

    x, y, w, h = cv.boundingRect(max_contour)

    cv.rectangle(im_rotated, (x, y), (x + w, y + h), (255, 0, 0), 3)
    im_cropped = im_rotated[y:y+h, x:x+w]
    plt.imshow(im_cropped)
    plt.show()


if __name__ == '__main__':
    main()
