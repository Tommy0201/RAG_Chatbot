import cv2
from scipy.spatial import distance as dist
import numpy as np
import math
import matplotlib.pyplot as plt
import itertools
from pylsd.lsd import lsd



 #credit: https://github.com/andrewdcampbell/OpenCV-Document-Scanner/tree/master
 
def resize(image, width=None, height=None):
    dimension = None
    (h, w) = image.shape[:2]
    inter = cv2.INTER_AREA
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height / float(h)
        dimension = (int(w * r), int(height))
    elif height is None:
        r = width / float(w)
        dimension = (int(width), int(h * r))
    
    return cv2.resize(image, dimension, interpolation=inter)

def filter_corners(corners, min_dist=20):
    def predicate(representatives, corner):
        return all(dist.euclidean(representative, corner) >= min_dist
                    for representative in representatives)

    filtered_corners = []
    for c in corners:
        if predicate(filtered_corners, c):
            filtered_corners.append(c)
    return filtered_corners

def find_corners(img): 
    # the lsd function detect all lines of image
    lines = lsd(img) 
        # massages the output from LSD
        # LSD operates on edges. One "line" has 2 edges, and so we need to combine the edges back into lines
        # 1. separate out the lines into horizontal and vertical lines.
        # 2. Draw the horizontal lines back onto a canvas, but slightly thicker and longer.
        # 3. Run connected-components on the new canvas
        # 4. Get the bounding box for each component, and the bounding box is final line.
        # 5. The ends of each line is a corner
        # 6. Repeat for vertical lines
        # 7. Draw all the final lines onto another canvas. Where the lines overlap are also corners

    corners = []
    if lines is not None:
        # separate out the horizontal and vertical lines, and draw them back onto separate canvases
        lines = lines.squeeze().astype(np.int32).tolist()
        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2, _ = line
            if abs(x2 - x1) > abs(y2 - y1):
                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
            else:
                (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

        lines = []

        # find the horizontal lines (connected-components -> bounding boxes -> final lines)
        (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
        horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            min_x = np.amin(contour[:, 0], axis=0) + 2
            max_x = np.amax(contour[:, 0], axis=0) - 2
            left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
            right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
            lines.append((min_x, left_y, max_x, right_y))
            cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
            corners.append((min_x, left_y))
            corners.append((max_x, right_y))

        # find the vertical lines (connected-components -> bounding boxes -> final lines)
        (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
        vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
        for contour in contours:
            contour = contour.reshape((contour.shape[0], contour.shape[2]))
            min_y = np.amin(contour[:, 1], axis=0) + 2
            max_y = np.amax(contour[:, 1], axis=0) - 2
            top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
            bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
            lines.append((top_x, min_y, bottom_x, max_y))
            cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
            corners.append((top_x, min_y))
            corners.append((bottom_x, max_y))

        # find the corners
        corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
        corners += zip(corners_x, corners_y)

    # remove corners in close proximity
    corners = filter_corners(corners)
    return corners

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
 
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
 
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
 
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype = "float32")

def angle_range(quad):
    """
    Returns the range between max and min interior angles of quadrilateral.
    The input quadrilateral must be a numpy array with vertices ordered clockwise
    starting with the top left vertex.
    """
    tl, tr, br, bl = quad
    ura = get_angle(tl[0], tr[0], br[0])
    ula = get_angle(bl[0], tl[0], tr[0])
    lra = get_angle(tr[0], br[0], bl[0])
    lla = get_angle(br[0], bl[0], tl[0])

    angles = [ura, ula, lra, lla]
    return np.ptp(angles)   

def get_angle(p1, p2, p3):
    """
    Returns the angle between the line segment from p2 to p1  and the line segment from p2 to p3 in degrees
    """
    a = np.radians(np.array(p1))
    b = np.radians(np.array(p2))
    c = np.radians(np.array(p3))

    v1 = a-b
    v2 = c - b
    cos_theta = np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_radians = math.acos(cos_theta)
    return np.degrees(angle_radians)

def add_contour_corners(potential_corners,img_height,img_width,MIN_QUAD_RATIO,MAX_QUAD_ANGLE_RANGE, possible_contour):
    # First way to find contours: find all potential corners, make them quadrilaterals, find quadrilaterals with largest contourArea,
    if len(potential_corners) >= 4:
        quads = []
        for quad in itertools.combinations(potential_corners,4):
            points = np.array(quad)
            points = order_points(points)
            points = np.array([[p] for p in points], dtype = "int32")
            quads.append(points)
        # Sort by quads with largest contourArea
        quads = sorted(quads, key=cv2.contourArea,reverse=True)[:5]
        quads = sorted(quads, key=angle_range)
        cnt = quads[0]
        if (len(cnt)==4 and cv2.contourArea(cnt)> img_height*img_width*MIN_QUAD_RATIO) and angle_range(cnt) < MAX_QUAD_ANGLE_RANGE:
            possible_contour.append(cnt)
            # cv2.drawContours(image, [cnt], -1, (20, 20, 255), 2)
            # plt.scatter(*zip(*potential_corners))
            # plt.imshow(image)
            # plt.show()
            
def add_contour_library(edged,img_height,img_width,MIN_QUAD_RATIO,MAX_QUAD_ANGLE_RANGE, possible_contour):
    # Also find contour using cv2 library
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:     # loop over the contours
        # approximate the contour
        cnt = cv2.approxPolyDP(c, 80, True)
        if (len(cnt)==4 and cv2.contourArea(cnt)> img_height*img_width*MIN_QUAD_RATIO) and angle_range(cnt) < MAX_QUAD_ANGLE_RANGE:
            possible_contour.append(cnt)
            break

def find_contour(image):
    MORPH, CANNY, HOUGH = 9, 84, 25
    MIN_QUAD_RATIO, MAX_QUAD_ANGLE_RANGE = 0.25, 40
    
    img_height, img_width, _ = image.shape
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    
    # dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(MORPH,MORPH))
    dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # find edges
    edged = cv2.Canny(dilated, 0, CANNY)
    
    #find corners
    potential_corners = find_corners(edged)
    
    possible_contour = []
    add_contour_corners(potential_corners,img_height,img_width,MIN_QUAD_RATIO, MAX_QUAD_ANGLE_RANGE, possible_contour)
    add_contour_library(edged,img_height,img_width,MIN_QUAD_RATIO, MAX_QUAD_ANGLE_RANGE, possible_contour)
    
    if not possible_contour:
        top_right = (img_width,0)
        bot_right = (img_width,img_height)
        bot_left = (0,img_height)
        top_left = (0,0)
        finalCnt = np.array([[top_right],[bot_right],[bot_left],[top_left]])
    else:
        finalCnt = max(possible_contour, key=cv2.contourArea)
        
    return finalCnt.reshape(4,2)


def warped_transform(image,pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_bot = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(width_bot), int(width_top))


    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(height_right), int(height_left))


    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def main_preprocessing(image):  #output is an image fully rescaled and transformed
    
    RESCALED_HEIGHT = 500.0
    ratio = image.shape[0] / RESCALED_HEIGHT
    original = image.copy()
    rescaled_image = resize(image, height=RESCALED_HEIGHT)
    contours_image = find_contour(rescaled_image)

    warped = warped_transform(original, contours_image * ratio)
    
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sharpen = cv2.GaussianBlur(gray, (0,0), 3)
    sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
    
    thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
    
    return thresh


if __name__=="__main__":
    
    name = "ocr2.jpg"
    image = cv2.imread("input/"+name)
    RESCALED_HEIGHT = 500.0
    ratio = image.shape[0] / RESCALED_HEIGHT
    original = image.copy()
    rescaled_image = resize(image, height=RESCALED_HEIGHT)
    contours_image = find_contour(rescaled_image)
    
    warped = warped_transform(original, contours_image * ratio)
    
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sharpen = cv2.GaussianBlur(gray, (0,0), 3)
    sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
    
    thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
    cv2.imwrite("output/thresh_"+name,thresh)

    
    # cv2.imwrite("output/ocr1_rescaled.jpg",scanned_image)