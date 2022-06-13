# Project
# Let us import all the necessary libraries
import cv2
import numpy as np
import cv2.aruco as aruco
import math

# This Function is used to rotate the image by the specified angle.


def rotation(img, angle, rotatepoint=None):
    (y, x) = img.shape[:2]
    if rotatepoint == None:
        rotatepoint = (x/2, y/2)

    rotMatrix = cv2.getRotationMatrix2D(rotatepoint, angle, 1.0)
    dimension = (x, y)
    return cv2.warpAffine(img, rotMatrix, dimension)

# Function used for pasting


def augumented(corners, id, img, imgAug, draw=True):
    tl = corners[0][0], corners[0][1]
    tr = corners[1][0], corners[1][1]
    br = corners[2][0], corners[2][1]
    bl = corners[3][0], corners[3][1]
    h, w, c = imgAug.shape
    pt1 = np.array([tl, tr, br, bl])
    pt2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pt2, pt1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, pt1.astype(int), (0, 0, 0))
    imgOut = img+imgOut
    return imgOut


# Importing all the required images
id_3 = cv2.imread("images/id_3.jpg")
id_4 = cv2.imread("images/id_4.jpg")
id_1 = cv2.imread("images/id_1.jpg")
id_2 = cv2.imread("images/id_2.jpg")

# Used to find the ID of the Aruco Markers


def findAruco(img, draw=True):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, 'DICT_5X5_250')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
        img, arucoDict, parameters=arucoParam)
    (x1, y1) = (int(corners[0][0][0][0]), int(corners[0][0][0][1]))
    (x2, y2) = (int(corners[0][0][1][0]), int(corners[0][0][1][1]))
    length = ((y2-y1)**2+(x2-x1)**2)**0.5  # Finding the slope.
    tantheta = float(y2-y1)/(x2-x1)
    theta = math.atan(tantheta) * (180/3.14)
    if draw:
        aruco.drawDetectedMarkers(img, corners)
    # print(ids)
    return ids[0][0], corners, theta, (int(x1), int(y1)), length


# findAruco(img=id_3) #3 #Black
# findAruco(img=id_4) #4 #Pink-Peach
# findAruco(img=id_1) #1 #Green
# findAruco(img=id_2) #2 #Orange

# Shape Detection
def countorfind(img, color):
    _, thrash = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        approx = cv2.approxPolyDP(
            contour, 0.01*cv2.arcLength(contour, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx) == 4:
            x1, y1, w, h = cv2.boundingRect(approx)
            asp_rat = float(w/h)
            if asp_rat >= 0.98 and asp_rat <= 1.02:  # Detecting The Squares
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                # print(len(approx))
                # print(box)
                box1 = np.int0(box)
                cv2.drawContours(img, [box1], 0, (255, 0, 0), 2)
                cv2.drawContours(resizedimg, [approx], 0, (0, 0, 255), 1)
                # cv2.putText(resizedimg,colour,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(25,25,0),1)
                return box, x1, y1


img = cv2.imread("images/CVtask.jpg")
resizedimg = cv2.resize(img, (877, 620))  # Resizing the images
imhsv = cv2.cvtColor(resizedimg, cv2.COLOR_BGR2HSV)

# finding pinkpeach colour
lower = np.array([4, 4, 216])
upper = np.array([29, 35, 251])
pinkpeach = cv2.inRange(imhsv, lower, upper)


# finding green colour
lower = np.array([26, 52, 196])
upper = np.array([60, 191, 217])
green = cv2.inRange(imhsv, lower, upper)

# finding orange colour
lower = np.array([10, 40, 40])
upper = np.array([30, 255, 255])
orange = cv2.inRange(imhsv, lower, upper)


# finding black colour
lower = np.array([0, 0, 0])
upper = np.array([10, 210, 210])
black = cv2.inRange(imhsv, lower, upper)
black = cv2.bilateralFilter(black, 10, 25, 25)
black = cv2.GaussianBlur(black, (3, 3), 0)


list = [id_3, id_4, id_1, id_2]

for i in list:
    id, corner, theta, point, length = findAruco(i)
    rotate = rotation(i, theta, point)
    cropped = rotate[point[1]:point[1] +
                     int(length), point[0]:point[0]+int(length)]
    # print(id)

    if id == 1:
        box, x, y = countorfind(green, str(id))
        resizedimg = augumented(box, id, resizedimg, cropped)
    elif id == 2:
        box, x, y = countorfind(orange, str(id))
        resizedimg = augumented(box, id, resizedimg, cropped)
    elif id == 3:
        box, x, y = countorfind(black, str(id))
        resizedimg = augumented(box, id, resizedimg, cropped)
    elif id == 4:
        box, x, y = countorfind(pinkpeach, str(id))
        resizedimg = augumented(box, id, resizedimg, cropped)

cv2.imwrite("final.jpg", resizedimg)
cv2.imshow("OPENCV", resizedimg)  # Final Image Output
cv2.waitKey(0)
cv2.destroyAllWindows()


# -----------------------------------------COMPLETED------------------------------------------------
