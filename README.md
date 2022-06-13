# Keshav-Gupta-21JE0467-OpenCV

In this task we have to put Aruco Markers on the squares of different colours. Each aruco marker of different id is used on different colour according to the table given in the problem statement. Firstly the Contour of the shapes are determined and then the shape got determined by the help of the various cv2 functions. After detemining the Number of squares we have found the colour of different squares by using the hsv trackbars values. After that we have found out the respective id of the aruco markers with the help of the function findAruco which was made with the help of several cv2 functions.
Then a for loop is used to rotate and crop each marker according to its coordinate.
Then the image of Aruco markers got rotated by the rotation function and then just pasted the image on their respective squares of different colours.
The colours and the aruco ids are mentioned here :)
Green----->1
Orange----->2
Black------>3
Pink-Peach----->4

Functions used in the code are-:
rotation --> Used to Rotate the image according to the specified angles.
augumented --> Used for Image Augumentation.
findAruco --> Used to find the ids of different Aruco Markers.
countorfind --> Used for the shape detection.
Functions which are in-built in opencv and numpy-:
cv2.getRotationMatrix2D
cv2.warpAffine
cv2.findHomography
cv2.warpPerspective
cv2.fillConvexPoly
cv2.imread
cv2.cvtColor
cv2.aruco.detectMarkers
cv2.threshold
cv2.findContours
cv2.approxPolyDP
cv2.RETR_TREE
cv2.CHAIN_APPROX_NONE
cv2.boundingRect
cv2.minAreaRect
cv2.boxPoints
cv2.drawContours
cv2.putText
cv2.resize
cv2.cvtColor
cv2.inRange
cv2.bilateralFilter
cv2.GaussianBlur
cv2.imshow
cv2.waitKey
cv2.destroyAllWindows
Numpy functions are-:
np.array
np.float32
Math functions-:
math.atan



Hence by using all the above functions the project is completed:)
