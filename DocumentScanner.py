# # -*- coding: utf-8 -*-
import cv2
import numpy as np

__author__ = "Héctor Del Campo Pando"
__version__ = "1.0"
__img_path__ = "test1.jpg"

__frame_name__ = "Document"

# Pipeline step 1: Read image
img = cv2.imread(__img_path__, cv2.IMREAD_UNCHANGED)

# Pipeline step 2: Preprocess
grey = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)  # Grey channel is enough for our goal
grey = cv2.bilateralFilter(grey, 7, 40, 40)  # Blur image to remove text and noise

# Pipeline step 3: Edge detection
edges = cv2.Canny(grey, 120, 180)

kernel = np.ones((16, 16), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)  # Dilate detected edges to avoid holes

# Pipeline step 4: Contour detection
_, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Pipeline step 5: Filter right contour (Biggest one & Squared)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

sheetContour = None
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * peri, True)

    if len(approx) == 4:
        sheetContour = approx
        break

# Pipeline step 6: Perspective transformation

# Sort the corners (From upper to lower, then left to right by pairs)
sheetContour = [x[0] for x in sheetContour]
sheetContour = sorted(sheetContour, key=lambda p: p[1])
sheetContour[:2] = sorted(sheetContour[:2], key=lambda p: p[0])
sheetContour[2:] = sorted(sheetContour[2:], key=lambda p: p[0])

# We've to keep document aspect ratio, use euclidean distance as reference
width = np.linalg.norm(sheetContour[0]-sheetContour[1])
height = np.linalg.norm(sheetContour[0]-sheetContour[2])

pts_src = np.float32(sheetContour)
pts_dst = np.float32([(0, 0), (width, 0), (0, height), (width, height)])

matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
img = cv2.warpPerspective(img, matrix, (int(width), int(height)))

# Pipeline step 7: Thresholding
img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

# Pipeline extra step: Visualise img status
cv2.namedWindow(__frame_name__, cv2.WINDOW_NORMAL)  # Enable manual resizing of window
cv2.imshow(__frame_name__, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
