# # -*- coding: utf-8 -*-
import cv2

__author__ = "Héctor Del Campo Pando"
__version__ = "1.0"

# Pipeline step 1: Read image
img = cv2.imread("test3.jpg", cv2.IMREAD_UNCHANGED)

# Pipeline extra step: Visualise img status
cv2.namedWindow("window", cv2.WINDOW_NORMAL) # Enable manual resizing of window
cv2.imshow("window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()