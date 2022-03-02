# Importing the necessary packages
from imutils.contours import sort_contours
import numpy as np
import pytesseract
import argparse
import imutils
import sys
import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
help= "path to the input image to be OCR'd")
args = vars(ap.parse_args())

# Load the input image, convert it to grayscale, and grab its dimensions
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(H, W) = gray.shape

# Initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,7))
agKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,21))

# Smooth the image using a 3 * 3 Gaussian blur and then apply a blackhat
# morphological operator to find dark regions on a light background
gray = cv2.GaussianBlur(gray, (3,3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
cv2.imshow("Blackhat", blackhat)

# Compute the Scharr gradient of the blackhat image and scale the 
# result into the range [0, 255]
grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
grad = np.abs(grad)
(minVal, maxVal) = (np.min(grad), np.max(grad))
grad = (grad - minVal) / (maxVal - minVal)
grad = (grad * 255).astype("uint8")
cv2.imshow("Gradient", grad)

# Applying a closing operation using the rectangular kernel to close gaps
# in between the letters --then apply Otsu's thresholding method
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(grad, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Rect Close", thresh)

# Perform another closing operation, this time using the square kernel to close gaps
# between lines of the MRZ, then perform a series of erosions 
# to break apart connected components
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, agKernel)
thresh = cv2.erode(thresh, None, iterations=2)
cv2.imshow("Square Close", thresh)

# Find contours in the thresholded image and sort them from the bottom
# to top (since the MRZ will always be at the bottom of the passport)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method = "bottom-to-top")[0]

# Initialize the bounding box associated with the MRZ
mrzBox = None

# Loop over the contours
for c in cnts:
    # Compute the bounding box of the contour and then derive 
    # how much of the image the bounding box occupies in terms
    # of both width and height
    (x, y, w, h) = cv2.boundingRect(c)
    percentWidth = w / float(W)
    percentHeight = h / float(H)

    # If the bounding box occupies > 80% width and > 4% height of the image,
    # then we assume we have found the MRZ
    if percentWidth > 0.8 and percentHeight > 0.04:
        mrzBox = (x, y, w, h)
        break

# If the MRZ was not found, exit the script
if mrzBox is None:
    print("[INFO] MRZ could not be found")
    sys.exit(0)

# Pad the bounding box since we applied erosions and now we need to re-grow it
(x, y, w, h) = mrzBox
pX = int((x + w) * 0.03)
pY = int((y + h) * 0.03)
(x,y) = (x - pX, y - pY)
(w, h) = (w + (pX * 2), h + (pY * 2))

# Extract the padded MRZ from the image
mrz = image[y:y + h, x:x + w]

# OCR the MRZ region of interest using Tesseract, removing any occurrences of spaces
mrzText = pytesseract.image_to_string(mrz)
mrzText = mrzText.replace(" ", "")
print(mrzText)

# Show the MRZ image
cv2.imshow("MRZ", mrz)
cv2.waitKey(0)