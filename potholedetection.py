import cv2
import numpy as np

FRAME_WIDTH = 600

img = cv2.imread('images/e6.png')
img = cv2.imread('images/filled2.jpg')
img = cv2.imread('images/f2.png')
img = cv2.imread('images/fake_road4.png')
img = cv2.imread('images/k2_cropped_small.png')
img = cv2.imread('images/im3_cropped_small.png')

# Resize Image
height, width, depth = img.shape
img = cv2.resize(img, (int(FRAME_WIDTH), int(FRAME_WIDTH * height / width)))

# Greyscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow('Greyscale', gray)

# Blurring
blur = cv2.GaussianBlur(gray, (3, 3), 0)
#cv2.imshow('Blur', blur)

# Thresholding
retval, threshold = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow('threshold',threshold)

# Adaptive Thresholding
adThresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 5)
#cv2.imshow('Adaptive3',adThresh)

# Erosion
kernel = np.ones((2,2), np.uint8)
erosion = cv2.erode(adThresh, kernel, iterations = 3)	# best
cv2.imshow('Eroded', erosion)

# Dilate the image
kernel = np.ones((2,2),np.uint8)
dilation = cv2.dilate(erosion, kernel, iterations=1)
#cv2.imshow('Dilate',dilation)

# Get Contours in Image
_, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find Contours within Area Limits
for cnt in contours:
	area = cv2.contourArea(cnt)
	#print(area)
	if area < 280 or area > 3000:
		continue
	
	# Draw ellipse around contours
	ellipse = cv2.fitEllipse(cnt)
	cv2.ellipse(img, ellipse, (0,255,0), 2)
	
cv2.imshow('original',img)
cv2.imshow('Ellipse', img)

# Draw Contours on Image
cv2.drawContours(img, contours, -1, (0,0,255),2)
#cv2.imshow('All Contours', img)

cv2.waitKey(0)
cv2.destroyAllWindows()