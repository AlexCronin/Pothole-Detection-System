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
	
def main():

	global img

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
	
# Grey scaling & Thresholding
def test1():
	global img
	# Greyscale
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('Greyscale', grey)
	
	# Threshold
	retval, threshold = cv2.threshold(grey, 80, 255, cv2.THRESH_BINARY_INV)
	cv2.imshow('threshold',threshold)
	
	_, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Contours
	for cnt in contours:
		area = cv2.contourArea(cnt)
		#print(area)
		if area < 80 or area > 3000:
			continue

		# Draw ellipse around contours
		ellipse = cv2.fitEllipse(cnt)
		cv2.ellipse(img, ellipse, (0,255,0), 2)
		
	cv2.imshow('Ellipse', img)

	cv2.drawContours(img, contours, -1, (0,0,255),2)
	cv2.imshow('All Contours', img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
# Grey Scaling & Adaptive Thresholding
def test2():
	global img
	
	# Greyscale
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('Greyscale', grey)
	
	#Adaptive Thresholding
	adThresh3 = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 5)
	
	_, contours, hierarchy = cv2.findContours(adThresh3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Contours
	for cnt in contours:
		area = cv2.contourArea(cnt)
		#print(area)
		if area < 80 or area > 3000:
			continue
		
		# Draw ellipse around contours
		ellipse = cv2.fitEllipse(cnt)
		cv2.ellipse(img, ellipse, (0,255,0), 2)
		
	cv2.imshow('Ellipse', img)

	cv2.drawContours(img, contours, -1, (0,0,255),2)
	cv2.imshow('All Contours', img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
# Grey Scaling & Erosion
def test3():
	global img
	# Greyscale
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('Greyscale', grey)
	
	# Erosion
	kernel = np.ones((2,2), np.uint8)
	erosion2 = cv2.erode(grey, kernel, iterations = 3)	# best
	cv2.imshow('Eroded2', erosion2)
	
	_, contours, hierarchy = cv2.findContours(erosion2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Contours
	for cnt in contours:
		area = cv2.contourArea(cnt)
		#print(area)
		if area < 80 or area > 3000:
			continue

		
		# Draw ellipse around contours
		ellipse = cv2.fitEllipse(cnt)
		cv2.ellipse(img, ellipse, (0,255,0), 2)
		
	cv2.imshow('Ellipse', img)

	cv2.drawContours(img, contours, -1, (0,0,255),2)
	cv2.imshow('All Contours', img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
# Grey Scaling & Dilation
def test4():
	global img
	# Greyscale
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('Greyscale', grey)
	
	# Dilate the image
	kernel = np.ones((2,2),np.uint8)
	dilation = cv2.dilate(grey, kernel, iterations=1)
	cv2.imshow('Dilate',dilation)
	
	_, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Contours
	for cnt in contours:
		area = cv2.contourArea(cnt)
		#print(area)
		if area < 80 or area > 3000:
			continue

		
		# Draw ellipse around contours
		ellipse = cv2.fitEllipse(cnt)
		cv2.ellipse(img, ellipse, (0,255,0), 2)
		
	cv2.imshow('Ellipse', img)

	cv2.drawContours(img, contours, -1, (0,0,255),2)
	cv2.imshow('All Contours', img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
# Grey Scaling and Thresholding and Erosion
def test5():
	global img
	
	# Greyscale
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('Greyscale', grey)
	
	# Threshold
	retval, threshold = cv2.threshold(grey, 80, 255, cv2.THRESH_BINARY_INV)
	cv2.imshow('Threshold', threshold)

	# Erosion
	kernel = np.ones((2,2), np.uint8)
	erosion2 = cv2.erode(threshold, kernel, iterations = 3)	# best
	cv2.imshow('Eroded2', erosion2)
	
	_, contours, hierarchy = cv2.findContours(erosion2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Contours
	for cnt in contours:
		area = cv2.contourArea(cnt)
		#print(area)
		if area < 80 or area > 3000:
			continue

		
		# Draw ellipse around contours
		ellipse = cv2.fitEllipse(cnt)
		cv2.ellipse(img, ellipse, (0,255,0), 2)
		
	cv2.imshow('Ellipse', img)

	cv2.drawContours(img, contours, -1, (0,0,255),2)
	cv2.imshow('All Contours', img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
# Grey Scaling and Thresholding and Dilation
def test6():
	global img
	
	# Greyscale
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('Greyscale', grey)
	
	# Threshold
	retval, threshold = cv2.threshold(grey, 80, 255, cv2.THRESH_BINARY_INV)
	cv2.imshow('Threshold', threshold)

	# Dilate the image
	kernel = np.ones((2,2),np.uint8)
	dilation = cv2.dilate(threshold, kernel, iterations=1)
	cv2.imshow('Dilate',dilation)
	
	_, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Contours
	for cnt in contours:
		area = cv2.contourArea(cnt)
		#print(area)
		if area < 80 or area > 3000:
			continue

		
		# Draw ellipse around contours
		ellipse = cv2.fitEllipse(cnt)
		cv2.ellipse(img, ellipse, (0,255,0), 2)
		
	cv2.imshow('Ellipse', img)

	cv2.drawContours(img, contours, -1, (0,0,255),2)
	cv2.imshow('All Contours', img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	
# Grey Scaling and Adaptive Thresholding and Erosion
def test7():
	global img
	
	# Greyscale
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('Greyscale', grey)
	
	# Adaptive Thresholding
	adThresh3 = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 5)	#best

	# Erosion
	kernel = np.ones((2,2), np.uint8)
	erosion2 = cv2.erode(adThresh3, kernel, iterations = 3)	# best
	cv2.imshow('Eroded2', erosion2)
	
	_, contours, hierarchy = cv2.findContours(erosion2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Contours
	for cnt in contours:
		area = cv2.contourArea(cnt)
		#print(area)
		if area < 80 or area > 3000:
			continue

		
		# Draw ellipse around contours
		ellipse = cv2.fitEllipse(cnt)
		cv2.ellipse(img, ellipse, (0,255,0), 2)
		
	cv2.imshow('Ellipse', img)

	cv2.drawContours(img, contours, -1, (0,0,255),2)
	cv2.imshow('All Contours', img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
# Grey Scaling and Adaptive Thresholding and Dilation
def test8():
	global img
	
	# Greyscale
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('Greyscale', grey)
	
	# Adaptive Thresholding
	adThresh3 = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 5)	#best
	cv2.imshow('AdaptiveThreshold', adThresh3)

	# Dilate the image
	kernel = np.ones((2,2),np.uint8)
	dilation = cv2.dilate(adThresh3, kernel, iterations=1)
	cv2.imshow('Dilate',dilation)
	
	_, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Contours
	for cnt in contours:
		area = cv2.contourArea(cnt)
		#print(area)
		if area < 80 or area > 3000:
			continue

		
		# Draw ellipse around contours
		ellipse = cv2.fitEllipse(cnt)
		cv2.ellipse(img, ellipse, (0,255,0), 2)
		
	cv2.imshow('Ellipse', img)

	cv2.drawContours(img, contours, -1, (0,0,255),2)
	cv2.imshow('All Contours', img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
# Grey Scaling Erosion and Adaptive Thresholding
def test9():
	global img
	
	# Greyscale
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('Greyscale', grey)

	# Erosion
	kernel = np.ones((2,2), np.uint8)
	erosion2 = cv2.erode(grey, kernel, iterations = 3)	# best
	cv2.imshow('Eroded2', erosion2)
	
	# Adaptive Thresholding
	adThresh3 = cv2.adaptiveThreshold(erosion2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 5)	#best
	cv2.imshow('AdaptiveThreshold', adThresh3)
	
	_, contours, hierarchy = cv2.findContours(adThresh3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Contours
	for cnt in contours:
		area = cv2.contourArea(cnt)
		#print(area)
		if area < 80 or area > 3000:
			continue

		
		# Draw ellipse around contours
		ellipse = cv2.fitEllipse(cnt)
		cv2.ellipse(img, ellipse, (0,255,0), 2)
		
	cv2.imshow('Ellipse', img)

	cv2.drawContours(img, contours, -1, (0,0,255),2)
	cv2.imshow('All Contours', img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
# Grey Scaling Dilation and Adaptive Thresholding
def test10():
	global img
	
	# Greyscale
	grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('Greyscale', grey)

	# Dilate the image
	kernel = np.ones((2,2),np.uint8)
	dilation = cv2.dilate(grey, kernel, iterations=1)
	cv2.imshow('Dilate',dilation)
	
	# Adaptive Thresholding
	adThresh3 = cv2.adaptiveThreshold(dilation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 115, 5)	#best
	cv2.imshow('AdaptiveThreshold', adThresh3)
	
	_, contours, hierarchy = cv2.findContours(adThresh3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Contours
	for cnt in contours:
		area = cv2.contourArea(cnt)
		#print(area)
		if area < 80 or area > 3000:
			continue


		# Draw ellipse around contours
		ellipse = cv2.fitEllipse(cnt)
		cv2.ellipse(img, ellipse, (0,255,0), 2)
		
	cv2.imshow('Ellipse', img)

	cv2.drawContours(img, contours, -1, (0,0,255),2)
	cv2.imshow('All Contours', img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
if __name__ == '__main__':

	func_dict = {'test1':test1, 'test2':test2, 'test3':test3, 'test4':test4, 'test5':test5, 'test6':test6, 'test7':test7,'test8':test8, 'test9':test9,'test10':test10}
	print("Enter the test number you wish to run (1-10), e.g test6")
	func = input('>')
	func_dict[func]()
	
	#main()