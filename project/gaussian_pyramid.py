import cv2
import numpy as np
#import imutils

A = cv2.imread('toy4.jpeg')
#B = cv2.imread('toy5.jpeg')
height, width, channels = A.shape

def pyramid(image, scale=1.6, minSize=(30, 30)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = cv2.imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image

pyramid1 = np.array(pyramid(A))
print(len(pyramid1))