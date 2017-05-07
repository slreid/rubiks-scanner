import skimage.transform
import skimage.filters
import numpy as np
import argparse
import cv2
import imutils
import tensorflow
import keras
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras import backend as K
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left

	rect = np.zeros((4, 2), dtype="float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


# In[13]:

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype="float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
	return warped


def get_scorecard():
	image = cv2.imread("D:\\Google Drive\\UVa\\Classes\\Semester 6\\CS 4501\\Project 4\\test_images\\close_center.jpg")
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height=500)

	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)
	# gray = cv2.equalizeHist(gray)
	edged = cv2.Canny(gray, 30, 180)

	# show the original image and the edge detected image
	print("STEP 1: Edge Detection")
	# cv2.imshow("Image", image)
	# cv2.imshow("Edged", edged)

	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break

	# show the contour (outline) of the piece of paper
	print("STEP 2: Find contours of paper")
	# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
	# cv2.imshow("Outline", image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# apply the four point transform to obtain a top-down
	# view of the original image
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

	# convert the warped image to grayscale, then threshold it
	# to give it that 'black and white' paper effect
	out_bgr = skimage.transform.rotate(skimage.transform.resize(warped, (650, 650), mode='constant'), 90)
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	# warped = threshold_adaptive(warped, 251, offset = 10)
	# warped = warped.astype("uint8") * 255

	# show the original and scanned images
	print("STEP 3: Apply perspective transform")
	# cv2.imshow("Original", imutils.resize(orig, height = 650))
	out_gray = skimage.transform.rotate(skimage.transform.resize(warped, (650, 650), mode='constant'), 90)
	# cv2.imshow("Scanned", out)
	# cv2.imwrite("scanned.png", out)
	# cv2.waitKey(0)
	return out_bgr, out_gray


def predict_digit(image):
	# input image dimensions
	img_rows, img_cols = 28, 28
	image = skimage.transform.resize(image, (img_cols, img_rows), mode='constant')
	if K.image_data_format() == 'channels_first':
		image = image.reshape(1, img_rows, img_cols)
	else:
		image = image.reshape(img_rows, img_cols, 1)

	model = load_model('CNN\\mnist_cnn_32.h5')

	return model.predict(np.asarray([image]))[0]


def test_contours(image):
	image = (image * 255).astype("uint8")
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	cv2.imshow("blurred", blurred)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# gray = cv2.equalizeHist(gray)
	edged = cv2.Canny(blurred, 1, 80)
	cv2.imshow("Edges", edged)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# cnts = cnts[:7] #sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
	polys = []
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		polys.append(approx)
		rect = cv2.minAreaRect(c)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
		cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
		cv2.imshow("Outline", image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	cv2.drawContours(image, polys, -1, (128, 255, 0), 2)
	cv2.imshow("Outline", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def test_contours_2(image):
	bw = image < skimage.filters.threshold_local(image, 267)
	bw = bw.astype("float32")
	cv2.imshow("scorecard", bw)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def get_digits_from_scorecard(image):
	# print("shape", image.shape)
	bw = image < skimage.filters.threshold_local(image, 267)
	bw = bw.astype("float32")
	# cv2.imshow("Black and White", bw)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	digits = []
	for i in range(0, 7):
		digit = bw[298:298 + 50, 58 + 54 * i: 58 + 54 * i + 50]
		digits.append(digit)
	# cv2.imshow("digit", digit)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return digits


def construct_time(digits):
	time = str(digits[0]) + str(digits[1]) + ":" + str(digits[2]) + str(digits[3]) + "." + str(digits[4]) + str(
		digits[5]) + str(digits[6])
	return time


def keypoints(image):
	template = cv2.imread(
		"D:\\Google Drive\\UVa\\Classes\\Semester 6\\CS 4501\\Project 4\\test_images\\template_cross.png")
	# Use the SIFT descriptor to find keypoint features in the left and right images
	image = cv2.cvtColor((image * 255).astype("uint8"), cv2.COLOR_BGR2GRAY)
	template = cv2.cvtColor((template * 255).astype("uint8"), cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	keypoints_image, descriptor_image = sift.detectAndCompute(image, None)
	keypoints_template, descriptor_template = sift.detectAndCompute(template, None)

	# Compare the features in both images. For each feature in the first image,
	# find the closest matching feature in the other image
	# print 'Computing matching features.'
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(descriptor_image, descriptor_template, k=2)

	# Apply the ratio test to remove matches which are too similar to one another
	# Keeping only the unique matches
	# print 'Applying ratio test to matches.'
	good = []
	for m, n in matches:
		if m.distance < 0.5 * n.distance:
			good.append([m])

	img3 = cv2.drawMatchesKnn(image, keypoints_image, template, keypoints_template, good, None, flags=2)
	cv2.imshow("matches", img3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


scorecard_color, scorecard_gray = get_scorecard()
# keypoints(scorecard_color)

bw = scorecard_gray < skimage.filters.threshold_local(scorecard_gray, 267)
bw = bw.astype("float32")
cv2.imshow("scorecard", bw)
cv2.waitKey(0)
cv2.destroyAllWindows()


# for i in range(0, 7):
# 	image = scorecard_color[297:347, 56 + i * 54:56 + i * 54 + 50]
# 	test_contours(image)
# digits = get_digits_from_scorecard(scorecard_gray)
# predicted_digits = []
# for digit in digits:
# 	prediction = predict_digit(digit)
# 	print(prediction)
# 	which_digit = np.argmax(prediction)
# 	predicted_digits.append(which_digit)
# 	print(which_digit)
# print("Predicted Time:", construct_time(predicted_digits))
