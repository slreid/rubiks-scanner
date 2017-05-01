import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage.transform
import skimage.filters
import numpy as np
import cv2
from keras import backend as K
from keras.models import load_model
from rubiks_database import getWinners, addInfoToDatabase


def get_scorecard_sift(image, template):

	MIN_MATCH_COUNT = 10

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(image, None)
	kp2, des2 = sift.detectAndCompute(template, None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)

	matches = flann.knnMatch(des1, des2, k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m, n in matches:
		if m.distance < 0.5 * n.distance:
			good.append(m)

	print("# Matches: ", len(good))

	if len(good) > MIN_MATCH_COUNT:
		src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
		dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()
		print(M)
		h, w = image.shape
		pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
		dst = cv2.perspectiveTransform(pts, M)
		img2 = cv2.polylines(template, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

		h_2, w_2 = img2.shape
		adjusted_image = cv2.warpPerspective(image, M, (w_2, h_2))
		# cv2.imshow("Warped", adjusted_image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		return adjusted_image
	else:
		print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
		matchesMask = None
		return None


def predict_digit(image):
	# input image dimensions
	img_rows, img_cols = 28, 28
	image = skimage.transform.resize(image, (img_cols, img_rows), mode='constant')
	if K.image_data_format() == 'channels_first':
		image = image.reshape(1, img_rows, img_cols)
	else:
		image = image.reshape(img_rows, img_cols, 1)

	model = load_model('CNN\\samantha.h5')

	prediction = model.predict(np.asarray([image]))[0]
	which_digit = np.argmax(prediction)
	confidence = np.max(prediction)
	return which_digit, confidence


def get_id_from_scorecard(image):
	bw = image < skimage.filters.threshold_local(image, 101)
	bw = bw.astype("float32")
	digits = []
	for column in range(0, 3):
		min_y = 134
		max_y = 177
		min_x = 43 + 54 * column
		max_x = 90 + 54 * column
		digit = bw[min_y:max_y, min_x:max_x]
		digits.append(digit)
		# cv2.imshow("digit", digit)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
	return digits


def get_row_of_digits_from_scorecard(image, row_num):
	adjusted_row_num = row_num - 1
	# print("shape", image.shape)
	bw = image < skimage.filters.threshold_local(image, 101)
	bw = bw.astype("float32")
	# cv2.imshow("Black and White", bw)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	digits = []
	for column in range(0, 7):
		min_y = 233 + 49 * adjusted_row_num
		max_y = 273 + 49 * adjusted_row_num
		min_x = 43 + 54 * column
		max_x = 90 + 54 * column
		digit = bw[min_y:max_y, min_x:max_x]
		digits.append(digit)
		# cv2.imshow("digit", digit)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
	return digits


def construct_id(digit_images):
	digits = []
	for digit in digit_images:
		predicted_digit, confidence = predict_digit(digit)
		print("Predicted:", predicted_digit, "with confidence", 100*confidence)
		digits.append(predicted_digit)
	comp_ip = str(digits[0]) + str(digits[1]) + str(digits[2])
	return comp_ip


def construct_time(digit_images):
	digits = []
	for digit in digit_images:
		predicted_digit, confidence = predict_digit(digit)
		print("Predicted:", predicted_digit, "with confidence", 100*confidence)
		digits.append(predicted_digit)
	time = str(digits[0]) + str(digits[1]) + ":" + str(digits[2]) + str(digits[3]) + ":" + str(digits[4]) + str(
		digits[5]) + str(digits[6])
	return time

all_times = []
image = cv2.imread('test_images\\samanthas_shadow.jpg', 0)
template = cv2.imread('test_images\\template_new.png', 0)
adjusted_image = get_scorecard_sift(image, template)
if adjusted_image is None:
	print("Could not extract image")
else:
	id_digits = get_id_from_scorecard(adjusted_image)
	comp_ip = construct_id(id_digits)
	print("Competitor id:", comp_ip)
	for row in range(1, 6):
		row_of_digits = get_row_of_digits_from_scorecard(adjusted_image, row)
		constructed_time = construct_time(row_of_digits)
		all_times.append(constructed_time)
		print(constructed_time)

	addInfoToDatabase(comp_ip, all_times)
	getWinners()

