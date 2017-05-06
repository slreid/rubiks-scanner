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
from skimage import img_as_ubyte


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
	# plt.imshow(bw), plt.show()
	cv2.imwrite('presentation_images\\bw_digits.png', img_as_ubyte(bw))

	# cv2.imshow("Black and White", bw)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	digits = []
	for column in range(0, 7):
		min_y = 233 + 49 * adjusted_row_num
		max_y = 273 + 49 * adjusted_row_num
		min_x = 43 + 54 * column
		max_x = 90 + 54 * column
		digit_not_bw = image[min_y:max_y, min_x:max_x]
		digit = bw[min_y:max_y, min_x:max_x]
		# plt.imshow(digit_not_bw), plt.show()
		# plt.imshow(digit), plt.show()

		cnts = cv2.findContours(img_as_ubyte(digit.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[1]
		digitCnts = []
		digit_draw = digit_not_bw.copy()
		# print("Found", cnts, "number of contours")
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
		# loop over the digit area candidates
		added = False
		for c in cnts:
			# compute the bounding box of the contour
			(x, y, w, h) = cv2.boundingRect(c)
			# if the contour is sufficiently large, it must be a digit
			if not ((40 >= h >= 12) and (40 >= w >= 3)):
				continue
			else:
				added = True
			digitCnts.append(c)
			cv2.rectangle(digit_draw, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
			# cv2.imshow("Show", digit_draw)
			# cv2.waitKey()
			# cv2.destroyAllWindows()
			digit_crop = digit[y:y+h, x:x+w]
			# plt.imshow(digit), plt.show()
			# plt.imshow(digit_crop), plt.show()
			resize_ratio = min(20./w, 20./h)
			resize_width = int(resize_ratio * w)
			resize_height = int(resize_ratio * h)
			digit_resized = skimage.transform.resize(digit_crop, (resize_height, resize_width), mode='constant')
			# plt.imshow(digit), plt.show()
			# plt.imshow(digit_resized), plt.show()
			digit_28_28 = np.zeros((28, 28), dtype=float)
			lower_bound_y = int((28 - resize_height)/2)
			upper_bound_y = int(resize_height + (28 - resize_height)/2)
			lower_bound_x = int((28 - resize_width)/2)
			upper_bound_x = int(resize_width + (28 - resize_width)/2)
			digit_28_28[lower_bound_y:upper_bound_y, lower_bound_x:upper_bound_x] = digit_resized
			# plt.imshow(digit, 'gray'), plt.show()
			# plt.imshow(digit_28_28, 'gray'), plt.show()
			cv2.imwrite('presentation_images\\digits\\' + str(row_num) + '_' + str(column + 1) + '.png',
			            img_as_ubyte(digit_not_bw))
			cv2.imwrite('presentation_images\\digits\\box_' + str(row_num) + '_' + str(column + 1) + '.png',
						img_as_ubyte(digit_draw))
			cv2.imwrite('presentation_images\\digits\\bw_' + str(row_num) + '_' + str(column + 1) + '.png',
						img_as_ubyte(digit_crop))
			cv2.imwrite('presentation_images\\digits\\bw_28_' + str(row_num) + '_' + str(column + 1) + '.png',
						img_as_ubyte(digit_28_28))
			digits.append(img_as_ubyte(digit_28_28))
		if not added:
			digits.append(None)


	# cv2.imshow("digit", digit)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
	return digits


def construct_id(digit_images):
	digits = []
	digit_flags = []
	i = 0
	for digit in digit_images:
		predicted_digit, confidence = predict_digit(digit)
		print("Predicted:", predicted_digit, "with confidence", 100*confidence)
		digits.append(predicted_digit)
		if confidence < 0.5:
			digit_flags.append(str(i))
		i += 1
	comp_ip = str(digits[0]) + str(digits[1]) + str(digits[2])
	return comp_ip, digit_flags


def construct_time(digit_images):
	digits = []
	digit_flags = []
	i = 0
	for digit in digit_images:
		if digit is None:
			predicted_digit = 0
			confidence = 0.
		else:
			predicted_digit, confidence = predict_digit(digit)
		predicted_digit, confidence = predict_digit(digit)
		print("Predicted:", predicted_digit, "with confidence", 100*confidence)
		digits.append(predicted_digit)
		if confidence < 0.5:
			digit_flags.append(str(i))
		i += 1
	time = str(digits[0]) + str(digits[1]) + ":" + str(digits[2]) + str(digits[3]) + ":" + str(digits[4]) + str(
			digits[5]) + str(digits[6])
	return time, digit_flags

all_times = []
image = cv2.imread('test_images\\samanthas_shadow.jpg', 0)
template = cv2.imread('test_images\\template_new.png', 0)
adjusted_image = get_scorecard_sift(image, template)
if adjusted_image is None:
	print("Could not extract image")
else:
	all_flags = []
	id_digits = get_id_from_scorecard(adjusted_image)
	comp_ip, flags = construct_id(id_digits)
	all_flags.append(flags)
	print("Competitor id:", comp_ip)
	for row in range(1, 6):
		row_of_digits = get_row_of_digits_from_scorecard(adjusted_image, row)
		constructed_time, flags = construct_time(row_of_digits)
		all_flags.append(flags)
		all_times.append(constructed_time)
		print(constructed_time)
	for flag in all_flags:
		print(flag)
	addInfoToDatabase(comp_ip, all_times, all_flags)
	getWinners()

