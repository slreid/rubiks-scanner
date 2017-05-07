import skimage.transform
import skimage.filters
import numpy as np
import cv2
from keras import backend as K
from keras.models import load_model
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
		if m.distance < 0.55 * n.distance:
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


def predict_digits(digit_images, digit_flags):
	img_rows = 28
	img_cols = 28
	for i in range(len(digit_images)):
		if K.image_data_format() == 'channels_first':
			digit_images[i] = digit_images[i].reshape(1, img_rows, img_cols)
		else:
			digit_images[i] = digit_images[i].reshape(img_rows, img_cols, 1)
	model = load_model('CNN\\new_model.h5')
	print("# Digits:", len(digit_images))
	predictions = model.predict(np.asarray(digit_images))
	# print(predictions)
	predicted_digits = []
	flags = []
	i = 0
	for prediction in predictions:
		which_digit = np.argmax(prediction)
		confidence = np.max(prediction)
		print("Predicted", which_digit, "with confidence", confidence)
		predicted_digits.append(which_digit)
		if confidence < 0.75 or digit_flags[i] == 1:
			flags.append(1)
		else:
			flags.append(0)
		i += 1

	return predicted_digits, flags


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


def extract_digit(digit):
	cnts = cv2.findContours(img_as_ubyte(digit.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[1]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
	# loop over the digit area candidates
	added = False
	for c in cnts:
		# compute the bounding box of the contour
		(x, y, w, h) = cv2.boundingRect(c)
		# if the contour is sufficiently large, it must be a digit
		if not ((40 >= h >= 12) and (40 >= w >= 3)):
			continue
		added = True
		digit_crop = digit[y:y + h, x:x + w]
		resize_ratio = min(20. / w, 20. / h)
		resize_width = int(resize_ratio * w)
		resize_height = int(resize_ratio * h)
		digit_resized = skimage.transform.resize(digit_crop, (resize_height, resize_width), mode='constant')
		digit_28_28 = np.zeros((28, 28), dtype=float)
		lower_bound_y = int((28 - resize_height) / 2)
		upper_bound_y = int(resize_height + (28 - resize_height) / 2)
		lower_bound_x = int((28 - resize_width) / 2)
		upper_bound_x = int(resize_width + (28 - resize_width) / 2)
		digit_28_28[lower_bound_y:upper_bound_y, lower_bound_x:upper_bound_x] = digit_resized
		return digit_28_28, 0
	if not added:
		# If a digit was not found, add on a blank image and a flag to show it isn't a digit
		return np.zeros((28, 28)), 1


def get_digits_from_scorecard(image):
	bw = image < skimage.filters.threshold_local(image, 101)
	# plt.imshow(bw), plt.show()
	cv2.imwrite('presentation_images\\bw_digits.png', img_as_ubyte(bw))

	# cv2.imshow("Black and White", bw)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	digits = []
	flags = []

	for column in range(0, 3):
		min_y = 134
		max_y = 177
		min_x = 43 + 54 * column
		max_x = 90 + 54 * column
		digit, flag = extract_digit(bw[min_y:max_y, min_x:max_x])
		digits.append(digit)
		flags.append(flag)

	for row in range(0, 5):
		for column in range(0, 7):
			min_y = 233 + 49 * row
			max_y = 273 + 49 * row
			min_x = 43 + 54 * column
			max_x = 90 + 54 * column
			# cv2.imshow("bw", np.float32(bw[min_y:max_y, min_x:max_x]))
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			digit, flag = extract_digit(bw[min_y:max_y, min_x:max_x])
			# cv2.imshow("digit", np.float32(digit))
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			digits.append(digit)
			flags.append(flag)

	return digits, flags


def construct_id(digits):
	return str(digits[0]) + str(digits[1]) + str(digits[2])


def construct_times(digits):
	times = []
	for i in range(0, 5):
		round_time = str(digits[0 + 7*i]) + str(digits[1 + 7*i]) + ":" + str(digits[2 + 7*i]) + str(digits[3 + 7*i]) + ":" + str(digits[4 + 7*i]) + str(
			digits[5 + 7*i]) + str(digits[6 + 7*i])
		times.append(round_time)
	return times


def found_contour_of_template(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 20, 150)  # 75, 200
	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

	edged = cv2.cvtColor(np.float32(edged), cv2.COLOR_GRAY2BGR)

	rectangle = []
	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.03 * peri, True)  # 0.02
		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			rectangle = approx
			break
	if len(rectangle) > 0:
		point_1 = rectangle[0][0]
		point_2 = rectangle[1][0]
		point_3 = rectangle[2][0]

		dist_1_2 = np.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)
		dist_2_3 = np.sqrt((point_3[0] - point_2[0])**2 + (point_3[1] - point_2[1])**2)

		ratio = 0
		if dist_1_2 > dist_2_3:
			ratio = dist_1_2 / dist_2_3
			print("Found ratio of", ratio)
		else:
			ratio = dist_2_3 / dist_1_2
			print("Found ratio of", ratio)
		if 1.1 <= ratio <= 1.6:
			cv2.drawContours(image, [rectangle], -1, (0, 255, 0), 2)
			return True
		return False
	return False
