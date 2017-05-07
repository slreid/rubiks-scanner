import skimage.transform
import skimage.filters
import urllib.request
import numpy as np
import sys
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

		h_2, w_2, _ = img2.shape
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
	digitCnts = []
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
		# cv2.imshow("Show", digit_draw)
		# cv2.waitKey()
		# cv2.destroyAllWindows()
		digit_crop = digit[y:y + h, x:x + w]
		# plt.imshow(digit), plt.show()
		# plt.imshow(digit_crop), plt.show()
		resize_ratio = min(20. / w, 20. / h)
		resize_width = int(resize_ratio * w)
		resize_height = int(resize_ratio * h)
		digit_resized = skimage.transform.resize(digit_crop, (resize_height, resize_width), mode='constant')
		# plt.imshow(digit), plt.show()
		# plt.imshow(digit_resized), plt.show()
		digit_28_28 = np.zeros((28, 28), dtype=float)
		lower_bound_y = int((28 - resize_height) / 2)
		upper_bound_y = int(resize_height + (28 - resize_height) / 2)
		lower_bound_x = int((28 - resize_width) / 2)
		upper_bound_x = int(resize_width + (28 - resize_width) / 2)
		digit_28_28[lower_bound_y:upper_bound_y, lower_bound_x:upper_bound_x] = digit_resized
		# plt.imshow(digit, 'gray'), plt.show()
		# plt.imshow(digit_28_28, 'gray'), plt.show()
		# cv2.imwrite('presentation_images\\digits\\bw_' + str(row_num) + '_' + str(column + 1) + '.png',
		#             img_as_ubyte(digit_crop))
		# cv2.imwrite('presentation_images\\digits\\bw_28_' + str(row_num) + '_' + str(column + 1) + '.png',
		#             img_as_ubyte(digit_28_28))
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


def start_video_stream(host):
	if len(sys.argv) > 1:
		host = sys.argv[1]

	hoststr = 'http://' + host + '/video?x.mjpeg'
	print('Streaming ' + hoststr)

	stream = urllib.request.urlopen(hoststr)

	template = cv2.imread('test_images\\template_new.png')
	bytes_from_stream = bytes()
	i = 0
	while True:
		bytes_from_stream += stream.read(1024)
		a = bytes_from_stream.find(b'\xff\xd8')
		b = bytes_from_stream.find(b'\xff\xd9')
		if a != -1 and b != -1:
			jpg = bytes_from_stream[a:b + 2]
			bytes_from_stream = bytes_from_stream[b + 2:]
			frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
			cv2.imshow(host, frame)
			if i % 30 == 0:
				adjusted_image = get_scorecard_sift(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), template)
				if adjusted_image is not None:
					cv2.imshow("image", adjusted_image)
					cv2.waitKey(0)
					cv2.destroyAllWindows()
					all_digits, digit_flags = get_digits_from_scorecard(adjusted_image)
					predictions, prediction_flags = predict_digits(all_digits, digit_flags)
					comp_id = construct_id(predictions[0:3])
					times = construct_times(predictions[3:])
					print("Comp ID:", comp_id)
					for time in times:
						print(time)
					print(prediction_flags)
					prediction_flags_formatted = []
					# Handle ID
					for i in range(0, 2):
						prediction_flags_formatted.append([])
						if prediction_flags[i] == 1:
							prediction_flags_formatted[0].append(str(i))
					# Handle the 5 rounds
					for rounds in range(0, 4):
						prediction_flags_formatted.append([])
						for i in range(0, 6):
							if prediction_flags[3 + i + 7*rounds] == 1:
								prediction_flags_formatted[rounds + 1].append(str(i))
					print(prediction_flags_formatted)
					addInfoToDatabase(comp_id, times, prediction_flags_formatted)
					getWinners()

			# Press escape to close
			if cv2.waitKey(1) == 27:
				exit(0)
		i += 1

start_video_stream("192.168.137.94:8080")
