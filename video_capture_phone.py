import cv2
import urllib.request
import numpy as np
import sys

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



host = "192.168.137.82:8080"
if len(sys.argv) > 1:
	host = sys.argv[1]

hoststr = 'http://' + host + '/video?x.mjpeg'
print('Streaming ' + hoststr)

stream = urllib.request.urlopen(hoststr)

template = cv2.imread('test_images\\template_new.png')
bytes = bytes()
i = 0
while True:
	bytes += stream.read(1024)
	a = bytes.find(b'\xff\xd8')
	b = bytes.find(b'\xff\xd9')
	if a != -1 and b != -1:
		jpg = bytes[a:b + 2]
		bytes = bytes[b + 2:]
		frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
		cv2.imshow(host, frame)
		if i % 30 == 0:
			adjusted_image = get_scorecard_sift(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), template)
			if adjusted_image is not None:
				cv2.imshow("image", adjusted_image)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
		# Press escape to close
		if cv2.waitKey(1) == 27:
			exit(0)
	i += 1

