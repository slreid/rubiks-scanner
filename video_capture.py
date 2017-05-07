import numpy as np
import cv2

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
		cv2.imshow("image", img2)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		h_2, w_2 = img2.shape
		adjusted_image = cv2.warpPerspective(image, M, (w_2, h_2))
		cv2.imshow("Warped", adjusted_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		return adjusted_image
	else:
		print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
		matchesMask = None
		return None


cap = cv2.VideoCapture(0)
template = cv2.imread('test_images\\template_new.png', 0)
while True:
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	detected_scorecard = get_scorecard_sift(gray, template)
	if detected_scorecard is None:
		print("No scorecard found.")
	else:
		print("Got a scorecard.")
		break
	# Display the resulting frame
	cv2.imshow('frame', gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break  # When everything done, release the capture

cap.release()
cv2.destroyAllWindows()

cv2.imshow("scorecard", detected_scorecard)
cv2.waitKey(0)
cv2.destroyAllWindows()

