import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('test_images\\final_template_test.jpg', 0)
template = cv2.imread('test_images\\template_new.png', 0)

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp_image = orb.detect(image, None)
kp_template = orb.detect(template, None)

# compute the descriptors with ORB
kp_image, des_image = orb.compute(image, kp_image)
kp_template, des_template = orb.compute(template, kp_template)

# draw only keypoints location,not size and orientation
image_keypoints = cv2.drawKeypoints(image, kp_image, None, color=(0, 255, 0), flags=0)
plt.imshow(image_keypoints), plt.show()

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des_image, des_template)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img_matches = cv2.drawMatches(image, kp_image, template, kp_template, matches[:10], None, flags=2)

plt.imshow(img_matches), plt.show()

############ Flann Matcher

# MIN_MATCH_COUNT = 10
#
# FLANN_INDEX_KDTREE = 0
# FLANN_INDEX_LSH = 6
# index_params = dict(algorithm=FLANN_INDEX_LSH,
#                     table_number=6,  # 12
#                     key_size=12,  # 20
#                     multi_probe_level=1)  # 2
# # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)
#
# flann = cv2.FlannBasedMatcher(index_params, search_params)
#
# matches = flann.knnMatch(des_image, des_template, k=2)

######## Brute Force Matcher

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des_image, des_template)

# # Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# store all the good matches as per Lowe's ratio test.
# good = []
# for m_n in matches:
# 	if len(m_n) != 2:
# 		continue
# 	(m, n) = m_n
# 	if m.distance < 0.6 * n.distance:
# 		good.append(m)
good = matches

print("# Matches: ", len(good))

if len(good) > 0:
	# Get a list of matching points in the scene image
	src_pts = np.float32([kp_image[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
	# Get a list of matching points in the template image
	dst_pts = np.float32([kp_template[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
	# Find a homography which maps scene points to the template points
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10)
	matchesMask = mask.ravel().tolist()
	print(M)
	# Get the corners of the scene image
	h, w = image.shape
	pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
	print(pts)
	# Use the homography to warp the corners of the scene image to the template image
	dst = cv2.perspectiveTransform(pts, M)
	print(dst)
	# Draw the warped scene corners to the template image in the scene image
	img2 = cv2.polylines(template, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
	cv2.imshow("image", img2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# Warp the scene image to the size of the template image
	h_2, w_2 = img2.shape
	adjusted_image = cv2.warpPerspective(image, M, (w_2, h_2))
	cv2.imshow("image", adjusted_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
# cv2.imshow("Warped", adjusted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
