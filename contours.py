# import the necessary packages
import imutils
from skimage.filters import threshold_adaptive
import numpy as np
import argparse
import cv2
import urllib.request
import sys



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


def contour_test(image):
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height=500)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 20, 150)  # 75, 200
	# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

	edged = cv2.cvtColor(np.float32(edged), cv2.COLOR_GRAY2BGR)

	screenCnt = []
	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.03 * peri, True)  # 0.02
		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break
	return screenCnt, edged

	# apply the four point transform to obtain a top-down
	# view of the original image
	# warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
	# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	# warped = threshold_adaptive(warped, 251, offset=10)
	# warped = warped.astype("uint8") * 255


def start_video_stream(host):
	if len(sys.argv) > 1:
		host = sys.argv[1]

	hoststr = 'http://' + host + '/video?x.mjpeg'
	print('Streaming ' + hoststr)

	stream = urllib.request.urlopen(hoststr)

	template = cv2.imread('test_images\\template_new.png')
	bytes_from_stream = bytes()
	i = 0
	contours = []
	while True:
		bytes_from_stream += stream.read(1024)
		a = bytes_from_stream.find(b'\xff\xd8')
		b = bytes_from_stream.find(b'\xff\xd9')
		if a != -1 and b != -1:
			jpg = bytes_from_stream[a:b + 2]
			bytes_from_stream = bytes_from_stream[b + 2:]
			frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
			contours, edge = contour_test(frame)
			if i % 30 == 0:
				contours = []
			if len(contours) > 0:
				print(contours)
				cv2.drawContours(frame, [contours], -1, (0, 255, 0), 2)
				stream = urllib.request.urlopen(hoststr)
				bytes_from_stream = bytes()

			cv2.imshow(host, frame)
			# Press escape to close
			if cv2.waitKey(1) == 27:
				exit(0)
		i += 1

start_video_stream("192.168.137.94:8080")
