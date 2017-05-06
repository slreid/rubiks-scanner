import cv2
import urllib.request
import numpy as np
import sys

host = "192.168.137.82:8080"
if len(sys.argv) > 1:
	host = sys.argv[1]

hoststr = 'http://' + host + '/video?x.mjpeg'
print('Streaming ' + hoststr)

stream = urllib.request.urlopen(hoststr)

<<<<<<< HEAD
bytes = bytes()
while True:
	bytes += stream.read(1024)
	a = bytes.find(b'\xff\xd8')
	b = bytes.find(b'\xff\xd9')
	if a != -1 and b != -1:
		jpg = bytes[a:b + 2]
		bytes = bytes[b + 2:]
		i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
		cv2.imshow('i', i)
		if cv2.waitKey(1) == 27:
			exit(0)
=======
bytes = ''
while True:
	bytes += str(stream.read(1024))
	print(bytes)
	a = bytes[3:].find('\xff\xd8')
	b = bytes[3:].find('\xff\xd9')
	print(a, b)
	if a != -1 and b != -1:
		jpg = bytes[a:b + 2]
		bytes = bytes[b + 2:]
		i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
		cv2.imshow(hoststr, i)
		if cv2.waitKey(1) == 27:
			exit(0)

>>>>>>> 72fe82e43a8dda41286475ad4e92385e6d68860b
