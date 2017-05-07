import cv2
import urllib.request
import numpy as np
from rubiks_database import getWinners, addInfoToDatabase
from rubiks_scanner_core import get_scorecard_sift, get_digits_from_scorecard, predict_digits, construct_id, construct_times, found_contour_of_template


host = "192.168.137.94:8080"

hoststr = 'http://' + host + '/video?x.mjpeg'
print('Streaming ' + hoststr)

stream = urllib.request.urlopen(hoststr)

template = cv2.imread('test_images\\template_new.png', 0)
bytes_from_stream = bytes()
frame_count = 0
while True:
	bytes_from_stream += stream.read(1024)
	a = bytes_from_stream.find(b'\xff\xd8')
	b = bytes_from_stream.find(b'\xff\xd9')
	if a != -1 and b != -1:
		jpg = bytes_from_stream[a:b + 2]
		bytes_from_stream = bytes_from_stream[b + 2:]
		frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
		cv2.imshow(host, frame)
		if frame_count % 5 == 0:
			# Every fifth frame, look in the video frame for a contour which matches the rectangle shape of the template
			found = found_contour_of_template(frame)
			cv2.imshow(host, frame)
			if found:
				# Extract the scorecard using SIFT from the frame containing the good contour
				adjusted_image = get_scorecard_sift(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), template)
				if adjusted_image is not None:
					cv2.imshow("image", adjusted_image)
					cv2.waitKey(0)
					cv2.destroyAllWindows()
					# Extract the digits
					all_digits, digit_flags = get_digits_from_scorecard(adjusted_image)
					# Run all the extracted digits through the neural network
					predictions, prediction_flags = predict_digits(all_digits, digit_flags)
					# Construct the competitor IDs and the solve times for each round
					comp_id = construct_id(predictions[0:3])
					times = construct_times(predictions[3:])
					# Print our results
					print("Comp ID:", comp_id)
					for time in times:
						print(time)
					print(prediction_flags)

					# Send the competitor ID and the solve times off to the database
					# We must format our 'bad digit' flags for the database
					prediction_flags_formatted = []
					# Handle ID
					prediction_flags_formatted.append([])
					for i in range(0, 2):
						if prediction_flags[i] == 1:
							prediction_flags_formatted[0].append(str(i))
					# Handle the 5 rounds
					for rounds in range(0, 5):
						prediction_flags_formatted.append([])
						for i in range(0, 7):
							if prediction_flags[3 + i + 7 * rounds] == 1:
								prediction_flags_formatted[rounds + 1].append(str(i))
					print(prediction_flags_formatted)
					addInfoToDatabase(comp_id, times, prediction_flags_formatted)
					getWinners()
			stream = urllib.request.urlopen(hoststr)
			bytes_from_stream = bytes()

		# Press escape to close
		if cv2.waitKey(1) == 27:
			exit(0)
	frame_count += 1