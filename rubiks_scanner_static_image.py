import cv2
from rubiks_database import getWinners, addInfoToDatabase
from rubiks_scanner_core import get_scorecard_sift, get_digits_from_scorecard, predict_digits, construct_id, construct_times

# Grab the scorecard image and the template image
image = cv2.imread('test_images\\final_template_test.jpg', 0)
template = cv2.imread('test_images\\template_new.png', 0)
# Extract the scorecard using SIFT
adjusted_image = get_scorecard_sift(image, template)
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

