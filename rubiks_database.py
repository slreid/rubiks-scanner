import pyrebase
from datetime import datetime

# Configure Firebase
config = {
	"apiKey": "AIzaSyDyTulUCX5swJauYm8YM55Vr5a5vv8ipPQ",
	"authDomain": "rubiksscanner.firebaseapp.com",
	"databaseURL": "https://rubiksscanner.firebaseio.com",
	"storageBucket": "rubiksscanner.appspot.com"
}

# Get database
firebase = pyrebase.initialize_app(config)
db = firebase.database()


# Used to compute the average number of seconds to complete each round
def computeAvgSeconds(comp_id, times):
	all_times = []
	for r, time in enumerate(times):
		# add each round's time to database
		round = r + 1
		db.child("EventName").child("Competitors").child(comp_id).child(round).set(time)

		# get time in total seconds
		time = time.replace(".", ":")
		divided = time.split(":")
		minutes = divided[0]
		seconds = divided[1]
		miliseconds = divided[2]
		total_seconds = float(minutes) * 60 + float(seconds) + float(miliseconds) * 0.001
		# print(total_seconds)

		# add each time to list
		all_times.append(total_seconds)

	# remove max and min time from list
	all_times.remove(max(all_times))
	all_times.remove(min(all_times))

	# compute average of remaining times
	average_time = sum(all_times) / 3

	return average_time


# Used to convert average time back to string format
def convertTimeFormat(average_time):
	int_time = int(average_time)

	miliseconds = "%.3f" % (average_time - int_time)
	minutes = int(average_time / 60)
	seconds = int_time - minutes * 60

	# add 0 to front of minutes if <10
	minutes = str(minutes)
	if len(minutes) == 1:
		minutes = "0" + minutes

	# same with seconds
	seconds = str(seconds)
	if len(seconds) == 1:
		seconds = "0" + seconds

	avg_string = minutes + ":" + seconds + "." + str(miliseconds[2:])

	return avg_string


# Add info to database given id of competitor and array of 5 strings representing
# the times it took to complete each round
def addInfoToDatabase(comp_id, times, flagged):
	seconds = computeAvgSeconds(comp_id, times)
	print(seconds)

	avg_time = convertTimeFormat(seconds)
	print(avg_time)

	# create strings to add to database for flagged solve times

	flagged_id = ""
	for f in flagged[0]:
		flagged_id += f + ","

	flagged_1 = ""
	for f in flagged[1]:
		flagged_1 += f + ","

	flagged_2 = ""
	for f in flagged[2]:
		flagged_2 += f + ","

	flagged_3 = ""
	for f in flagged[3]:
		flagged_3 += f + ","

	flagged_4 = ""
	for f in flagged[4]:
		flagged_4 += f + ","

	flagged_5 = ""
	for f in flagged[5]:
		flagged_5 += f + ","

	# add the average time and average seconds to database
	db.child("EventName").child("Competitors").child(comp_id).child("seconds").set(seconds)
	db.child("EventName").child("Competitors").child(comp_id).child("avg").set(avg_time)
	db.child("EventName").child("Competitors").child(comp_id).child("Flagged").child("ID").set(flagged_id)
	db.child("EventName").child("Competitors").child(comp_id).child("Flagged").child(1).set(flagged_1)
	db.child("EventName").child("Competitors").child(comp_id).child("Flagged").child(2).set(flagged_2)
	db.child("EventName").child("Competitors").child(comp_id).child("Flagged").child(3).set(flagged_3)
	db.child("EventName").child("Competitors").child(comp_id).child("Flagged").child(4).set(flagged_4)
	db.child("EventName").child("Competitors").child(comp_id).child("Flagged").child(5).set(flagged_5)


# Return Dictionary of competitors organized in order of average completion time
def getWinners():
	# dictionary of competitors
	winners_dict = {}
	competitors = db.child("EventName").child("Competitors").get()

	# add competitor id/times to dictionary and check if times are correct
	for c in competitors.each():
		if type(c.val()) is dict:
			comp_id = str(c.key())
			seconds = c.val()['seconds']
			round_1 = c.val()['1']
			round_2 = c.val()['2']
			round_3 = c.val()['3']
			round_4 = c.val()['4']
			round_5 = c.val()['5']
			times = [round_1, round_2, round_3, round_4, round_5]
			new_seconds = computeAvgSeconds(comp_id, times)
			# check to see if round data has been changed
			if (new_seconds == seconds):
				winners_dict[comp_id] = seconds
			else:
				# if data changed, update average times in database
				print("in else for id" + comp_id)
				winners_dict[comp_id] = new_seconds
				new_average = convertTimeFormat(new_seconds)
				db.child("EventName").child("Competitors").child(comp_id).child("seconds").set(new_seconds)
				db.child("EventName").child("Competitors").child(comp_id).child("avg").set(new_average)


	# sort dictionary by average seconds
	ordered_winners = sorted(winners_dict.items(), key=lambda x: x[1])

	print("Winners")  # Print winners

	place = 1
	for key, value in ordered_winners:
		# get average time for each competitor from database
		time = db.child("EventName").child("Competitors").child(str(key)).child('avg').get()
		print(str(place) + ". Competitor id " + str(key) + ": " + time.val())
		place += 1

	return ordered_winners