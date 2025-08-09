import json
import os
import sys
import time

from typing import Union, Set, List
from dotenv import load_dotenv

import cv2
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch

from src.car import Car

# Choose between:
# - yolov5n: Nano model, fastest but least accurate
# - yolov5s: Small model, good balance of speed and accuracy
# - yolov5m: Medium model, better accuracy but slower
# - yolov5l: Large model, high accuracy but slowest
# - yolov5x: Extra Large model, highest accuracy but very slow
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_categories = {
	2, # = car
	5, # = bus
	7, # = truck
	3, # = motorcycle
	4, # = bicycle
}

CLIPS_FOLDER = "/clips"
ANALYSIS_FOLDER = "/analysis"

# Load project specific environment variables
# These should be updated any time the camera orientation changes
load_dotenv(dotenv_path=".env")
def load_var(name: str, des_type: type):
	tmp = os.getenv(name)
	if tmp is None:
		print(f"Failed to load critical value {name} from env file")
		sys.exit(1)
	return des_type(tmp)
camera_fov = load_var("CAMERA_FOV_DEG", float)
pixel_left = load_var("CAMERA_LEFT_SIDE", int)
pixel_right = load_var("CAMERA_RIGHT_SIDE", int)
# Distance in meters between the two pixel values
actual_distance = load_var("ACTUAL_DISTANCE", float)
# Value to determine if the car is new or close enough to an existing car to be considered the same car
# not used for frame to frame tracking, but for situations where a car is added and dropped in the same frame
new_car_threshold = load_var("NEW_CAR_THRESHOLD", float)
tracker_persistence_count = load_var("CAR_TRACKER_PERSISTENCE", int)

def find_closest_car(cars: Set[Car], box: List[int]) -> Union[Car, None]:
	"""
	Find the closest car to the given bounding box based on similarity score,
	and removes it from the set of cars if found
	Parameters:
	- cars: A set of Car objects to search through
	- box: A list of bounding box coordinates [x1, y1, x2, y2]
	Returns:
	- The closest Car object if found, otherwise None
	"""
	# There's an error with min on an empty set, so filter that out first
	if len(cars) == 0:
		return None
	
	# loop through all the cars being tracked, find the best fit
	min_similarity = min(cars, key=lambda car: car.box_similarity(box))
	# variable unbound error is incorrectly reported here, variable is 100% defined
	sim_score = min_similarity.box_similarity(box)
	print(f"sim score {sim_score} <? {new_car_threshold}")
	if sim_score < new_car_threshold:
		return min_similarity
	return None

def annotate_with_speed(frame: cv2.typing.MatLike, car: Car, frame_count: int, framerate: float):
	# Ignore if we aren't tracking the car yet
	if len(car.speed_timing_frames) == 0:
		return
	# clamp the frame count to whatever is listed for the tracked car
	# this stops the calculation from shifting after we stop the timer
	end_frame = car.get_end_frame_count(frame_count)
	tracked_distance_pixels = car.get_tracked_distance_pixels()
	if tracked_distance_pixels is None:
		return
	# (actual distance * scaled pixel distance) roughly eaquals scaled actual distance
	meters = actual_distance * tracked_distance_pixels / float(pixel_right-pixel_left)
	if (end_frame-car.speed_timing_frames[0]) == 0 or framerate == 0:
		return
	seconds = (end_frame-car.speed_timing_frames[0])/framerate
	meters_per_second = meters / seconds
	# convert back to freeeeeeddooooooom units
	miles_per_hour = meters_per_second * 2.23694
	label = f"Speed: {miles_per_hour:.2f} mph"
	cv2.putText(
		frame, label, (int(car.current_center[0]), int(car.current_center[1])-10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
	)

def process_video(clips_folder: str, filename: str, output_folder: str):
	fpath = os.path.join(clips_folder, filename)
	opath = os.path.join(output_folder, "output_"+filename)
	print("analyzing", fpath)
	# Lets be lazy pieces of shit and catch all exceptions and errors,
	# and this is what we'll return if things go pearshaped
	tracked_cars: Set[Car] = set()
	analysis_results = {
		"status": "An Error occured processing file ",
		"cars": []
	}

	# Okay so lets give it the old college try then...
	cap = cv2.VideoCapture(fpath)
	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	framerate = cap.get(cv2.CAP_PROP_FPS)
	num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print(f"handling {num_frames} frames")
	# idfk man, pylance gives an error but the code runs
	# something to do with the way opencv binds to codecs at runtime
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
	print("Creating annotated video", opath)
	out = cv2.VideoWriter(opath, fourcc, framerate, (frame_width, frame_height), isColor=True)
	if not out.isOpened():
		print("Failed to create output video, continuing anyways")

	frame_count = 0
	car_tags = 0
	frame_diagnostics = []
	while True:
		# capture frame by frame
		ret, frame = cap.read()
		frame_count += 1
		if ret == False or frame is None:
			break

		if frame_count % 100 == 0:
			print(f"Calculated {frame_count} frames")

		# detect cars in the video, idk why pylance complains about pytorch here...
		yolo_results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # type: ignore

		frame_data = {}
		frame_data["detections"] = len(yolo_results.xyxy[0])
		frame_data["boxes"] = []
		car_tags += len(yolo_results.xyxy[0])

		# Loop through all the detections, only process vehicles
		for *box, conf, cls in yolo_results.xyxy[0]:
			if int(cls) in yolo_categories:
				x1, y1, x2, y2 = map(int, box)
				# Find a similar car already being tracked, add it if it's new
				closest_car = find_closest_car(tracked_cars, [x1, y1, x2, y2])
				if closest_car is None:
					# Camera found a new car that isn't tracked yet!
					print("Found a new car on frame", frame_count)
					print("Tracked car list size", len(tracked_cars))
					closest_car = Car([x1, y1, x2, y2], conf.item(), frame_count)
				closest_car.update_box([x1, y1, x2, y2], conf.item(), frame_count)
				tracked_cars.add(closest_car)
				if pixel_left < x2:
					closest_car.start_tracking(frame_count)
				if pixel_right < x2:
					closest_car.end_tracking(frame_count)
				# Record the box data to output and image
				frame_data["boxes"].append({
					"x1": x1,
					"y1": y1,
					"x2": x2,
					"y2": y2,
					"confidence": conf.item(),
					"class": int(cls)
				})
				cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
				# Annotate the current speed onto the box
				annotate_with_speed(frame, closest_car, frame_count, framerate)
		# Lets carry over cars in case a frame loses a track for some reason
		# if we ever fail on a detection we want to keep a goldfish memory of
		# cars we just saw in case they "come back"
		cars_to_remove = []
		for car in tracked_cars:
			if car.is_expired(frame_count, tracker_persistence_count):
				cars_to_remove.append(car)
		for car in cars_to_remove:
			tracked_cars.remove(car)

		frame_diagnostics.append(frame_data)
		out.write(frame)

	analysis_results["status"] = "Successfully finished processing " + fpath
	analysis_results["total_reads"] = car_tags
	analysis_results["frame_diagnostics"] = frame_diagnostics
	cap.release()
	out.release()
	with open(os.path.join(output_folder, filename+".json"), 'w') as fp:
		json.dump(analysis_results, fp)
	print("done with", fpath)
	return analysis_results
	
def get_list_of_clips():
	return os.listdir(CLIPS_FOLDER)

def video_already_processed(clip_path: str):
	analysis_contents = os.listdir(ANALYSIS_FOLDER)
	if clip_path in analysis_contents:
		return True
	return False

if __name__ == "__main__":
	loop_count: int = 0
	while True:
		if loop_count != 0:
			time.sleep(20)
		loop_count += 1
		# scan folder for video sources
		clips_contents = get_list_of_clips()
		if len(clips_contents) == 0:
			print("No clips found, sleeping...")
			continue
		# scan folder for analysis
		# if video doesn't have analysis, start chugging
		for path in clips_contents:
			print("Found video: ", path)
			if video_already_processed("output_"+path):
				print("Found output_"+path+" -- skipping.")
				continue

			# dig through it, record all vehicles, return a dict of results
			process_video(CLIPS_FOLDER, path, ANALYSIS_FOLDER)
