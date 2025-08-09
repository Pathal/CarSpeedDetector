import math
from typing import List, Union

class Car:
	"""
	- "frame": the frame number where the car was detected
	- "box": the bounding box coordinates of the car in the format [x1, y1, x2, y2]
			 the (x2,y2) coordinate is the right side of the car from the camera's perspective
	- "confidence": the confidence score of the detection
	"""

	BOX_EDGE_INDEX = 2 # Index of the right edge of the bounding box in the list

	def __init__(self, box: List[int], confidence: float, frame_count: int):
		self.speed_timing_frames = [] # List to store frame nums for speed calculation
		self.timing_frame_column = [] # Store the column of the car's bounds for that frame
		self.initial_box = box
		self.last_update = frame_count
		self.current_box = box
		self.current_center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
		self.confidence = confidence
		self.direction = 0 # 0 = unknown, 1 = to the right, -1 = to the left

	def box_similarity(self, other_box: List[int]) -> float:
		"""
		Lower score is better, 0 is perfect match

		Parameters:
		- other_box: A list of bounding box coordinates [x1, y1, x2, y2]

		Returns:
		- A float representing the similarity score between the current box and the other box
		"""
		self_width = self.current_box[2] - self.current_box[0]
		other_width = other_box[2] - other_box[0]
		direction_check = 1
		if self.direction == 1 and other_box[0] < self.current_box[0]:
			# If this car's direction is in the opposite direction, double the error
			direction_check = 2
		elif self.direction == -1 and self.current_box[0] < other_box[0]:
			# If the car is going to them left, the other box should be to the left of the current box
			direction_check = 2
		# Add the distance of one edge to the other edge
		# and the width of the box to the other box together
		# and that the direction is consistent (other - current means positive to the right)
		# the more similar both of these metrics are, the more similar the boxes are
		# good enough for government work in any case
		return (math.fabs(other_box[0] - self.current_box[0]) +
		  		math.fabs(self_width - other_width)) \
				* direction_check
	
	def update_box(self, new_box: List[int], confidence: float, frame_count: int):
		"""
		MUST BE CALLED BEFORE ANY OTHER METHOD THAT USES THIS OBJECT

		new_box: the bounding box coordinates of the car in the format [x1, y1, x2, y2]
		confidence: new confidence value
		"""
		self.current_box = new_box
		self.last_update = frame_count
		self.confidence = confidence
		self.direction = 1 if self.current_box[0] > self.initial_box[0] else -1
		self.current_center = ((new_box[0]+new_box[2])/2, (new_box[1]+new_box[3])/2)

	def start_tracking(self, frame_id: int):
		"""
		Must be called after update_box to start tracking the car
		"""
		if len(self.speed_timing_frames) == 0:
			self.speed_timing_frames.append(frame_id)
			self.timing_frame_column.append(self.current_box[Car.BOX_EDGE_INDEX])

	def end_tracking(self, frame_id: int):
		"""
		Must be called after update_box to stop tracking the car
		"""
		if len(self.speed_timing_frames) == 1:
			self.speed_timing_frames.append(frame_id)
			self.timing_frame_column.append(self.current_box[Car.BOX_EDGE_INDEX])

	def get_end_column(self, default_value: int) -> int:
		"""
		Returns the column of the car's bounds at the end of the speed tracking
		or a default value if not available
		"""
		if len(self.timing_frame_column) < 2:
			return default_value
		return self.timing_frame_column[1]
	
	def get_end_frame_count(self, default_value: int) -> int:
		"""
		Returns the frame count at the end of the speed tracking
		or a default value if not available
		"""
		if len(self.speed_timing_frames) < 2:
			return default_value
		return self.speed_timing_frames[1]
	
	def get_tracked_distance_pixels(self) -> Union[int, None]:
		"""
		Returns the distance in pixels between the start and end of the speed tracking
		"""
		if len(self.timing_frame_column) == 0:
			return None
		return self.get_end_column(self.current_box[Car.BOX_EDGE_INDEX]) - self.timing_frame_column[0]

	def is_expired(self, frame_count: int, expiration_count: int) -> bool:
		"""
		Returns true if it expired, false otherwise
		"""
		if self.last_update+expiration_count < frame_count:
			return True
		return False

	def __repr__(self):
		return f"Car(frame={self.speed_timing_frames}, box={self.current_box}, confidence={self.confidence})"