# Imports
import torch
import cv2
import time
import argparse
from video_processing.car_tracker import CarTracker
from video_processing.video_extractor import VideoExtractor
from video_processing.speed_estimator import SpeedEstimator
from video_processing.distance_estimator import DistanceEstimation
from video_processing.plate_extractor import PlateExtractor
from utils.parser import get_config
import pandas as pd
import pytesseract
import numpy as np
import copy


# Set your Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_mmdetection", type=str,
                        default="./configs/mmdet.yaml")
    parser.add_argument("--config_detection", type=str,
                        default="./configs/yolov8.yaml")
    parser.add_argument("--config_deepsort", type=str,
                        default="./configs/deep_sort.yaml")
    parser.add_argument("--config_fastreid", type=str,
                        default="./configs/fastreid.yaml")
    parser.add_argument("--detect_model", type=str, default="yolov8")
    parser.add_argument("--fastreid", action="store_true")
    parser.add_argument("--mmdet", action="store_true")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default=r"./output/v/output_video.mp4")
    parser.add_argument("--cpu", dest="use_cuda",
                        action="store_false", default=True)
    parser.add_argument("--camera", action="store",
                        dest="cam", type=int, default="-1")
    parser.add_argument("--distance_violation", type=int, default=3)
    parser.add_argument("--speed_violation", type=int, default=60)
    return parser.parse_args()


def initialize_video_writer(frame, fps):
    """Initialize the video writer."""
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.save_path,
                          fourcc, fps, (width, height))
    return out


def draw_lanes(frame):
    height, width, _ = frame.shape

    # Define the starting and ending points for the left lane line
    left_lane_start = (width // 3, height)
    left_lane_end = (width // 3, 0)

    # Define the starting and ending points for the right lane line
    right_lane_start = (2 * (width // 3), height)
    right_lane_end = (2 * (width // 3), 0)

    # Draw the lane lines on the frame
    cv2.line(frame, left_lane_start, left_lane_end, (0, 255, 0), 2)
    cv2.line(frame, right_lane_start, right_lane_end, (0, 255, 0), 2)

    return frame


def process_tracked_cars(frame, tracked_cars, relative_speeds, boxes_distance, i, speed):
    width = frame.shape[1]
    left_lane_x = width // 3
    right_lane_x = 2 * (width // 3)
    """Process tracked cars and draw bounding boxes and info on the frame."""
    for j in range(len(tracked_cars[0])):
        bbox_tlwh = tracked_cars[0][j]
        box_id = tracked_cars[1][j]
        x, y, w, h = bbox_tlwh
        distance = distance_estimator.calc_distance(w)
        center_x = x
        if distance < args.distance_violation:
            if left_lane_x < center_x < right_lane_x:
                if box_id in detected_licenses and detected_licenses[box_id]['text'] != "undefind":
                    if distance_violated_cars.get(box_id):
                        distance_violated_cars[detected_licenses[box_id]['text']] = distance_violated_cars.pop(box_id)
                        distance_violated_cars[detected_licenses[box_id]['text']]["Violation value"].append(distance)
                        distance_violated_cars[detected_licenses[box_id]['text']]["Frame of violation"].append(i)
                    elif distance_violated_cars.get(detected_licenses[box_id]['text']):
                        distance_violated_cars[detected_licenses[box_id]['text']]["Violation value"].append(distance)
                        distance_violated_cars[detected_licenses[box_id]['text']]["Frame of violation"].append(i)
                    else:
                        distance_violated_cars[detected_licenses[box_id]['text']] = {
                            "type": "Distance violation",
                            "cost": "50$",
                            "Violation value": [distance],
                            "Frame of violation": [i]
                        }
                else:
                    if distance_violated_cars.get(box_id):
                        distance_violated_cars[box_id]["Violation value"].append(distance)
                        distance_violated_cars[box_id]["Frame of violation"].append(i)
                    else:
                        distance_violated_cars[box_id] = {
                            "type": "Distance violation",
                            "cost": "50$",
                            "Violation value": [distance],
                            "Frame of violation": [i]
                        }
            print(f"Car with id {box_id} has a distance violation")

        # Extract the bounding box of the car
        car_bbox = frame[int(y):int(y+h), int(x):int(x+w)]

        # Pass the bounding box to process_plates function
        # if not detected_licenses.get(box_id):
        #     _, license = process_plates(car_bbox)
        #     if license != 'undefind':
        #         detected_licenses[box_id] = license
        # else:
        #     process_plates(car_bbox, license=detected_licenses[box_id])
        process_plates(car_bbox, box_id)
        # Calculate relative speeds
        if not relative_speeds.get(box_id):
            relative_speeds[box_id] = []
        if i % relative_speed_calc_interval:
            if not boxes_distance.get(box_id):
                boxes_distance[box_id] = distance
            rv = ((distance - boxes_distance[box_id]) / (relative_speed_calc_interval / fps)) * 3.6
            relative_speeds[box_id].append(rv)
            boxes_distance[box_id] = distance

            if speed + rv >= args.speed_violation:
                if box_id in detected_licenses  and detected_licenses[box_id]['text'] != "undefind":
                    if speed_violated_cars.get(box_id):
                        speed_violated_cars[detected_licenses[box_id]['text']] = speed_violated_cars.pop(box_id)
                        speed_violated_cars[detected_licenses[box_id]['text']]["Violation value"].append(distance)
                        speed_violated_cars[detected_licenses[box_id]['text']]["Frame of violation"].append(i)
                    elif speed_violated_cars.get(detected_licenses[box_id]['text']):
                        speed_violated_cars[detected_licenses[box_id]['text']]["Violation value"].append(speed+rv)
                        speed_violated_cars[detected_licenses[box_id]['text']]["Frame of violation"].append(i)

                    else:
                        speed_violated_cars[detected_licenses[box_id]['text']] = {
                            "type": "Speed violation",
                            "cost": "100$",
                            "Violation value": [speed+rv],
                            "Frame of violation": [i]
                        }
                else:
                    if speed_violated_cars.get(box_id):
                        speed_violated_cars[box_id]["Violation value"].append(speed+rv)
                        speed_violated_cars[box_id]["Frame of violation"].append(i)
                    else:
                        speed_violated_cars[box_id] = {
                            "type": "Speed violation",
                            "cost": "100$",
                            "Violation value": [speed+rv],
                            "Frame of violation": [i]
                        }

        # Draw bounding box and info
        cv2.rectangle(frame, (int(x), int(y)),
                      (int(x+w), int(y+h)), (0, 0, 255), 2)
        box_i_rv = sum(relative_speeds[box_id]) / \
            (len(relative_speeds[box_id]) + 0.0001)
        label = f"{box_id}, D: {distance:.2f} m, RV: {box_i_rv:.2f}"
        cv2.putText(frame, label, (int(x), int(y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if i % 120 == 0:
            relative_speeds = {}
    return frame


def process_plates(frame, box_id, license=None):
    # cv2.imshow('Frame', frame)
    # cv2.waitKey(0)
    plates = plates_extractor.process_frame(frame)
    plate_text = 'undefind'
    for plate in plates:
        x_min, y_min, x_max, y_max = plate['bbox']
        # Assuming you have 'text' key in the plate dictionary
        if not detected_licenses.get(box_id):
            plate_text = plate['text']
            detected_licenses[box_id] = {
                "text": plate_text, "score": plate['score']}
        else:
            if detected_licenses[box_id]["score"] < plate['score']:
                plate_text = plate['text']
                detected_licenses[box_id] = {
                    "text": plate_text, "score": plate['score']}
            else:
                plate_text = detected_licenses[box_id]["text"]

        # Draw the rectangle on the frame
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                      (0, 255, 0), 2)  # Green bounding box

        # Draw the text above the rectangle
        cv2.putText(frame, plate_text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return frame, plate_text


# Initialization
start = time.time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

args = parse_args()
cfg = get_config()
video_path = args.VIDEO_PATH
video_extractor = VideoExtractor(video_path)
distance_violated_cars = {}
speed_violated_cars = {}

cfg.merge_from_file(args.config_detection)
cfg.USE_MMDET = False
cfg.DETECT_MODEL = args.detect_model
cfg.merge_from_file(args.config_deepsort)
cfg.USE_FASTREID = False
car_tracker = CarTracker(cfg, args)

plates_model = r'./models/platesYolov8n.pt'

features_path = r'./data/feature_map3_combine.txt'
model_path = r'./models/Model.h5'
required_resize = 0.5
required_x_slice = 8
required_y_slice = 6
speed_estimator = SpeedEstimator(
    features_path, model_path, required_resize, required_x_slice, required_y_slice)

distance_estimator = DistanceEstimation()
plates_extractor = PlateExtractor(plates_model)

detected_licenses = {}

# Variables for processing
speed = torch.tensor([0])
speeds = []
boxes_distance = {}
speed_calc_interval = 1
relative_speed_calc_interval = 10
fps = 30
relative_speeds = {}
i = 0
previous_frame = None
out_frames = []

for frame in video_extractor.extract_frame():
    # Check for 's' key press to skip 1 second
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        for _ in range(fps):
            next(video_extractor.extract_frame())
        continue
    if previous_frame is not None:
        if i % speed_calc_interval == 0:
            speed = speed_estimator.speed_detection(previous_frame, frame)
            # speed = torch.tensor([0])
            speeds.append(speed)
            speed = sum(speeds) / len(speeds)
        previous_frame = copy.deepcopy(frame)
    else:
        previous_frame = copy.deepcopy(frame)

    tracked_cars = car_tracker.process_frame(frame)
    if tracked_cars:
        frame = process_tracked_cars(
            frame, tracked_cars, relative_speeds, boxes_distance, i, speed)
        # frame = process_plates(frame)

    if i % 150 == 0:
        speeds = []

    # Draw speed info and display frame

    cv2.putText(frame, f"Speed: {speed.item():.2f} km/h",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    frame = draw_lanes(frame)

    frame = cv2.resize(frame, (1280, 640))
    # if i == 0:
    #     out = initialize_video_writer(frame, fps)
    # out.write(frame)
    out_frames.append(frame)
    cv2.imshow('Frame', frame)
    cv2.waitKey(1)
    i += 1

# out.release()
cv2.destroyAllWindows()

out = initialize_video_writer(out_frames[0], fps=video_extractor.fps)
for frame in out_frames:
    out.write(frame)

out.release()

for key in list(speed_violated_cars.keys()):
    if len(speed_violated_cars[key]["Frame of violation"]) < 20:
        speed_violated_cars.pop(key)

plate_numbers = list(distance_violated_cars.keys()) + list(speed_violated_cars.keys())
types = [v["type"] for v in distance_violated_cars.values()] + [v["type"] for v in speed_violated_cars.values()]
costs = [v["cost"] for v in distance_violated_cars.values()] + [v["cost"] for v in speed_violated_cars.values()]
violation_values = [v["Violation value"] for v in distance_violated_cars.values()] + [v["Violation value"][1:] for v in speed_violated_cars.values()]
frames_of_violation = [v["Frame of violation"] for v in distance_violated_cars.values()] + [v["Frame of violation"][1:] for v in speed_violated_cars.values()]

# Create a DataFrame
final = {
    "Plate numbers": plate_numbers,
    "Type": types,
    "Cost": costs,
    "Violation value": violation_values,
    "Frame of violation": frames_of_violation,
}

df = pd.DataFrame(final)
df.to_csv('./output/violations.csv', index=False)
print(df)
print(f'Time taken = {time.time() - start:.2f}')  # Corrected the time format here as well
