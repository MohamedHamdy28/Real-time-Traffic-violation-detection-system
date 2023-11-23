# import numpy as np
# import onnxruntime as ort

# # Check if GPU is available
# providers = ort.get_available_providers()
# if 'CUDAExecutionProvider' not in providers:
#     raise ValueError(
#         "ONNX Runtime GPU is not available. Ensure you've installed the GPU version.")

# # Load the ONNX model
# session = ort.InferenceSession(
#     r"models/raft.onnx", providers=['CUDAExecutionProvider'])

# # Get all input names
# input_names = [input_meta.name for input_meta in session.get_inputs()]

# # Create some dummy data for inference (ensure the shape matches your model's input shape)
# dummy_input1 = np.random.randn(1, 3, 184, 320).astype(np.float32)
# dummy_input2 = np.random.randn(1, 3, 184, 320).astype(np.float32)

# # Create a dictionary with input names and their corresponding dummy inputs
# input_feed = {input_names[0]: dummy_input1, input_names[1]: dummy_input2}

# # Get all the model's output names
# output_names = [output_meta.name for output_meta in session.get_outputs()]

# # Perform inference
# predictions = session.run(output_names, input_feed)

# # Print the predictions
# for pred in predictions:
#     print(pred)

# print(input_feed.keys())


# from video_processing.car_tracker import CarTracker
# from video_processing.video_extractor import VideoExtractor
# from video_processing.speed_estimator import SpeedEstimator
# from video_processing.distance_estimator import DistanceEstimation
# from utils.parser import get_config
# import torch
# import cv2
# import time
# import argparse
# from video_processing.plate_extractor import PlateExtractor

# model_path = r'./models/platesYolov8n.pt'
# plate_extractor = PlateExtractor(model_path)


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("VIDEO_PATH", type=str)
#     parser.add_argument("--config_mmdetection", type=str,
#                         default="./configs/mmdet.yaml")
#     parser.add_argument("--config_detection", type=str,
#                         default="./configs/yolov3.yaml")
#     parser.add_argument("--config_deepsort", type=str,
#                         default="./configs/deep_sort.yaml")
#     parser.add_argument("--config_fastreid", type=str,
#                         default="./configs/fastreid.yaml")
#     parser.add_argument("--detect_model", type=str, default="yolov3")
#     parser.add_argument("--fastreid", action="store_true")
#     parser.add_argument("--mmdet", action="store_true")
#     # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
#     parser.add_argument("--display", action="store_true")
#     parser.add_argument("--frame_interval", type=int, default=1)
#     parser.add_argument("--display_width", type=int, default=800)
#     parser.add_argument("--display_height", type=int, default=600)
#     parser.add_argument("--save_path", type=str, default="./output/")
#     parser.add_argument("--cpu", dest="use_cuda",
#                         action="store_false", default=True)
#     parser.add_argument("--camera", action="store",
#                         dest="cam", type=int, default="-1")
#     return parser.parse_args()


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using {device}")
# video_path = r'./data/v.mp4'
# # Add the path to your txt file here
# results_path = r'./data/results_v.txt'

# video_extractor = VideoExtractor(video_path)

# for frame in video_extractor.extract_frame():
#     output = plate_extractor.process_frame(frame)
#     for obj in output:
#         for box in obj.boxes:
#             xyxy = box.xyxy[0].cpu().numpy()

#             # Extract coordinates from xyxy
#             x_min, y_min, x_max, y_max = map(int, xyxy)

#             # Draw the rectangle on the frame
#             cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
#                           (0, 255, 0), 2)  # Green bounding box

#             print(xyxy)

#     # Display the frame with bounding boxes
#     cv2.imshow('Detected Objects', frame)
#     cv2.waitKey(1)  # Wait until a key is pressed
# cv2.destroyAllWindows()

# import cv2
# import numpy as np


# def moving_average(values, window):
#     """Compute the moving average of values over a window."""
#     weights = np.repeat(1.0, window) / window
#     return np.convolve(values, weights, 'valid')


# def roi_selection(event, x, y, flags, param):
#     global roi_points
#     if event == cv2.EVENT_LBUTTONDOWN:
#         roi_points.append((x, y))
#         cv2.circle(roi_frame, (x, y), 5, (0, 0, 255), -1)
#         if len(roi_points) == 2:
#             cv2.rectangle(
#                 roi_frame, roi_points[0], roi_points[1], (0, 255, 0), 2)
#             cv2.imshow('Define ROI', roi_frame)
#             cv2.waitKey(500)
#             cv2.destroyAllWindows()


# roi_points = []
# cap = cv2.VideoCapture('./data/v1.mp4')
# ret, old_frame = cap.read()

# # Resize the frame
# old_frame = cv2.resize(old_frame, (1280, 640))

# # User defines the ROI
# roi_frame = old_frame.copy()
# cv2.imshow('Define ROI', roi_frame)
# cv2.setMouseCallback('Define ROI', roi_selection)
# cv2.waitKey(0)

# # Extract ROI coordinates
# x1, y1 = roi_points[0]
# x2, y2 = roi_points[1]
# # Ensure coordinates are in correct order
# x1, x2 = min(x1, x2), max(x1, x2)
# y1, y2 = min(y1, y2), max(y1, y2)

# # Check that the coordinates are valid
# if x1 >= x2 or y1 >= y2 or x2 > old_frame.shape[1] or y2 > old_frame.shape[0]:
#     print("Invalid ROI coordinates!")
#     exit()
# old_gray = cv2.cvtColor(old_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
# old_gray = cv2.GaussianBlur(old_gray, (5, 5), 0)

# REFERENCE_DISTANCE = 10
# DISTANCE_UNIT = 'feet'
# TIME_UNIT = 'second'
# CONVERSION_TO_MPH = 0.6818
# frame_rate = cap.get(cv2.CAP_PROP_FPS)

# speeds = []

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Resize the frame
#     frame = cv2.resize(frame, (1280, 640))

#     frame_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
#     frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

#     flow = cv2.calcOpticalFlowFarneback(
#         old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

#     median_flow = np.median(mag)

#     if median_flow > 1.0:
#         speed = (median_flow * REFERENCE_DISTANCE *
#                  frame_rate) * CONVERSION_TO_MPH
#         speeds.append(speed)

#         if len(speeds) > 10:
#             avg_speed = moving_average(np.array(speeds), 10)[-1]
#             cv2.putText(frame, f"Speed: {avg_speed:.2f} mph", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, f"Speed: {speed:.2f} mph", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow('Dashcam Speed', frame)

#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

#     old_gray = frame_gray.copy()

# cap.release()
# cv2.destroyAllWindows()

d = {
    "a": [1, 2],
    "B": [2, 3]
}
c = {
    "a": [1, 2],
    "B": [2, 3]
}

import pandas as pd

final = {
    "ids": list(d.keys()) + list(c.keys()),
    "numbers": [v for v in d.values()] + [v for v in c.values()]
}
df = pd.DataFrame(final)
print(df)
