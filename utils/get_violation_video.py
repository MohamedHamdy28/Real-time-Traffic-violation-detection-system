import cv2
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_video", type=str, default=r"./output/v/output_video.mp4")
    parser.add_argument("--violations_csv_path", type=str, default=r"./output/violations.csv")
    parser.add_argument("--car_id", type=str)
    parser.add_argument("--outputs_path", type=str, default=r"./output/violations videos")
    return parser.parse_args()

def extract_frames_and_create_video(input_video_path, output_video_path, start_frame_index, end_frame_index):
    # Open the input video file.
    cap = cv2.VideoCapture(input_video_path)
    # Get the frame rate and frame size of the input video.
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Create a list to store the extracted frames.
    frames = []

    # Skip to the start frame index.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    
    i = start_frame_index
    # Extract the frames between the starting and ending frame indices.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        if i >= end_frame_index:
            break
        i+=1
    # Create the output video writer object.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_height, frame_width, layers = frames[0].shape
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Write the extracted frames to the output video.
    for frame in frames:
        out.write(frame)
        
    # Release the input video capture object and the output video writer object.
    cap.release()
    out.release()


args = parse_args()
violations_df = pd.read_csv(args.violations_csv_path)

rows = []
for row in violations_df.iterrows():
    if str(row[1]["Plate numbers"]) == args.car_id: # plate numbers 
        rows.append(row[1])

if len(rows) == 0:
    print("This car_id did not commit any violation")
else:
    for row in rows:
        violation_type = row['Type']
        frames = row["Frame of violation"]
        out_video_path = args.outputs_path + rf"/{args.car_id}_{violation_type}.mp4"
        start_frame = ''
        for i in range(1, len(frames)):
            if frames[i] == ',':
                break
            start_frame += frames[i]
        end_frame = ''
        for i in reversed(frames):
            if i == ',':
                break
            if i == ']':
                continue
            end_frame += i
        end_frame = end_frame[::-1]
        extract_frames_and_create_video(args.original_video, out_video_path, int(start_frame), int(end_frame))



