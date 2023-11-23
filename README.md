# Real-time Traffic Violation Detection System

## Introduction
The Real-time Traffic Violation Detection System is an innovative end-to-end solution designed to detect traffic violations in video footage. It excels in accurately detecting vehicle speed and distance, recognizing number plates, and identifying violations of speed or distance limits. The system generates comprehensive reports that detail violations, including plate numbers, types of violations, penalties, specific values (such as speed or distance), and the exact frames where the violation occurred.

## Tools Used for Each Part of the System:
1. **Vehicle Speed and Distance Detection:**
   - Vehicle speeds are estimated using optical flow techniques. The implementation is inspired by [this example](https://car-speed-detection.readthedocs.io/en/latest/Example%20Code.html).
   - Distance estimation is based on the size of the vehicle within the video frame.

2. **Vehicle Tracking:**
   - Vehicle tracking is achieved using the Deep SORT algorithm with PyTorch, supporting YOLO series models. More information can be found [here](https://github.com/xuarehere/yolo_series_deepsort_pytorch).

3. **Plate Recognition:**
   - Utilizing the bounding boxes from the tracking step, the system crops the image and employs a YOLOv8 model to detect plates in the frame.
   - The EasyOCR library is then used to read the numbers on the plates.
   - Once plate numbers are detected, they are linked with the tracked vehicle's ID, ensuring identification even if the plates are not readable in subsequent frames.

4. **Violation Detection:**
   - Violations are predefined before running the program.
   - The system detects violated cars using the estimated speed and distance.
   - The number plates of violated cars, along with the violation details, cost, and frames of the violation, are stored in a CSV file.
   - These frames are later used to generate a video clip of the violation.

## Installation
1. Clone this repository.
2. Install Anaconda by following the guide at [Anaconda Installation](https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html).
3. In your terminal, navigate to the code directory and execute the following command to set up the environment: ```conda env create -f environment.yml```


## Usage
- Place your video file in the "data" folder.
- Run the system using the `main.py` script with the following arguments:
- `--VIDEO_PATH`: Specify the path to your video file.
- `--save_path`: Designate the path to save the output.
- `--distance_violation`: Set the distance limit as an integer.
- `--speed_violation`: Set the speed limit as an integer.
- The output will be a CSV file named `violations.csv` in the output folder.

## Example Command
```python main.py --VIDEO_PATH "./data/video.mp4" --save_path "./output/v/output_video2.mp4" --distance_violation 10 --speed_violation 60```


## Note
The system is optimized for GPU usage. For optimal performance, ensure that PyTorch with GPU support is installed.

---

If you are interested in purchasing this product or would like more information, please feel free to contact me at hamody28522@gmail.com .

