# import time
# from efficientnet_pytorch import EfficientNet
# import numpy as np
# from argparse import Namespace
# from utils.raft import RAFT
# import torch
# import sys
# import cv2
# from utils.utils.utils import InputPadder
# sys.path.append('core')

# args = Namespace(images_dir='test_video', output_dir='test_result')


# class SpeedEstimator:
#     def __init__(self, device):
#         self.model = torch.nn.DataParallel(RAFT(args))
#         self.model.load_state_dict(torch.load(
#             r'models/raft-things.pth', map_location=torch.device(device)))
#         self.model = self.model.module

#         self.model.to(device)
#         self.model.eval()
#         self.device = device

#         self.speed_model = EfficientNet.from_pretrained(
#             f'efficientnet-b0', in_channels=2, num_classes=1)
#         state = torch.load(r"models/b0.pth", map_location=torch.device(device))
#         self.speed_model.load_state_dict(state)
#         self.speed_model.to(device)

#     # def load_image(self, img):
#     #     img = np.array(img).astype(np.uint8)

#     #     # Calculate the center of the image
#     #     center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

#     #     scale = 256
#     #     # Calculate the starting and ending coordinates for the crop
#     #     start_x, end_x = center_x - scale, center_x + scale
#     #     start_y, end_y = center_y - scale, center_y + scale

#     #     # Crop the image to get the 32x32 square from the center
#     #     cropped_img = img[start_y:end_y, start_x:end_x]

#     #     cropped_img = torch.from_numpy(cropped_img).permute(2, 0, 1).float()
#     #     return cropped_img[None].to(self.device)
#     def load_image(self, img):
#         img_resized = np.array(img).astype(np.uint8)

#         # Resize the image to 128x128
#         img_resized = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))

#         img_resized = torch.from_numpy(img_resized).permute(2, 0, 1).float()
#         return img_resized[None].to(self.device)

#     def estimate_speed(self, image1, image2):
#         """
#             Takes as input 2 numpy images and calculate the speed of the car
#         """
#         with torch.no_grad():
#             image1 = self.load_image(image1)
#             image2 = self.load_image(image2)

#             padder = InputPadder(image1.shape)
#             image1, image2 = padder.pad(image1, image2)
#             # s = time.time()
#             _, flow_up = self.model(image1, image2, iters=5, test_mode=True)
#             # print(f'It took {time.time() - s} to calculate the optical flow')
#             # s = time.time()
#             prediction = self.speed_model(flow_up)
#             # print(f'It took {time.time() - s} to calculate the speed')
#             torch.cuda.empty_cache()

#         return prediction

import cv2
import time
import os
import re
import shutil
import math
import numpy as np
import pandas as pd
import tensorflow.keras
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


class SpeedEstimator:
    def __init__(self, features_path, model_path, required_resize, required_x_slice, required_y_slice) -> None:
        # Check if model and video exist
        if not os.path.exists(model_path):
            print(f"model path: '{model_path}' doesn't exist")
            return False
        # Check if input format is .h5 and .mp4
        if not re.search(r'.*\.h5', model_path):
            print(f"please input a h5 file for model")
            return False
        self._get_dataset(read_path=features_path)
        self.model = tensorflow.keras.models.load_model(
            model_path)  # load the model
        self.required_resize = required_resize
        self.require_x_slice = required_x_slice
        self.required_y_slice = required_y_slice

    def _get_dataset(self, read_path):
        if not os.path.exists(read_path):
            print(f"read path: '{read_path}' doesn't exist")

        # Read in csv file and shuffle the data if needed
        np.random.seed(10)
        reader = pd.read_csv(read_path)
        dataset = reader.values
        _, column = dataset.shape
        column -= 1

        # standardlize the data
        x = dataset[:, 0:column]
        self.mean_const = x.mean(axis=0)
        self.std_const = x.std(axis=0)

    def _calculate_optical_mag(self, image1, image2):
        # Check if image have the same size
        if image1.shape != image2.shape:
            print("Image has different size")
            return False

        # Convert image to grayscale to reduce noise
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Calculate the optical flow, the return vector is in Cartesian Coordinates
        flow = cv2.calcOpticalFlowFarneback(
            image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Extract the magnitude of each vector by transforming Cartesian Coordinates to Polar Coordinates
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # normalize the magnitude
        mag_matrix = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        return mag_matrix

    def _slice_matrix(self, mag_matrix, x_slice, y_slice):
        # Calculate the sum of each mag area and return the sqrt of the area sum
        height, width = mag_matrix.shape
        height_seg_len = height // y_slice
        width_seg_len = width // x_slice
        result = []
        for h in range(y_slice):
            for w in range(x_slice):
                mag_area_sum = np.sum(
                    mag_matrix[h * height_seg_len:(h + 1) * height_seg_len, w * width_seg_len:(w + 1) * width_seg_len])
                # round the sqrt to 2
                result.append(round(math.sqrt(mag_area_sum), 2))

        return result

    def speed_detection(self, image1, image2):
        """
            Detect the speed of the automobile using the pretrained model from 'model_path' and input video from 'video', and then output the txt file of the detected speed at 'output_path'.

            Args:
                model_path (str): Path to the pretrained model.
                output_path (str): Path to the output.
                required_resize (int): Resize scale that was used for the pretrained model.
                required_x_slice: x slice that was used for the pretrained model.
                required_y_slice: y slice that was used for the pretrained model.
                MEAN_CONST: Mean of the training set, used to normalize the testing set.
                STD_CONST: Standard Deviation of the training set, used to normalize the testing set.

            Returns:
                tuple: tuple containing:
                    (index (int): index of the last frame that is predicted, detection_time (float): Total time took to predict the speed of the car from the input video).
        """

        height, width, _ = image1.shape
        image1 = cv2.resize(
            image1, (int(width * self.required_resize), int(height * self.required_resize)))

        # read in image2 and resize
        height, width, _ = image2.shape
        image2 = cv2.resize(
            image2, (int(width * self.required_resize), int(height * self.required_resize)))
        # calculate optical flow and slice
        mag_matrix = self._calculate_optical_mag(image1, image2)
        optical_mag_list = self._slice_matrix(
            mag_matrix, self.require_x_slice, self.required_y_slice)

        images_to_predict = np.array(optical_mag_list)
        images_to_predict = images_to_predict.reshape(1, -1)
        images_to_predict -= self.mean_const
        images_to_predict /= self.std_const
        # flatten the matrix so it could be input to the CNN model
        predictions = self.model.predict(x=images_to_predict)[0][0]

        return predictions
