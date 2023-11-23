# from efficientnet_pytorch import EfficientNet
# import numpy as np
# from argparse import Namespace
# from utils.raft import RAFT
# import torch
# import sys
# import cv2
# from utils.utils.utils import InputPadder
# import onnxruntime as ort
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

#         providers = ort.get_available_providers()
#         self.session = ort.InferenceSession(
#             r"models/raft.onnx", providers=['CUDAExecutionProvider'])
#         self.input_names = [
#             input_meta.name for input_meta in self.session.get_inputs()]
#         # Get the model's output names
#         self.output_names = [
#             output_meta.name for output_meta in self.session.get_outputs()]

#     def load_image(self, img):
#         img = np.array(img).astype(np.uint8)

#         scale = 512
#         # Resize the image to 128x128
#         img_resized = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))

#         img_resized = torch.from_numpy(img_resized).permute(2, 0, 1).float()
#         return img_resized[None].to(self.device)

#     def estimate_speed(self, image1, image2):
#         """
#             Takes as input 2 numpy images and calculate the speed of the car
#         """
#         image1 = self.load_image(image1)
#         image2 = self.load_image(image2)

#         padder = InputPadder(image1.shape)
#         image1, image2 = padder.pad(image1, image2)

#         input_feed = {self.input_names[0]: image1.cpu().numpy(),
#                       self.input_names[1]: image2.cpu().numpy()}

#         prediction = self.session.run(self.output_names, input_feed)

#         prediction = self.speed_model(
#             torch.tensor(prediction[-1]).to(self.device))

#         return prediction