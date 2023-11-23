import cv2
import torch
import warnings
import numpy as np
import torch
from ultralytics import YOLO
import easyocr
import string
import pytesseract  # Import pytesseract


class PlateExtractor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.dict_char_to_int = {'O': '0',
                                 'I': '1',
                                 'J': '3',
                                 'A': '4',
                                 'G': '6',
                                 'S': '5',
                                 'B': '8',
                                 'Z': '2',
                                 'Q': '0',
                                 'D': '0',
                                 'T': '7',
                                 'U': '0',
                                 'Y': '7',
                                 'P': '0',
                                 'F': '5',
                                 'L': '4',
                                 'E': 6}

        self.dict_int_to_char = {'0': 'O',
                                 '1': 'I',
                                 '3': 'J',
                                 '4': 'A',
                                 '6': 'G',
                                 '5': 'S',
                                 '8': 'B',
                                 '2': 'Z',
                                 '7': 'T',
                                 '9': 'P'}

    def license_complies_format(self, text):
        """
        Check if the license plate text complies with the required format.

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """
        # if len(text) == 8:
        #     print(text)
        #     return True

        if len(text) != 7:
            return False

        if (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in self.dict_char_to_int.keys()) and \
            (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in self.dict_char_to_int.keys()) and \
            (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in self.dict_char_to_int.keys()) and \
            (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in self.dict_char_to_int.keys()) and \
            (text[4] in string.ascii_uppercase or text[4] in self.dict_int_to_char.keys()) and \
            (text[5] in string.ascii_uppercase or text[5] in self.dict_int_to_char.keys()) and \
                (text[6] in string.ascii_uppercase or text[6] in self.dict_int_to_char.keys()):
            return True
        else:
            return False

    def format_license(self, text):
        """
        Format the license plate text by converting characters using the mapping dictionaries.

        Args:
            text (str): License plate text.

        Returns:
            str: Formatted license plate text.
        """
        # if len(text) == 8:
        #     return text
        license_plate_ = ''
        mapping = {0: self.dict_char_to_int, 1: self.dict_char_to_int, 4: self.dict_int_to_char, 5: self.dict_int_to_char, 6: self.dict_int_to_char,
                   2: self.dict_char_to_int, 3: self.dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

        return license_plate_

    def process_frame(self, frame):
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self.model(im)
        result = []
        for obj in output:
            for box in obj.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = map(int, xyxy)

                # Extract the region of interest (ROI) from the frame
                roi = frame[y_min:y_max, x_min:x_max]
                # cv2.imshow('plate', roi)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Use easyocr to extract text from the ROI
                text_results = self.reader.readtext(roi)
                if len(text_results) > 0:
                    # print(text_results)
                    # cv2.imshow('plate', roi)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    text = ""
                    score = 0
                    for detection in text_results:
                        text += detection[1]
                        score += detection[2]
                    text = text.upper().replace(' ', '')
                    if self.license_complies_format(text):
                        result.append({
                            'bbox': [x_min, y_min, x_max, y_max],
                            'text': self.format_license(text),
                            'score': score
                        })
                    else:
                        result.append({
                            'bbox': [x_min, y_min, x_max, y_max],
                            'text': "undefind",
                            'score': 0
                        })
                else:
                    result.append({
                        'bbox': [x_min, y_min, x_max, y_max],
                        'text': "undefind",
                        'score': 0
                    })
        return result
