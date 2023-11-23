import pickle
from utils.io import write_results
from utils.log import get_logger
from utils.parser import get_config
from utils.draw import draw_boxes
from deep_sort import build_tracker
from detector import build_detector
import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import sys
import torch


class CarTracker:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.track_class = cfg.DEEPSORT.TRACK_CLASS
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn(
                "Running in cpu mode which maybe very slow!", UserWarning)
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def process_frame(self, frame):
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox_xywh, cls_conf, cls_ids = self.detector(im)
        if self.track_class == -1:
            mask = cls_ids == cls_ids
        else:
            mask = (cls_ids == self.track_class[0]) | (
                cls_ids == self.track_class[1]) | (cls_ids == self.track_class[2])
        bbox_xywh = bbox_xywh[mask]
        # bbox_xywh[:, 3:] *= 1.2
        cls_conf = cls_conf[mask]
        outputs, _ = self.deepsort.update(bbox_xywh, cls_conf, im)
        if len(outputs) > 0:
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            for bb_xyxy in bbox_xyxy:
                bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
            return bbox_tlwh, identities
        return None
