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

sys.path.append(os.path.join(
    os.path.dirname(__file__), 'thirdparty/fast-reid'))
sys.path.append(os.path.join(os.path.dirname(
    __file__), 'thirdparty/mmdetection'))


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        # self.track_class
        # yolov3,   person_id = 0; car_id = 2,  选择人cls_ids=0，作为跟踪; car, cls_ids=2, 具体见 coco.name;
        # -1  所有目标都进行跟踪
        self.track_class = cfg.DEEPSORT.TRACK_CLASS
        self.logger = get_logger("root")
        self.reid_feature_dic = {}
        self.now_frame = 0
        # self.cam_id =  video_path.split(os.sep)[-2]     # video_path = /workspace/dataset/aic22-mcmt/S06/c112/vdo.avi

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn(
                "Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def reid_feature_ele_init(self, reid_dic={}):
        # frame_id = 1, track_id=5):
        """
        key_names = "img" + str(frame_id).zfill(6) + "_" + str(track_id).zfill(3)
        unit = {
            'img000011_020':{
            'bbox': (1056, 448, 1262, 590), 
            'frame': 'img000011', 
            'id': 20, 
            'imgname': 'img000011_020.png', 
            'class': 2, 
            'conf': 0.8876953125, 
            'feat': array([ 0.55748284, ...e=float32)   }
            }
        """
        frame = "img" + str(reid_dic["frame_id"]).zfill(6)
        key_names = frame + "_" + str(reid_dic["track_id"]).zfill(3)
        unit = {
            'bbox': reid_dic["bbox"],
            'frame': frame,
            'id': reid_dic["track_id"],
            'imgname': key_names + '.png',
            'class': reid_dic["class"],
            'conf': reid_dic["conf"],
            'feat': reid_dic["feat"]}
        return key_names, unit

    def update_reid_feature_dic(self, frame=0, detections=None):
        out_dict = {}
        for idx, item in enumerate(detections):
            idx_frame = "img" + str(frame).zfill(6)
            key_names = idx_frame + "_" + str(idx).zfill(3)
            out_dict[key_names] = {
                'bbox': item.x1y1x2y2,
                'frame': idx_frame,
                'id': str(idx),
                'imgname': key_names + ".png",
                'class': self.track_class,
                'conf': item.confidence,
                'feat': item.feature
            }
        return out_dict

    def save_dic_to_pkl(self, dic={}, output_path="./", cam="c112"):
        feat_pkl_file = os.path.join(
            self.args.save_path, f'{cam}_dets_feat.pkl')
        pickle.dump(dic, open(feat_pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        print('save pickle in %s' % feat_pkl_file)

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(
                self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(
                self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(
                self.save_video_path, fourcc, 15, (self.im_width, self.im_height))        # defualt, fps=20

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        while self.vdo.grab():
            self.now_frame = idx_frame
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # 画框到一张图片上，并保存
            # imga = draw_boxes(im, bbox_xywh, cls_ids,  cls_conf , class_name_map=self.detector.class_names)
            # cv2.imwrite("./1.png", imga[:, :, (2, 1, 0)])

            if self.track_class == -1:
                # track all id
                mask = cls_ids == cls_ids
            else:
                # t rack specify id
                mask = (cls_ids == self.track_class[0]) | (
                    cls_ids == self.track_class[1]) | (
                    cls_ids == self.track_class[2])

            bbox_xywh = bbox_xywh[mask]

            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            bbox_xywh[:, 3:] *= 1.2     #
            cls_conf = cls_conf[mask]

            # do tracking
            outputs, detections = self.deepsort.update(bbox_xywh, cls_conf, im)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy,
                                    identities)      # 画框到原始图片上

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}"
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))
            # reid feature
            if len(detections) > 0:
                out_dic = self.update_reid_feature_dic(
                    frame=self.now_frame, detections=detections)
                self.reid_feature_dic.update(out_dic)

        self.save_dic_to_pkl(dic=self.reid_feature_dic,
                             output_path=self.args.save_path, cam=1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_mmdetection", type=str,
                        default="./configs/mmdet.yaml")
    parser.add_argument("--config_detection", type=str,
                        default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str,
                        default="./configs/deep_sort.yaml")
    parser.add_argument("--config_fastreid", type=str,
                        default="./configs/fastreid.yaml")
    parser.add_argument("--detect_model", type=str, default="yolov3")
    parser.add_argument("--fastreid", action="store_true")
    parser.add_argument("--mmdet", action="store_true")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda",
                        action="store_false", default=True)
    parser.add_argument("--camera", action="store",
                        dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    print(args)
    if args.mmdet:
        cfg.merge_from_file(args.config_mmdetection)
        cfg.USE_MMDET = True
    else:
        cfg.merge_from_file(args.config_detection)
        cfg.USE_MMDET = False
        cfg.DETECT_MODEL = args.detect_model
    cfg.merge_from_file(args.config_deepsort)
    if args.fastreid:
        cfg.merge_from_file(args.config_fastreid)
        cfg.USE_FASTREID = True
    else:
        cfg.USE_FASTREID = False

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
