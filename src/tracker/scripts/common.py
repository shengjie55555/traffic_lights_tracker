#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 2021/7/25 下午5:01
# @Author :
# @File : Tracker.py
# @Software: CLion

from os import name

from numpy.core.fromnumeric import argmin, mean
from numpy.core.numeric import ones
import rospy
from tracker.msg import traffic_lights_num, traffic_lights_state

import time
import cv2
import torch
import numpy as np
import copy

from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, plot_one_box, set_logging)
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


class Traffic_Light_Filter:
    """
    接收消息，确定前方红绿灯个数，按照tracker的输出从左到右依次赋值给results
    """
    def __init__(self, num=3, maxsize=5, init=False, init_data=2):
        self.results = np.zeros((num, maxsize + 3), np.int)
        self.maxsize = maxsize
        self.num = num
        self.init_data = init_data
        if init:
            self.results[:, 3:] = init_data
        print("Initialize Traffic Light Filter")

    def append(self, num, tracker_out):
        # 如果交通灯个数变化，重新初始化
        if self.results.shape[0] != num:
            self.results = np.zeros((num, self.maxsize + 3), np.int) * 2
            self.results[:, 3:] = self.init_data
            print("Re-Initialize Traffic Light Filter")

        # 从上到下进行筛选，保证得到感兴趣的交通灯检测结果
        tracker_out = tracker_out[np.argsort(tracker_out[:, 1])]
        y0 = 400
        mask = tracker_out[:, 1] < y0
        while(y0 <= 1280):
            if np.sum(mask) > 0:
                pos = np.mean(tracker_out[mask], axis=0)
                mask = tracker_out[:, 1] < (pos[1] + 10)
                break
            y0 += 10
            mask = tracker_out[:, 1] < y0
        tracker_out = tracker_out[mask]
        print("selected tracker out")
        print(tracker_out)
        
        # tracker输出大等于于num时，采用相对位置
        # tracker输出小与num时，采用id
        if tracker_out.shape[0] >= num:
            print("match by position")
            self.match_by_position(num, tracker_out)
        else:
            print("match by id")
            self.match_by_id(num, tracker_out)
        
    def match_by_position(self, num, tracker_out):
        # 选择概率最大的num个跟踪结果
        tracker_out = tracker_out[np.argsort(-tracker_out[:, 5])[:num]]
        # 从左到右排序
        tracker_out = tracker_out[np.argsort(tracker_out[:, 0])]
        for i in range(tracker_out.shape[0]):
            self.results[i, 0] = tracker_out[i, 4]         # id
            self.results[i, 1] = tracker_out[i, 0]         # x1
            self.results[i, 2] = tracker_out[i, 1]         # y1
            self.results[i, 3:-1] = self.results[i, 4:]
            self.results[i, -1] = tracker_out[i, 6]        # cls
    
    def match_by_id(self, num, tracker_out):
        r_idx = [_ for _ in range(num)]
        t_idx = [_ for _ in range(tracker_out.shape[0])]
        # 从左到右排序
        tracker_out = tracker_out[np.argsort(tracker_out[:, 0])]
        # 优先id匹配
        for i in range(tracker_out.shape[0]):
            tracker_id = tracker_out[i, 4]
            result_idx = np.squeeze(np.argwhere(self.results[:, 0] == tracker_id))
            if result_idx.size > 0:
                self.results[result_idx, 0] = tracker_id
                self.results[result_idx, 1] = tracker_out[i, 0]
                self.results[result_idx, 2] = tracker_out[i, 1]
                self.results[result_idx, 3:-1] = self.results[result_idx, 4:]
                self.results[result_idx, -1] = tracker_out[i, 6]
                r_idx.remove(result_idx)
                t_idx.remove(i)
        # 对于首次出现的tracker，利用位置匹配
        for i in t_idx:
            result_idx = np.argmin(np.abs(tracker_out[i, 0] - self.results[r_idx, 1]))
            self.results[r_idx[result_idx], 0] = tracker_out[i, 4]
            self.results[r_idx[result_idx], 1] = tracker_out[i, 0]
            self.results[r_idx[result_idx], 2] = tracker_out[i, 1]
            self.results[r_idx[result_idx], 3:-1] = self.results[r_idx[result_idx], 4:]
            self.results[r_idx[result_idx], -1] = tracker_out[i, 6]
            r_idx.pop(result_idx)


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def draw_boxes(img, bbox, identities, scores, cls, names, colors, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    tf = max(tl - 1, 1)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = colors[cls[i]]
        label = "%d" % id
        t_size = cv2.getTextSize(label, cv2.LINE_AA, fontScale=tl / 3, thickness=tf)[0]
        cv2.rectangle(img, (x1 - t_size[0] - 5, y1 - t_size[1] - 3), (x1, y1), color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x1 - t_size[0] - 5, y1), 0, tl/3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        label = '{:.2f}{}{:s}'.format(scores[i], " ", names[cls[i]])
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 - t_size[1] - 4), color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x1, y1), 0, tl/3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def draw_results(img, results, final_results, colors, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    tf = max(tl - 1, 1)
    for i in range(len(results)):
        id, x1, y1 = [int(i) for i in results[i, :3]]
        color = colors[final_results[i]]
        label = "%d" % id
        t_size = cv2.getTextSize(label, cv2.LINE_AA, fontScale=tl / 3, thickness=tf)[0]
        cv2.rectangle(img, (x1, 0), (x1 + t_size[0], t_size[1] + 3), color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x1, t_size[1] + 3), 0, tl/3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def get_patch(img0, center, size, scale0, scale1, padded=False):
    h, w = size
    # patch0
    patch0 = copy.deepcopy(img0)
    if padded:
        start0, end0 = (center - np.array([h * scale0, w * scale0])).astype(np.int), \
                       (center + np.array([h * scale0, w * scale0])).astype(np.int)
        patch0[start0[0]: end0[0], start0[1]: end0[1], :] = 114
    # patch1
    start1, end1 = (center - np.array([h * scale1, w * scale1])).astype(np.int), \
                   (center + np.array([h * scale1, w * scale1])).astype(np.int)
    ratio_h, ratio_w = (end1[0] - start1[0]) / h, (end1[1] - start1[1]) / w
    patch1 = img0[start1[0]: end1[0], start1[1]: end1[1], :]
    return (patch0, patch1), (ratio_h, ratio_w), start1


def transform(patch, args):
    patch0, patch1 = patch
    patch0 = letterbox(patch0, new_shape=args["imgsz"])[0]
    patch1 = letterbox(patch1, new_shape=args["imgsz"])[0]
    h, w = patch0.shape[0], patch0.shape[1]
    patch = np.concatenate((np.expand_dims(patch0, axis=0), np.expand_dims(patch1, axis=0)), axis=0)

    patch = patch[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to 3x416x416
    patch = np.ascontiguousarray(patch)
    return patch[-1], (h, w)


def detect_and_track_8(data, args):
    with torch.no_grad():
        # Run inference
        t0 = time.time()

        # Read img from sensor_msg
        img0 = data

        if args["patch_mode"]:
            # Obtain patch
            center = np.array([1200 / 2, 1920 / 2])
            h, w = img0.shape[0], img0.shape[1]
            scale0, scale1 = 1 / 8, 1 / 6
            (patch0, patch1), (ratio_h, ratio_w), start1 = get_patch(img0, center, (h, w), scale0, scale1)

            # Padded resize
            patch, (new_h, new_w) = transform((patch0, patch1), args)
        else:
            patch = letterbox(img0, new_shape=args["imgsz"])[0]
            patch = patch[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            patch = np.ascontiguousarray(patch)

        patch = torch.from_numpy(patch).to(args["device"])
        patch = patch.half() if args["half"] else patch.float()  # uint8 to fp16/32
        patch /= 255.0  # 0 - 255 to 0.0 - 1.0
        if patch.ndimension() == 3:
            patch = patch.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = args["model"](patch, augment=args["augment"])[0]

        if args["patch_mode"]:
            # Transform to img coordinate and concatenate
            pred[-1][:, 2] *= ratio_w
            pred[-1][:, 3] *= ratio_h
            pred[-1][:, 0] = pred[-1][:, 0] * ratio_w + start1[1] * new_w / w
            pred[-1][:, 1] = pred[-1][:, 1] * ratio_h + start1[0] * new_h / h
            mask = torch.max(pred[-1, :, 5:], dim=1)[1] <= 1
            pred[-1, mask, :] = 0
            pred = pred.view(-1, 5+len(args["names"])).unsqueeze(0)

        # Apply NMS
        pred = non_max_suppression(pred, args["conf_thres"], args["iou_thres"],
                                   classes=args["classes"], agnostic=args["agnostic_nms"])
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = '%gx%g ' % patch.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(patch.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, args["names"][int(c)])  # add to string

                # Deepsort tracking
                xywh_bboxs, confs, classes = [], [], []
                for *xyxy, conf, cls in det:
                    # Draw detection result
                    # if args["view_img"]:  # Add bbox to image
                    #     label = '%s %.2f' % (args["names"][int(cls)], conf)
                    #     plot_one_box(xyxy, img0, label=label, color=args["colors"][int(cls)], line_thickness=1)
                    # To deep sort format
                    if 2 <= cls <= 3:
                        x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                        xywh_bboxs.append(xywh_obj)
                        confs.append([conf.item()])
                        classes.append([cls.item()])
                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                classess = torch.Tensor(classes)
                outputs = []
                if xywhs.size()[0] > 0:
                    outputs = args['deepsort_model'].update(xywhs, confss, classess, img0)
                if len(outputs):
                    print("----------")
                    print("original tracker out")
                    print(outputs)
                    args['filter'].append(args["lights_num"], outputs)
                    final_results = [2 if np.mean(result[3:]) <= 2.5 else 3 for result in args['filter'].results]
                    print(final_results)
                    for i in range(len(final_results)):
                        args["state_msg"].state.append(final_results[i])
                    args["pub"].publish(args["state_msg"])
                    args["state_msg"].state = []
                    # ------------------------------- visualization ------------------------------ #
                    # bbox_xyxy = outputs[:, :4]
                    # identities = outputs[:, 4]
                    # scores = outputs[:, 5] / 100
                    # cls = outputs[:, 6]
                    # draw_boxes(img0, bbox_xyxy, identities, scores, cls, args["names"], args['colors'], 3)
                    print(args["filter"].results)
                    draw_results(img0, args["filter"].results, final_results, args["colors"], 3)

                # Print time (inference + NMS)
                # print('Detection And Tracking Done. %s (%.3fs)' % (s, t2 - t1))

        # Stream results
        if args["view_img"]:
            img0_show = cv2.resize(img0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.namedWindow("press q to quit", 0)
            cv2.imshow("press q to quit", img0_show)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
        if args['save_img']:
            args['out_video'].write(img0)

        # print('Visualization Done. (%.3fs)' % (time.time() - t0))


def load_param_8():
    opt = {
        "weights": "/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/tracker/scripts/runs/exp1/weights/best.pt",
        "deep_sort_weights": "/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/tracker/scripts/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7",
        "config_deepsort": "/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/tracker/scripts/deep_sort_pytorch/configs/deep_sort.yaml",
        "imgsz": 640,
        "augment": False,
        "conf_thres": 0.4,
        "iou_thres": 0.5,
        "patch_mode": True,
        "agnostic_nms": False,
        "classes": None,
        "view_img": True,
        "save_img": False,
        "device": ''}

    # Initialize
    set_logging()
    device = select_device(opt["device"])
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt["weights"], map_location=device)  # load FP32 model
    imgsz = check_img_size(opt["imgsz"], s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Initial deepsort
    cfg = get_config()
    cfg.merge_from_file(opt["config_deepsort"])
    deepsort = DeepSort(opt["deep_sort_weights"],
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Get names and colors: ['vehicle', 'pedestrian', 'red', 'green', 'sign_p', 'sign_w', 'sign_m', 'light']
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[139, 139, 0], [0, 165, 255],
              [0, 0, 255], [0, 255, 0],
              [180, 105, 255], [0, 255, 255], [255, 0, 0],
              [128, 128, 128]]

    # Save result as video
    if opt['save_img']:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (1920, 1200)
        fps = 25
        out_video = cv2.VideoWriter("/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/get_camera/data/out.avi", fourcc, fps, size)
    else:
        out_video = None

    temp = {
        "device": device,
        "half": half,
        "model": model,
        "deepsort_model": deepsort,
        "imgsz": imgsz,
        "names": names,
        "colors": colors,
        "out_video": out_video,
        "flag": False,
        "filter": Traffic_Light_Filter(num=3, maxsize=5, init=True, init_data=2),
        "lights_num": 0,
        "pub": rospy.Publisher("Traffic_Lights_State", traffic_lights_state, queue_size=1),
        "state_msg": traffic_lights_state()
    }

    opt.update(temp)
    return opt


def callback(data, args):
    if data.num <= 0:
        args["flag"] = False
    else:
        args['flag'] = True
        args["lights_num"] = int(data.num)
    # print(data.num, args["lights_num"])


def detect_and_track_2(data, args):
    with torch.no_grad():
        # Run inference
        t0 = time.time()

        # Read img from sensor_msg
        img0 = data

        if args["patch_mode"]:
            # Obtain patch
            center = np.array([1200 / 2, 1920 / 2])
            h, w = img0.shape[0], img0.shape[1]
            scale0, scale1 = 1 / 8, 1 / 6
            (patch0, patch1), (ratio_h, ratio_w), start1 = get_patch(img0, center, (h, w), scale0, scale1)

            # Padded resize
            patch, (new_h, new_w) = transform((patch0, patch1), args)
        else:
            patch = letterbox(img0, new_shape=args["imgsz"])[0]
            patch = patch[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            patch = np.ascontiguousarray(patch)

        patch = torch.from_numpy(patch).to(args["device"])
        patch = patch.half() if args["half"] else patch.float()  # uint8 to fp16/32
        patch /= 255.0  # 0 - 255 to 0.0 - 1.0
        if patch.ndimension() == 3:
            patch = patch.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = args["model"](patch, augment=args["augment"])[0]

        if args["patch_mode"]:
            # Transform to img coordinate and concatenate
            pred[-1][:, 2] *= ratio_w
            pred[-1][:, 3] *= ratio_h
            pred[-1][:, 0] = pred[-1][:, 0] * ratio_w + start1[1] * new_w / w
            pred[-1][:, 1] = pred[-1][:, 1] * ratio_h + start1[0] * new_h / h
            pred = pred.view(-1, 5+len(args["names"])).unsqueeze(0)

        # Apply NMS
        pred = non_max_suppression(pred, args["conf_thres"], args["iou_thres"],
                                   classes=args["classes"], agnostic=args["agnostic_nms"])
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = '%gx%g ' % patch.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(patch.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, args["names"][int(c)])  # add to string

                # Deepsort tracking
                xywh_bboxs, confs, classes = [], [], []
                for *xyxy, conf, cls in det:
                    # Draw detection result
                    # if args["view_img"]:  # Add bbox to image
                    #     label = '%s %.2f' % (args["names"][int(cls)], conf)
                    #     plot_one_box(xyxy, img0, label=label, color=args["colors"][int(cls)], line_thickness=1)
                    # To deep sort format
                    if cls <= 1:
                        x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                        xywh_bboxs.append(xywh_obj)
                        confs.append([conf.item()])
                        classes.append([cls.item()])
                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                classess = torch.Tensor(classes)
                outputs = []
                if xywhs.size()[0] > 0:
                    outputs = args['deepsort_model'].update(xywhs, confss, classess, img0)
                if len(outputs):
                    print("----------")
                    print("original tracker out")
                    print(outputs)
                    args['filter'].append(args["lights_num"], outputs)
                    final_results = [0 if np.mean(result[3:]) <= 0.5 else 1 for result in args['filter'].results]
                    print(final_results)
                    for i in range(len(final_results)):
                        args["state_msg"].state.append(final_results[i])
                    args["pub"].publish(args["state_msg"])
                    args["state_msg"].state = []
                    # ------------------------------- visualization ------------------------------ #
                    # bbox_xyxy = outputs[:, :4]
                    # identities = outputs[:, 4]
                    # scores = outputs[:, 5] / 100
                    # cls = outputs[:, 6]
                    # draw_boxes(img0, bbox_xyxy, identities, scores, cls, args["names"], args['colors'], 3)
                    print(args["filter"].results)
                    draw_results(img0, args["filter"].results, final_results, args["colors"], 3)

                # Print time (inference + NMS)
                # print('Detection And Tracking Done. %s (%.3fs)' % (s, t2 - t1))

        # Stream results
        if args["view_img"]:
            img0_show = cv2.resize(img0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            cv2.namedWindow("press q to quit", 0)
            cv2.imshow("press q to quit", img0_show)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
        if args['save_img']:
            args['out_video'].write(img0)

        # print('Visualization Done. (%.3fs)' % (time.time() - t0))


def load_param_2():
    opt = {
        # todo: 修改weights和deep_sort_weights的路径为实际权重位置，config_deepsort修改路径即可
        "weights": "/home/sheng/code_space/python_projects/competition/yolov5/best.pt",
        "deep_sort_weights": "/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/tracker/scripts/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7",
        "config_deepsort": "/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/tracker/scripts/deep_sort_pytorch/configs/deep_sort.yaml",
        "imgsz": 640,
        "augment": False,
        "conf_thres": 0.4,
        "iou_thres": 0.5,
        "patch_mode": True,
        "agnostic_nms": False,
        "classes": None,
        "view_img": True,
        "save_img": False,
        "device": ''}

    # Initialize
    set_logging()
    device = select_device(opt["device"])
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt["weights"], map_location=device)  # load FP32 model
    imgsz = check_img_size(opt["imgsz"], s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Initial deepsort
    cfg = get_config()
    cfg.merge_from_file(opt["config_deepsort"])
    deepsort = DeepSort(opt["deep_sort_weights"],
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Get names and colors: ['red', 'green', 'other']
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[0, 0, 255], [0, 255, 0], [0, 255, 255]]

    # Save result as video
    if opt['save_img']:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (1920, 1200)
        fps = 25
        # todo: 此处为输出检测结果的路径
        out_video = cv2.VideoWriter("/home/sheng/code_space/python_projects/competition/Traffic_Lights_Tracker/src/get_camera/data/out.avi", fourcc, fps, size)
    else:
        out_video = None

    temp = {
        "device": device,
        "half": half,
        "model": model,
        "deepsort_model": deepsort,
        "imgsz": imgsz,
        "names": names,
        "colors": colors,
        "out_video": out_video,
        "flag": False,
        "filter": Traffic_Light_Filter(num=3, maxsize=5, init=True, init_data=2),
        "lights_num": 0,
        "pub": rospy.Publisher("Traffic_Lights_State", traffic_lights_state, queue_size=1),
        "state_msg": traffic_lights_state()
    }

    opt.update(temp)
    return opt
