import hydra
import torch
import argparse
import time
from pathlib import Path
import sys
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import csv
import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
import pandas as pd

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

data_deque = {}

deepsort = None



north_line = [(235, 477), (954, 459)]
west_line = [(185,470), (155,587)]
east_line = [(1101,461), (1525, 479)]
south_line = [(459,756), (1600, 477)]




#############################################
# eastbound
EBT_counter = {}
# westbount 
EBR_counter = {}

EBL_counter = {}
# entering parkage
WBT_counter = {}
# leaving parkade
WBL_counter = {}

WBR_counter = {}

NBL_counter = {}
# leaving parkade
NBR_counter = {}

NBT_counter = {}

ALL_counter = {}

SBR_counter = {}

SBL_counter = {}

SBT_counter = {}



initial_direction = {
    "South": [],
    "North": [],
    "East": [],
    "West": []
}

all_Counter = {
    "South": {},
    "North": {},
    "East": {},
    "West": {}
}

# line = [[(185,470), (155,587)], [(1101, 461), (1525, 479)], [(459,756), (1600, 477)], [(235, 477), (954, 459)]] 

testfile = None

def check_route(id, direction, data_deque_item, img, obj_name):
        # NORTHLINE
        # print(self.id ,self.direction_enter, self.direction_leave)
        if intersect(data_deque_item[0], data_deque_item[1], north_line[0], north_line[1]):
            cv2.line(img, north_line[0], north_line[1], (255, 255, 255), 3)
            if "South" in direction:
                # self.direction_enter = "North"
                initial_direction["North"].append(id)
            elif "North" in direction:
                # self.direction_leave = "North"
                
                if obj_name not in all_Counter["North"]:
                    all_Counter["North"][obj_name] = 1
                else:
                    all_Counter["North"][obj_name] += 1
                # # check initial direction
                if id in initial_direction["South"]:
                    if obj_name not in NBT_counter:
                        NBT_counter[obj_name] = 1
                    else:
                        NBT_counter[obj_name] += 1
                elif id in initial_direction["East"]:
                    if obj_name not in EBL_counter:
                        EBL_counter[obj_name] = 1
                    else:
                        EBL_counter[obj_name] += 1
                elif id in initial_direction["West"]:
                    if obj_name not in WBR_counter:
                        WBR_counter[obj_name] = 1
                    else:
                        WBR_counter[obj_name] += 1

                # print(all_Counter["North"])

        if intersect(data_deque_item[0], data_deque_item[1], south_line[0], south_line[1]):
               
                cv2.line(img, south_line[0], south_line[1], (255, 255, 255), 3)
                if "North" in direction:
                    # self.direction_enter = "South"
                    initial_direction["South"].append(id)
                elif "South" in direction:
                    # self.direction_leave = "South"
                    
                    if obj_name not in all_Counter["South"]:
                        all_Counter["South"][obj_name] = 1
                    else:
                        all_Counter["South"][obj_name] += 1
                    # # check initial direction
                    if id in initial_direction["North"]:
                        print("found route")
                        if obj_name not in SBT_counter:
                            SBT_counter[obj_name] = 1
                        else:
                            SBT_counter[obj_name] += 1
                    elif id in initial_direction["East"]:
                        if obj_name not in EBR_counter:
                            EBR_counter[obj_name] = 1
                        else:
                            EBR_counter[obj_name] += 1
                    elif id in initial_direction["West"]:
                        if obj_name not in WBL_counter:
                            WBL_counter[obj_name] = 1
                        else:
                            WBL_counter[obj_name] += 1
                # print(all_Counter["South"])

        if intersect(data_deque_item[0], data_deque_item[1], west_line[0], west_line[1]):
            cv2.line(img, west_line[0], west_line[1], (255, 255, 255), 3)
            if "East" in direction:
                # self.direction_enter = "West"
                initial_direction["West"].append(id)
            elif "West" in direction:
                # self.direction_leave = "Wast"
                
                if obj_name not in all_Counter["West"]:
                    all_Counter["West"][obj_name] = 1
                else:
                    all_Counter["West"][obj_name] += 1
                # # check initial direction
                if id in initial_direction["East"]:
                    print("found route")
                    if obj_name not in WBT_counter:
                        WBT_counter[obj_name] = 1
                    else:
                        WBT_counter[obj_name] += 1
                elif id in initial_direction["North"]:
                    if obj_name not in SBR_counter:
                        SBR_counter[obj_name] = 1
                    else:
                        SBR_counter[obj_name] += 1
                elif id in initial_direction["South"]:
                    if obj_name not in NBL_counter:
                        NBL_counter[obj_name] = 1
                    else:
                        NBL_counter[obj_name] += 1

        if intersect(data_deque_item[0], data_deque_item[1], east_line[0], east_line[1]):
            cv2.line(img, east_line[0], east_line[1], (255, 255, 255), 3)
            if "West" in direction:
                # self.direction_enter = "East"
                initial_direction["East"].append(id)
            elif "East" in direction:
                # self.direction_leave = "East"
                
                if obj_name not in all_Counter["East"]:
                    all_Counter["East"][obj_name] = 1
                else:
                    all_Counter["East"][obj_name] += 1
                # # check initial direction
                if id in initial_direction["West"]:
                    print("found route")
                    if obj_name not in EBT_counter:
                        EBT_counter[obj_name] = 1
                    else:
                        EBT_counter[obj_name] += 1
                elif id in initial_direction["South"]:
                    if obj_name not in NBR_counter:
                        NBR_counter[obj_name] = 1
                    else:
                        NBR_counter[obj_name] += 1
                elif id in initial_direction["North"]:
                    if obj_name not in SBL_counter:
                        SBL_counter[obj_name] = 1
                    else:
                        SBL_counter[obj_name] += 1

        return

def init_tracker():
    # intialize tracker
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

def xyxy_to_xywh(*xyxy):
    """"
        Calculates the relative bounding box from absolute pixel values.

        args: xyxy from bounding box

        returns: some points x_c, y_c, width, height 
    """

    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    """
        Get tlwh from bbox_xyxy

        args: bbox_xyxy

        returns: tlwh_bboxs
    """
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
        Simple function that adds fixed color depending on the class

        fixes color values for classes

        args: bbox_label
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    """
        Draw a border around detected object

        args: img, pt1, pt2, color, thickness, r, d
    """
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)

    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    """
        Plots one bounding box on image img Draws
    """

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str



def draw_boxes(frame, img, bbox, names,object_id, identities=None, offset=(0, 0)):
    """
        draw the boxes on objects in the queue. 
    """
    for i in line:

        cv2.line(img, i[0], i[1], (0,0,255), 3)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)


        # add center to buffer
        data_deque[id].appendleft(center)

        if len(data_deque[id]) >= 2:

            direction = get_direction(data_deque[id][0], data_deque[id][1])
            check_route(id,direction, data_deque[id], img, obj_name)
            # do this per item in data queue

        UI_box(box, img, label=label, color=color, line_thickness=2)
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

    #4. Display Count in top right corner
        # for idx, (key, value) in enumerate(object_counter1.items()):
        #     cnt_str = str(key) + ":" +str(value)
        #     cv2.line(img, (width - 500,25), (width,25), [85,45,255], 40)
        #     cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        #     cv2.line(img, (width - 150, 65 + (idx*40)), (width, 65 + (idx*40)), [85, 45, 255], 30)
        #     cv2.putText(img, cnt_str, (width - 150, 75 + (idx*40)), 0, 1, [255, 255, 255], thickness = 2, lineType = cv2.LINE_AA)

        # for idx, (key, value) in enumerate(object_counter.items()):
        #     cnt_str1 = str(key) + ":" +str(value)
        #     cv2.line(img, (20,25), (500,25), [85,45,255], 40)
        #     cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)    
        #     cv2.line(img, (20,65+ (idx*40)), (127,65+ (idx*40)), [85,45,255], 30)
        #     cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        
        # for idx, (key, value) in enumerate(object_counter.items()):
        #     cnt_str1 = str(key) + ":" +str(value)
        #     cv2.line(img, (20,25), (500,25), [85,45,255], 40)
        #     cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)    
        #     cv2.line(img, (20,65+ (idx*40)), (127,65+ (idx*40)), [85,45,255], 30)
        #     cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)


    print(all_Counter)
    print([
        EBT_counter,
        EBL_counter,
        EBR_counter,
        WBT_counter,
        WBL_counter,
        WBR_counter,
        NBT_counter,
        NBL_counter,
        NBR_counter,
        SBT_counter,
        SBL_counter,
        SBR_counter
    ])

    if (frame%(1800) == 0):
        write_to_csv(frame)
        
        
        for key in EBT_counter:
            EBT_counter[key] = 0
        
        for key in EBR_counter:
            EBR_counter[key] = 0
        
        for key in EBL_counter:
            EBL_counter[key] = 0

        for key in WBT_counter:
            WBT_counter[key] = 0

        for key in WBL_counter:
            WBL_counter[key] = 0
        
        for key in WBR_counter:
            WBR_counter[key] = 0

        for key in NBL_counter:
            NBL_counter[key] = 0

        for key in NBR_counter:
            NBR_counter[key] = 0

        for key in NBT_counter:
            NBT_counter[key] = 0

    # elif(frame >= 15910):
    #     write_to_csv(frame)
    
    return img



class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            # log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, oids, im0)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]

            draw_boxes(frame ,im0, bbox_xyxy, self.model.names, object_id,identities)

        return log_string
    
def write_to_csv(frame):
    # testfile = "test.csv"
    delimiter_line = [('minute: %s' + '-' * 50)%(frame*(1/30)/60)]
    delimiter_line2 = [('-' * 60)]
    

    # EBT

    if "car" in EBT_counter:
        ebt_c = EBT_counter["car"]
    else:
        ebt_c = 0
    if "truck" in EBT_counter:
        ebt_t = EBT_counter["truck"]
    else: 
        ebt_t = 0
    if "bus" in EBT_counter:
        ebt_bu = EBT_counter["bus"]
    else: 
        ebt_bu = 0
    if "bicycle" in EBT_counter:
        ebt_bi = EBT_counter["bicycle"]
    else:
        ebt_bi = 0
    if "person" in EBT_counter:
        ebt_p = EBT_counter["person"]
    else:
        ebt_p = 0
    if "motorcycle" in EBT_counter:
        ebt_m = EBT_counter["motorcycle"]
    else:
        ebt_m = 0 
    
    # EBR
    if "car" in EBR_counter:
        ebr_c = EBR_counter["car"]
    else:
        ebr_c = 0
    if "truck" in EBR_counter:
        ebr_t = EBR_counter["truck"]
    else: 
        ebr_t = 0
    if "bus" in EBR_counter:
        ebr_bu = EBR_counter["bus"]
    else: 
        ebr_bu = 0
    if "bicycle" in EBR_counter:
        ebr_bi = EBR_counter["bicycle"]
    else:
        ebr_bi = 0
    if "person" in EBR_counter:
        ebr_p = EBR_counter["person"]
    else:
        ebr_p = 0
    if "motorcycle" in EBR_counter:
        ebr_m = EBR_counter["motorcycle"]
    else:
        ebr_m = 0

    if "car" in EBL_counter:
        ebl_c = EBL_counter["car"]
    else:
        ebl_c = 0
    if "truck" in EBL_counter:
        ebl_t = EBL_counter["truck"]
    else: 
        ebl_t = 0
    if "bus" in EBL_counter:
        ebl_bu = EBL_counter["bus"]
    else: 
        ebl_bu = 0
    if "bicycle" in EBL_counter:
        ebl_bi = EBL_counter["bicycle"]
    else:
        ebl_bi = 0
    if "person" in EBL_counter:
        ebl_p = EBL_counter["person"]
    else:
        ebl_p = 0
    if "motorcycle" in EBL_counter:
        ebl_m = EBL_counter["motorcycle"]
    else:
        ebl_m = 0

    # WBT

    if "car" in WBT_counter:
        wbt_c = WBT_counter["car"]
    else:
        wbt_c = 0
    if "truck" in WBT_counter:
        wbt_t = WBT_counter["truck"]
    else: 
        wbt_t = 0
    if "bus" in WBT_counter:
        wbt_bu = WBT_counter["bus"]
    else: 
        wbt_bu = 0
    if "bicycle" in WBT_counter:
        wbt_bi = WBT_counter["bicycle"]
    else:
        wbt_bi = 0
    if "person" in WBT_counter:
        wbt_p = WBT_counter["person"]
    else:
        wbt_p = 0
    if "motorcycle" in WBT_counter:
        wbt_m = WBT_counter["motorcycle"]
    else:
        wbt_m = 0


    # WBL

    if "car" in WBL_counter:
        wbl_c = WBL_counter["car"]
    else:
        wbl_c = 0
    if "truck" in WBL_counter:
        wbl_t = WBL_counter["truck"]
    else: 
        wbl_t = 0
    if "bus" in WBL_counter:
        wbl_bu = WBL_counter["bus"]
    else: 
        wbl_bu = 0
    if "bicycle" in WBL_counter:
        wbl_bi = WBL_counter["bicycle"]
    else:
        wbl_bi = 0
    if "person" in WBL_counter:
        wbl_p = WBL_counter["person"]
    else:
        wbl_p = 0
    if "motorcycle" in WBL_counter:
        wbl_m = WBL_counter["motorcycle"]
    else:
        wbl_m = 0

    if "car" in WBR_counter:
        wbr_c = WBR_counter["car"]
    else:
        wbr_c = 0
    if "truck" in WBR_counter:
        wbr_t = WBR_counter["truck"]
    else: 
        wbr_t = 0
    if "bus" in WBR_counter:
        wbr_bu = WBR_counter["bus"]
    else: 
        wbr_bu = 0
    if "bicycle" in WBR_counter:
        wbr_bi = WBR_counter["bicycle"]
    else:
        wbr_bi = 0
    if "person" in WBR_counter:
        wbr_p = WBR_counter["person"]
    else:
        wbr_p = 0
    if "motorcycle" in WBR_counter:
        wbr_m = WBR_counter["motorcycle"]
    else:
        wbr_m = 0


    # NBL

    if "car" in NBL_counter:
        nbl_c = NBL_counter["car"]
    else:
        nbl_c = 0
    if "truck" in NBL_counter:
        nbl_t = NBL_counter["truck"]
    else: 
        nbl_t = 0
    if "bus" in NBL_counter:
        nbl_bu = NBL_counter["bus"]
    else: 
        nbl_bu = 0
    if "bicycle" in NBL_counter:
        nbl_bi = NBL_counter["bicycle"]
    else:
        nbl_bi = 0
    if "person" in NBL_counter:
        nbl_p = NBL_counter["person"]
    else:
        nbl_p = 0
    if "motorcycle" in NBL_counter:
        nbl_m = NBL_counter["motorcycle"]
    else:
        nbl_m = 0

    # NBR

    if "car" in NBR_counter:
        nbr_c = NBR_counter["car"]
    else:
        nbr_c = 0
    if "truck" in NBR_counter:
        nbr_t = NBR_counter["truck"]
    else: 
        nbr_t = 0
    if "bus" in NBR_counter:
        nbr_bu = NBR_counter["bus"]
    else: 
        nbr_bu = 0
    if "bicycle" in NBR_counter:
        nbr_bi = NBR_counter["bicycle"]
    else:
        nbr_bi = 0
    if "person" in NBR_counter:
        nbr_p = NBR_counter["person"]
    else:
        nbr_p = 0
    if "motorcycle" in NBR_counter:
        nbr_m = NBR_counter["motorcycle"]
    else:
        nbr_m = 0

    if "car" in NBT_counter:
        nbt_c = NBT_counter["car"]
    else:
        nbt_c = 0
    if "truck" in NBT_counter:
        nbt_t = NBT_counter["truck"]
    else: 
        nbt_t = 0
    if "bus" in NBT_counter:
        nbt_bu = NBT_counter["bus"]
    else: 
        nbt_bu = 0
    if "bicycle" in NBT_counter:
        nbt_bi = NBT_counter["bicycle"]
    else:
        nbt_bi = 0
    if "person" in NBT_counter:
        nbt_p = NBT_counter["person"]
    else:
        nbt_p = 0
    if "motorcycle" in NBT_counter:
        nbt_m = NBT_counter["motorcycle"]
    else:
        nbt_m = 0



    row = [(frame*(1/30)/60),ebt_c, ebt_bu, ebt_t, ebt_bi, ebt_m, ebt_p, ebr_c, ebr_bu, ebr_t, ebr_bi, ebr_m, ebr_p, ebl_c, ebl_bu, ebl_t, ebl_bi, ebl_m, ebl_p, wbt_c, wbt_bu, wbt_t, wbt_bi, wbt_m, wbt_p, wbl_c, wbl_bu, wbl_t, wbl_bi, wbl_m, wbl_p, wbr_c, wbr_bu, wbr_t, wbr_bi, wbr_m, wbr_p, nbl_c, nbl_bu, nbl_t, nbl_bi, nbl_m, nbl_p, nbr_c, nbr_bu, nbr_t, nbr_bi, nbr_m, nbr_p, nbt_c, nbt_bu, nbt_t, nbt_bi, nbt_m, nbt_p]
    with open(testfile, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

    return

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    row = "time_minute,EBT_C,EBT_Bu,EBT_T,EBT_Bi,EBT_M,EBT_P,EBR_C,EBR_Bu,EBR_T,EBR_Bi,EBR_M,EBR_P,EBL_C,EBL_Bu,EBL_T,EBL_Bi,EBL_M,EBL_P,WBT_C,WBT_Bu,WBT_T,WBT_Bi,WBT_M,WBT_P,WBL_C,WBL_Bu,WBL_T,WBL_Bi,WBL_M,WBL_P,WBR_C,WBR_Bu,WBR_T,WBR_Bi,WBR_M,WBR_P,NBL_C,NBL_Bu,NBL_T,NBL_Bi,NBL_M,NBL_P,NBR_C,NBR_Bu,NBR_T,NBR_Bi,NBR_M,NBR_P,NBT_C,NBT_Bu,NBT_T,NBT_Bi,NBT_M,NBT_P"
    row = row.split(',')
    testfile = sys.argv[2][7:-4]+".csv"
    with open(testfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    predict()
    # write_to_csv()