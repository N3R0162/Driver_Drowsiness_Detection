import streamlit as st 

import cv2, os
import numpy as np
import importlib
import time
from math import floor
import pandas as pd
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
from faceboxes_detector import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from networks import *
import data_utils
from functions import *
from attention_score import AttentionScorer
#Init model variables:
experiment_name = "pip_32_16_60_r18_l2_l1_10_1_nb10"
data_name = "WFLW"
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
video_file = "/dev/video2"  #Camera_path
save_dir = "snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10"

my_config = importlib.import_module(config_path, package='Course_Related_2')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name
meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)
print("====================================")
print("Loading the model")
resnet18 = models.resnet18(pretrained=cfg.pretrained)
net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
# Set device (CPU/GPU)
if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)
print("Model loaded")

# Load the state dictionary
weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
state_dict = torch.load(weight_file, map_location=device)
net.load_state_dict(state_dict)
print("State dictionary loaded")
print("====================================")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])


def play_webcam():
    #Normal Camera
    source_webcam = "/dev/video0"
    #Jetson Nano CSI Camera
    # pipeline = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'

    if st.sidebar.button('Run'):
        try:
                # Display the DataFrame using st.table()
            # label_holder = st.empty()
            # df = pd.DataFrame(columns=["Aspect Ratio"])
            # label_holder.table(df)
            label_holder = st.empty()
            df = pd.DataFrame(columns=["Aspect Ratio", "PERCLOS Score", "Driver's Status"])
            styled_df = style_table(df)
            label_holder.table(styled_df)
            detector = FaceBoxesDetector('FaceBoxes', 'FaceBoxesV2/weights/FaceBoxesV2.pth', True, torch.device("cuda:0"))
            my_thresh = 0.9
            det_box_scale = 1.2
            net.eval()
            count = 0
            sleepy_frames = 0
            print("Starting the video")
            #OpenCV camera
            cap = cv2.VideoCapture(source_webcam)
            #Jetson Nano CSI Camera
            # cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

            print("Video loaded")
            print("====================================")
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            st_frame = st.empty()
            #Set up parameters for the video
            t_0 = time.perf_counter()
            score = AttentionScorer(t_now=t_0, ear_thresh=0.1, gaze_thresh=0.2, perclos_thresh=0.2, roll_thresh=15, pitch_thresh=15, yaw_thresh=15, ear_time_thresh=0.2, gaze_time_thresh=0.2, pose_time_thresh=4.0, verbose=False)
            i = 0 
            time.sleep(0.01)
            print("Starting the processing loop")
            while (cap.isOpened()):
                t_now = time.perf_counter()
                fps = i/(t_now-t_0)
                if fps == 0:
                    fps = 10
                ret, frame = cap.read()
                # print("Start reading frame")
                if ret == True:        
                    # print("Frame readed")          
                    detections, _ = detector.detect(frame, my_thresh, 1)
                    # print("Start processing frame")
                    for i in range(len(detections)):
                        det_xmin = detections[i][2]
                        det_ymin = detections[i][3]
                        det_width = detections[i][4]
                        det_height = detections[i][5]
                        det_xmax = det_xmin + det_width - 1
                        det_ymax = det_ymin + det_height - 1

                        det_xmin -= int(det_width * (det_box_scale-1)/2)
                        det_ymin += int(det_height * (det_box_scale-1)/2)
                        det_xmax += int(det_width * (det_box_scale-1)/2)
                        det_ymax += int(det_height * (det_box_scale-1)/2)
                        det_xmin = max(det_xmin, 0)
                        det_ymin = max(det_ymin, 0)
                        det_xmax = min(det_xmax, frame_width-1)
                        det_ymax = min(det_ymax, frame_height-1)
                        det_width = det_xmax - det_xmin + 1
                        det_height = det_ymax - det_ymin + 1
                        cv2.rectangle(frame, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)
                        det_crop = frame[det_ymin:det_ymax, det_xmin:det_xmax, :]
                        det_crop = cv2.resize(det_crop, (256, 256))
                        inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
                        inputs = preprocess(inputs).unsqueeze(0)
                        inputs = inputs.to(torch.device("cuda:0"))
                        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, 256, 32, 10)
                        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
                        tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
                        tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
                        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
                        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
                        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
                        lms_pred = lms_pred.cpu().numpy()
                        lms_pred_merge = lms_pred_merge.cpu().numpy()
                        for i in range(cfg.num_lms):
                            x_pred = lms_pred_merge[i*2] * det_width
                            y_pred = lms_pred_merge[i*2+1] * det_height
                            cv2.circle(frame, (int(x_pred)+det_xmin, int(y_pred)+det_ymin), 1, (0, 0, 255), -1)
                        st_frame.image(image=frame, caption="Detected Video", channels="BGR", width=700)
                        average_aspect_ratio, left_aspect_ratio, right_aspect_ratio = calculate_aspect_ratio(lms_pred_merge)
                        tired, perclos_score = score.get_PERCLOS(t_now, fps, average_aspect_ratio)

                        # Format aspect ratio value
                        aspect_ratio_str = f"{average_aspect_ratio:.2f}"

                        # Format PERCLOS score value
                        perclos_score_str = f"{perclos_score:.2f}"

                        # Update the displayed DataFrame with sample data
                        df = pd.DataFrame({"Aspect Ratio": [aspect_ratio_str], "PERCLOS Score": [perclos_score_str], "Driver's Status": [tired]})
                        styled_df = style_table(df)
                        label_holder.table(styled_df)
                else:
                    cap.release()

                    print("Error", ret)
                    continue
        except Exception as e:
            print("Error loading video: " + str(e))
            st.sidebar.error("Error loading video: " + str(e))


