import cv2, os
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
import numpy as np
import pickle
import importlib
from math import floor
from faceboxes_detector import *
import time

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

if not len(sys.argv) == 4:
    print('Format:')
    print('python lib/demo_video.py config_file video_file')
    exit(0)
experiment_name = sys.argv[1].split('/')[-1][:-3]
data_name = sys.argv[1].split('/')[-2]
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)
video_file = sys.argv[2]

my_config = importlib.import_module(config_path, package='Driver_Drowsiness_Detection')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name

save_dir = sys.argv[3]

meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)
print("MAX LEN: ", max_len)
if cfg.backbone == 'resnet18':
    resnet18 = models.resnet18(pretrained=cfg.pretrained)
    net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
elif cfg.backbone == 'resnet50':
    resnet50 = models.resnet50(pretrained=cfg.pretrained)
    net = Pip_resnet50(resnet50, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
elif cfg.backbone == 'resnet101':
    resnet101 = models.resnet101(pretrained=cfg.pretrained)
    net = Pip_resnet101(resnet101, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
elif cfg.backbone == 'mobilenet_v2':
    mbnet = models.mobilenet_v2(pretrained=cfg.pretrained)
    net = Pip_mbnetv2(mbnet, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
else:
    print('No such backbone!')
    exit(0)

if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)

weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
state_dict = torch.load(weight_file, map_location=device)
net.load_state_dict(state_dict)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])

def calculate_aspect_ratio(lms_pred):
    point_0 = np.array([lms_pred[0], lms_pred[1]])
    point_1 = np.array([lms_pred[2], lms_pred[3]])
    point_2 = np.array([lms_pred[4], lms_pred[5]])
    point_3 = np.array([lms_pred[6], lms_pred[7]])
    point_4 = np.array([lms_pred[8], lms_pred[9]])
    point_5 = np.array([lms_pred[10], lms_pred[11]])
    point_6 = np.array([lms_pred[12], lms_pred[13]])
    point_7 = np.array([lms_pred[14], lms_pred[15]])
    point_8 = np.array([lms_pred[16], lms_pred[17]])
    point_9 = np.array([lms_pred[18], lms_pred[19]])
    point_10 = np.array([lms_pred[20], lms_pred[21]])
    point_11 = np.array([lms_pred[22], lms_pred[23]])
    point_12 = np.array([lms_pred[24], lms_pred[25]])
    point_13 = np.array([lms_pred[26], lms_pred[27]])
    point_14 = np.array([lms_pred[28], lms_pred[29]])
    point_15 = np.array([lms_pred[30], lms_pred[31]])


    left_height_1 = np.linalg.norm(point_1 - point_7)
    left_height_2 = np.linalg.norm(point_2 - point_6)
    left_height_3 = np.linalg.norm(point_3 - point_5)
    left_width = np.linalg.norm(point_0 - point_4)
    left_eye_aspect_ratio = (left_height_1 + left_height_2 + left_height_3) / (3 * left_width)

    right_height_1 = np.linalg.norm(point_9 - point_15)
    right_height_2 = np.linalg.norm(point_10 - point_14)    
    right_height_3 = np.linalg.norm(point_11 - point_13)
    right_width = np.linalg.norm(point_8 - point_12)
    right_eye_aspect_ratio = (right_height_1 + right_height_2 + right_height_3) / (3 * right_width)

    average_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2
    return average_aspect_ratio


def demo_video_sleepy(video_file, net, preprocess, input_size, net_stride, num_nb, use_gpu, device):
    detector = FaceBoxesDetector('FaceBoxes', 'FaceBoxesV2/weights/FaceBoxesV2.pth', use_gpu, device)
    my_thresh = 0.9
    det_box_scale = 1.2

    net.eval()
    if video_file == 'camera':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_file)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    count = 0
    sleepy_frames = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            detections, _ = detector.detect(frame, my_thresh, 1)
            for i in range(len(detections)):
                det_xmin = detections[i][2]
                det_ymin = detections[i][3]
                det_width = detections[i][4]
                det_height = detections[i][5]
                det_xmax = det_xmin + det_width - 1
                det_ymax = det_ymin + det_height - 1

                det_xmin -= int(det_width * (det_box_scale-1)/2)
                # remove a part of top area for alignment, see paper for details
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
                det_crop = cv2.resize(det_crop, (input_size, input_size))
                inputs = Image.fromarray(det_crop[:,:,::-1].astype('uint8'), 'RGB')
                inputs = preprocess(inputs).unsqueeze(0)
                inputs = inputs.to(device)
                lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb)
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
                
                # Extract relevant features and classify user as sleepy or not
                aspect_ratio = calculate_aspect_ratio(lms_pred_merge)
                print("Aspect ratio: ", aspect_ratio)
                if aspect_ratio < SLEEPY_THRESHOLD:
                    sleepy_frames += 1
                else:
                    sleepy_frames = 0
                
                # If user is classified as sleepy for a certain number of frames, take action
                if sleepy_frames >= SLEEPY_FRAME_THRESHOLD:
                    # Play alarm or send notification
                    print("User is sleepy!")
                    exit()
                    sleepy_frames = 0

            # count += 1
            # current_dir = os.getcwd()
            # result_folder = os.path.join(current_dir, 'results')
            # relative_path = os.path.join(result_folder, 'WFLW_mix_230713/Mix_200epoch_glasses')
            # if not os.path.exists(relative_path):
            #     os.makedirs(relative_path)
            # cv2.imwrite(relative_path+str(count)+'.jpg', frame)
            cv2.imshow('1', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

SLEEPY_THRESHOLD = 0
SLEEPY_FRAME_THRESHOLD = 10
demo_video_sleepy(video_file, net, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb, cfg.use_gpu, device)