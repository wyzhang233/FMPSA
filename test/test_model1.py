import torch
import os
import numpy as np
import cv2
from FMPSA import FMPSA
import glob
from utils.utils import reconstruction_whole_image_cpu, save_matv73

model_path = r"E:\test\model\net_epoch.pth"
result_path = r'E:\test\test_results'
img_path = r'E:\test\NTIRE2020_Test_Clean'
var_name = 'cube'

if not os.path.exists(result_path):
    os.makedirs(result_path)
model = FMPSA(3, 31, 128, 8)
save_point = torch.load(model_path, map_location='cpu')
model_param = save_point['state_dict']
model_dict = {}
for k1, k2 in zip(model.state_dict(), model_param):
    model_dict[k1] = model_param[k2]
model.load_state_dict(model_dict)

img_path_name = glob.glob(os.path.join(img_path, '*.png'))
img_path_name.sort()
       
for i in range(len(img_path_name)):
    rgb = cv2.imread(img_path_name[i])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = np.float32(rgb) / 255.0
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
    print(img_path_name[i].split('/')[-1])

    _, img_res = reconstruction_whole_image_cpu(rgb, model)
    mat_name = img_path_name[i].split('/')[-1][:-10] + '.mat'
    mat_dir = os.path.join(result_path, mat_name)

    save_matv73(mat_dir, var_name, img_res)
