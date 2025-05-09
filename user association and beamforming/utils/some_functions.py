
import os
from utils.camera_for_location import get_bounding_box_centers_matrix, construct_projection_matrix, obtain_K_w2c
import numpy as np
import torch
def create_directory(base_path, base_name,training=True):
    # Initialize directory path
    directory = os.path.join(base_path, base_name)
    i = 1

    # Check if the directory exists and modify the name if it does
    while os.path.exists(directory):
        directory = os.path.join(base_path, f"{base_name}{i}")
        i += 1
    if training:
        # Create the directory
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    return directory

def preprocess_data(imgs, targets, device, divide_num, bs_num, car_max_num, K_list, w2c_list, height, location_list,car_choose_num):
    batchsize = np.size(imgs, 0)
    imgs_stack=imgs.view(batchsize, bs_num, car_max_num, -1)
    imgs2 =imgs_stack[:,:,:car_choose_num,:]
    height2=height[:car_choose_num]
    center_matrix = get_bounding_box_centers_matrix(imgs2, bs_num)
    estimated_loc = construct_projection_matrix(center_matrix, K_list, np.stack(w2c_list), height2, np.stack(location_list))
    undetect_label = np.sum(center_matrix, axis=-1) < 0
    detect_label =np.sum(center_matrix, axis=-1) > 0
    detect_label_tensor = torch.tensor(detect_label, device=device).float().permute(0, 2, 1)
    estimated_loc[undetect_label, :] = -56
    location=targets[:,:car_choose_num,0,0,7:10]
    remove_loc_tensor = torch.tensor(estimated_loc, device=device).float().view(batchsize, -1)/56
    imgs_new = (imgs2 / divide_num).view(batchsize, -1).to(device).float()
    imgs_new[imgs_new == 0] = -1
    return imgs_new, remove_loc_tensor, detect_label_tensor