import numpy as np
import torch


def build_projection_matrix(w=1280, h=720, fov=100):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_camera_world_view_matrix(coords, K_inverse, c2w, desired_z, sensor_location):
    n = coords.shape[-2]
    batchsize, bs_num = coords.shape[0], coords.shape[1]
    device = coords.device
    ones = torch.ones_like(coords,device=device)
    v = torch.transpose(torch.cat((coords, ones), dim=-1)[..., :3], -1, -2)
    F = torch.tensor([[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=torch.float64, device=device)
    F = torch.inverse(F).unsqueeze(0).unsqueeze(1).repeat(batchsize, bs_num, 1, 1)
    K_inverse = K_inverse.unsqueeze(0).unsqueeze(1).repeat(batchsize, bs_num, 1, 1)
    c2w = c2w.unsqueeze(0).repeat(batchsize, 1, 1, 1)
    sensor_location = sensor_location.unsqueeze(0).unsqueeze(2).repeat(batchsize, 1, n, 1)

    to_camera = torch.matmul(K_inverse, v)
    to_camera = torch.matmul(F, to_camera)

    ones_new = torch.transpose(ones[..., :1], -1, -2)
    to_camera = torch.cat((to_camera, ones_new), dim=-2)
    to_world = torch.transpose(torch.matmul(c2w, to_camera), -1, -2)

    vec = to_world[..., :3] - sensor_location
    vec = vec / vec[..., 2].unsqueeze(-1) * (
                to_world[..., 2].unsqueeze(-1) - desired_z.unsqueeze(0).unsqueeze(1).unsqueeze(-1))
    location = to_world[..., :3] - vec

    return location

def word2camera(location, roll, pitch, yaw):
    # Convert angles from degrees to radians
    R = rotation_matrix(roll, pitch, yaw)
    # The translation vector
    T = np.array(location)
    # Create the extrinsic matrix
    # Note: The rotation matrix needs to be transposed to go from world to camera coordinates
    extrinsic = np.eye(4)  # Create a 4x4 identity matrix
    extrinsic[:3, :3] = R  # Transpose of rotation matrix
    extrinsic[:3, 3] =  T  # Negative because we translate the world in the opposite direction of the camera
    return extrinsic
def rotation_matrix(roll, pitch, yaw):
    roll = -np.radians(roll)
    pitch = -np.radians(pitch)
    yaw = np.radians(yaw)
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    # Matrix multiplication order depends on the specific convention being used
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    return R

def obtain_K_w2c():
    camera_configs = [
        {"location": [60, -55, 15], "rotation": [-30, -224, 0], "fov": 100},
        {"location": [-55, -60, 15], "rotation": [-30, -315, 0], "fov": 100},
        {"location": [-60, 55, 15], "rotation": [-30, -43, 0], "fov": 100},
        {"location": [55, 60, 15], "rotation": [-30, -135, 0], "fov": 100},
    ]
    w2c_list=[]
    K_list=[]
    location_list=[]
    for config in camera_configs:
        location = config["location"]
        location_list.append(location)
        rotation = config["rotation"]
        fov = config["fov"]
        yaw = rotation[1]
        roll = rotation[2]
        pitch = rotation[0]
        w2c_list.append(word2camera(location, roll=roll,pitch=pitch, yaw=yaw))
        K_list.append(build_projection_matrix(fov=fov))
    return w2c_list,K_list[0],location_list
def construct_projection_matrix(uv,K,w2c_list,location_z,location):
    aa=get_camera_world_view_matrix(uv, np.linalg.inv(K), w2c_list, location_z,location)
    return aa



def get_bounding_box_centers_matrix(imgs2,bs_num=4):
    """
    Loads a tensor from a file, computes the centers of bounding boxes, and returns them as a NumPy array.

    Parameters:
    tensor_file_path (str): The file path to the tensor data.

    Returns:
    np.ndarray: The computed centers of the bounding boxes as a NumPy array.
    """
    # Load the tensor from the given file path
    batchsize,bs_num,car_num,_=imgs2.size()

    # Compute the centers of the bounding boxes
    centen_cam0_x =( (imgs2[..., 0] + imgs2[..., 2]) / 2).unsqueeze(-1)
    centen_cam0_y =( (imgs2[..., 1] + imgs2[...,3]) / 2).unsqueeze(-1)

    centen_cam0_y=centen_cam0_y-280*(centen_cam0_y>0)
    centers = torch.cat((centen_cam0_x, centen_cam0_y), -1)

    # Convert the centers to a NumPy array and return
    # return centers.cpu().numpy()
    return centers


