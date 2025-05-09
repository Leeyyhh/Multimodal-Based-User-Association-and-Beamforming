import math

import numpy as np
import torch
import torch.nn.functional as F
import sys


class beam_Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for Python 3 compatibility.
        # This handles the flush command by doing nothing.
        # You might want to specify some extra behavior here.
        pass
def prepare_model_inputs(pred2, fixed_length=25):
    """
    Prepares model inputs by padding, stacking, and reshaping tensors.

    Parameters:
    - pred2: List of tensors after applying non-max suppression.
    - bs_relationships: Dictionary mapping BS indices to their related source indices.
    - fixed_length: The fixed length to which tensors are padded.

    Returns:
    - Tensor ready for model inputs, reshaped and gathered according to bs_relationships.
    """
    bs_relationships = {
        0: [0, 1, 3],  # BS1 gets information from itself, BS2, and BS4
        1: [0, 1, 2],  # BS2 from itself, BS1, and BS3
        2: [1, 2, 3],  # BS3 from itself, BS2, and BS4
        3: [0, 2, 3],  # BS4 from itself, BS1, and BS3
    }
    # Pad tensors to a fixed length and stack
    pred2_padded = [F.pad(tensor, (0, 0, 0, max(0, fixed_length - tensor.size(0)))) for tensor in pred2]
    pred2_stacked = torch.stack(pred2_padded).reshape(-1, 4, fixed_length * 6)

    # Determine new shape based on bs_relationships
    N, _, H = pred2_stacked.shape
    new_shape = (4, N, max(len(sources) for sources in bs_relationships.values()), H)

    # Initialize a tensor for gathered inputs
    gathered_pred2 = torch.zeros(new_shape, dtype=pred2_stacked.dtype, device=pred2_stacked.device)

    # Gather inputs according to bs_relationships
    for bs, sources in bs_relationships.items():
        for i, source in enumerate(sources):
            gathered_pred2[bs, :, i, :] = pred2_stacked[:, source, :]

    # Reshape for model input
    # gathered_pred2 = gathered_pred2.reshape(4, -1, fixed_length * 6)

    return N,gathered_pred2
import os

def prepare_model_inputs3(pred2, fixed_length=25):
    """
    Prepares model inputs by padding, stacking, and reshaping tensors.

    Parameters:
    - pred2: List of tensors after applying non-max suppression.
    - bs_relationships: Dictionary mapping BS indices to their related source indices.
    - fixed_length: The fixed length to which tensors are padded.

    Returns:
    - Tensor ready for model inputs, reshaped and gathered according to bs_relationships.
    """
    # bs_relationships = {
    #     0: [0],  # BS1 gets information from itself, BS2, and BS4
    #     1: [0],  # BS2 from itself, BS1, and BS3
    #     2: [1],  # BS3 from itself, BS2, and BS4
    #     3: [0],  # BS4 from itself, BS1, and BS3
    # }
    batchsize=len(pred2)

    # Create a tensor to gather all data (initialized to -1.0)
    gathered_pred2 = torch.full([batchsize, fixed_length, 6], -1.0)

    # Extract indices and use them for direct assignment
    for i, tensor in enumerate(pred2):
        indices = tensor[:, -1].long()  # Ensure indices are of type long for indexing
        valid_rows = indices < fixed_length  # Ensure indices do not exceed fixed_length
        gathered_pred2[i, indices[valid_rows]] = tensor[valid_rows]

    # Reshape for model input
    gathered_pred2 = gathered_pred2.reshape(batchsize//4, 4, fixed_length * 6).permute(1,0,2)

    return batchsize//4,gathered_pred2
def prepare_model_inputs2(pred2, fixed_length=25):
    """
    Prepares model inputs by padding, stacking, and reshaping tensors.

    Parameters:
    - pred2: List of tensors after applying non-max suppression.
    - bs_relationships: Dictionary mapping BS indices to their related source indices.
    - fixed_length: The fixed length to which tensors are padded.

    Returns:
    - Tensor ready for model inputs, reshaped and gathered according to bs_relationships.
    """
    # bs_relationships = {
    #     0: [0],  # BS1 gets information from itself, BS2, and BS4
    #     1: [0],  # BS2 from itself, BS1, and BS3
    #     2: [1],  # BS3 from itself, BS2, and BS4
    #     3: [0],  # BS4 from itself, BS1, and BS3
    # }

    bs_relationships = {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
    }
    pred2=pred2
    # Pad tensors to a fixed length and stack
    pred2_padded = [F.pad(tensor, (0, 0, 0, max(0, fixed_length - tensor.size(0)))) for tensor in pred2]
    pred2_stacked = torch.stack(pred2_padded).reshape(-1, 4, fixed_length * 6)

    # Determine new shape based on bs_relationships
    N, _, H = pred2_stacked.shape
    new_shape = (4, N, max(len(sources) for sources in bs_relationships.values()), H)

    # Initialize a tensor for gathered inputs
    gathered_pred2 = torch.zeros(new_shape, dtype=pred2_stacked.dtype, device=pred2_stacked.device)

    # Gather inputs according to bs_relationships
    for bs, sources in bs_relationships.items():
        for i, source in enumerate(sources):
            gathered_pred2[bs, :, i, :] = pred2_stacked[:, source, :]

    # Reshape for model input
    gathered_pred2 = gathered_pred2.reshape(4, -1, fixed_length * 6)

    return N,gathered_pred2

def deduct_new_model_input(gathered_pred2,fixed_length):
    new_bs_relationships = {
        0: [0, 1, 3],
        1: [0, 1, 2],
        2: [1, 2, 3],
        3: [0, 2, 3],
    }

    # The shape of 'gathered_pred2' would be (4, N, 1, H) based on 'prepare_model_inputs2'
    # You need to adjust it to match the new 'bs_relationships', potentially leading to a shape of (4, N, 3, H)
    bs_num, batchsize, H = gathered_pred2.shape
    # Pseudocode to adapt 'gathered_pred2' to new relationships
    new_shape = ( bs_num, batchsize, 3, H)  # As per new 'bs_relationships'
    adjusted_gathered_pred2 = torch.zeros(new_shape, dtype=gathered_pred2.dtype, device=gathered_pred2.device)

    for bs, sources in new_bs_relationships.items():
        for i, source in enumerate(sources):
            adjusted_gathered_pred2[bs, :, i, :] = gathered_pred2[source, :, :]  # Adjust indexing as needed

    # Reshape if necessary, depending on the final requirements for the model input
    adjusted_gathered_pred2 = adjusted_gathered_pred2.reshape(4, -1, 3*fixed_length * 6)
    return adjusted_gathered_pred2


def construct_car_size():
    data = [
        [0,2.6828393936157227, 0.9003620743751526, 0.7874829769134521],
        [1,2.427854299545288, 1.0163782835006714, 0.8246796727180481],
        [2,1.9938424825668335, 0.9254241585731506, 0.8085547685623169],
        [3,2.0906050205230713, 0.9970585703849792, 0.6926480531692505],
        [4,2.5039126873016357, 0.9408109784126282, 0.7673624753952026],
        [5,2.3368194103240967, 0.9059062600135803, 0.7209736704826355],
        [6,2.358762502670288, 0.947413444519043, 0.650469958782196],
        [7,2.0964150428771973, 0.9080929160118103, 0.7369155883789062],
        [8,2.3958897590637207, 1.081725001335144, 0.7438300251960754],
        [9,1.9029000997543335, 0.9851378202438354, 0.7375150918960571],
        [10,2.5133883953094482, 1.0757731199264526, 0.8253258466720581],
        [11,2.6787397861480713, 1.0166014432907104, 0.7053293585777283]
    ]
    vehicle_dimensions = np.array(data)
    vehicle_dimensions[:,1:]=vehicle_dimensions[:,1:]*2


    # Convert the list to a NumPy array

    return vehicle_dimensions

def deduct_new_model_input2(gathered_pred2,car_max_num):
    # You need to adjust it to match the new 'bs_relationships', potentially leading to a shape of (4, N, 3, H)
    adjusted_gathered_pred2=gathered_pred2.permute(1,0,2).reshape(-1,4*car_max_num * 6)

    # Reshape if necessary, depending on the final requirements for the model input
    return adjusted_gathered_pred2



def calculate_received_signal_strength_for_batches(input_arrays, beamforming_vectors_input, antenna_array,device,total_power,user_A_matrix,choosen_index=2):
    """
    Calculate the received signal strength for batches of scenarios, where each input array
    corresponds to a specific beamforming vector.

    Parameters:
    - input_arrays:  arrays, each of shape (batchsize,receiver_num, bs_num,number of multipath, 7), where each array corresponds to a specific scenario
    - beamforming_vectors:  PyTorch tensors, the value is (batchsize,receiver_num,bs_number,antenna_number), corresponding one-to-one with the input arrays
    - antenna_vectors: Antenna configuration vectors, representing the physical configuration of each antenna element
    - beamforming_vectors_list size:[batch_size,bs_num,antenna_num]
    -input_arrays size:a list of tensors,len(input_arrays)=batch_size,input_arrays[0].size()=[receiver_num,bs_num,mp_num,7]
    Returns:
    - all_received_signal_strengths: A list of lists, containing the received signal strengths for each receiver and multipath component for each scenario
    """
    sinr_functions = [cal_SINR_server, cal_SINR_server3, cal_SINR_server5, cal_SINR_server7]
    if choosen_index < 1 or choosen_index > len(sinr_functions):
        raise Exception("The choosen_index is not correct")
    sinr_function = sinr_functions[choosen_index - 1]

    rate, amplitude_abs, commitment_loss = sinr_function(input_arrays, beamforming_vectors_input, antenna_array, device, total_power, user_A_matrix)
    binary_rate, _, _ = sinr_function(input_arrays, beamforming_vectors_input, antenna_array, device, total_power, user_A_matrix, 1)

    # Calculate metrics
    max_rate = torch.max(rate, dim=1, keepdim=True).values.detach() + 1
    gap_rate = torch.mean(torch.square(max_rate - rate), dim=1)


    loss = torch.min(rate, dim=1).values
    min_values = loss
    min_values2 = torch.min(binary_rate, dim=1).values
    skip_training = ~torch.all(torch.isfinite(min_values))
    zero_num = torch.sum(min_values == 0)
    zero_num2 = torch.sum(min_values2 == 0)
    return loss,min_values,min_values2,skip_training,commitment_loss,zero_num,zero_num2


def calculate_sumrate_for_batches(input_arrays, beamforming_vectors_input, antenna_config,device,total_power,user_A_matrix):
    """
    Calculate the received signal strength for batches of scenarios, where each input array
    corresponds to a specific beamforming vector.

    Parameters:
    - input_arrays:  arrays, each of shape (batchsize,receiver_num, bs_num,number of multipath, 7), where each array corresponds to a specific scenario
    - beamforming_vectors:  PyTorch tensors, the value is (batchsize,receiver_num,bs_number,antenna_number), corresponding one-to-one with the input arrays
    - antenna_vectors: Antenna configuration vectors, representing the physical configuration of each antenna element
    - beamforming_vectors_list size:[batch_size,bs_num,antenna_num]
    -input_arrays size:a list of tensors,len(input_arrays)=batch_size,input_arrays[0].size()=[receiver_num,bs_num,mp_num,7]
    Returns:
    - all_received_signal_strengths: A list of lists, containing the received signal strengths for each receiver and multipath component for each scenario
    """
    SINR,applitude_abs,commitment_loss=cal_SINR_server(input_arrays, beamforming_vectors_input, antenna_config,device,total_power,user_A_matrix)

    # # Calculate rate for each user
    user_rate= torch.log2(1 + SINR)
    sumrate = torch.sum(torch.log2(1 + SINR),-1)

    # masked_tensor = torch.where(applitude_abs > 0, rate, torch.full_like(rate, float('inf')))
    #
    # # Find the minimum value among positive values along dimension 1
    # min_values = torch.min(masked_tensor, dim=1).values
    # skip_training=~torch.all(torch.isfinite(min_values))
    # sumrate=torch.sum(torch.abs(diagonal_values)**2 ,dim=1)
    skip_training=False
    return sumrate,skip_training,commitment_loss
def normalize_user_matrix(user_A_matrix, choose_method):
    """
    Normalizes the user_A_matrix using the specified method.

    Parameters:
    - user_A_matrix: A PyTorch tensor to be normalized.
    - choose_method: A string indicating the normalization method to use. Options are "softmax", "sigmoid", or "abs".

    Returns:
    - A normalized PyTorch tensor according to the specified method.
    """
    # Apply softmax normalization
    if choose_method == "softmax":
        Final_user_A_matrix = torch.nn.functional.softmax(user_A_matrix, dim=-1)
    # Apply sigmoid normalization and then divide by the sum of the tensor along the last dimension
    elif choose_method == "sigmoid":
        sigmoid_user_A_matrix = torch.nn.functional.sigmoid(user_A_matrix)
        eps = 1e-3
        Final_user_A_matrix = sigmoid_user_A_matrix / (torch.sum(sigmoid_user_A_matrix, dim=-1, keepdim=True) + eps)
    # Apply absolute value normalization and then divide by the sum of the tensor along the last dimension
    elif choose_method == "abs":
        abs_user_A_matrix = torch.abs(user_A_matrix)
        eps = 1e-3
        Final_user_A_matrix = abs_user_A_matrix / (torch.sum(abs_user_A_matrix, dim=-1, keepdim=True) + eps)
    else:
        raise ValueError("Unsupported normalization method. Choose 'softmax', 'sigmoid', or 'abs'.")

    return Final_user_A_matrix
def generate_position_grid(antenna_array, d, device,bs_num):
    """
    Generates a mesh grid of x and z positions for antennas, centered around the mean position.

    Parameters:
    - antenna_array: Numpy array of shape (N, 2), where N is the number of antennas.
                     Each row represents an antenna with its x and z positions.
    - d: Float, the spacing between positions.
    - device: The target device for PyTorch tensors (e.g., 'cpu' or 'cuda').

    Returns:
    - x_grid, z_grid: Two 2D PyTorch tensors representing the mesh grid of x and z positions.
    """
    # Reshape antenna_array if necessary (commented out since it might already be in required shape)
    antenna_array = np.reshape(antenna_array, [bs_num, -1])
    # Find the maximum x and z positions
    x_max_antenna = np.max(antenna_array[:, 0])
    z_max_antenna = np.max(antenna_array[:, 1])

    # Generate and center x positions
    x_positions = (torch.arange(x_max_antenna, dtype=torch.float32,device=device) * d)
    x_positions_new = x_positions - torch.mean(x_positions)

    # Generate and center z positions
    z_positions = (torch.arange(z_max_antenna, dtype=torch.float32,device=device) * d)
    z_positions_new = z_positions - torch.mean(z_positions)

    # Create a mesh grid using the centered positions
    x_grid, z_grid = torch.meshgrid(x_positions_new, z_positions_new, indexing='ij')

    return x_grid, z_grid

def cal_SINR_server(input_arrays, beamforming_vectors_input, antenna_array, device, total_power, user_A_matrix, binary=0):
    bs_num, lambda_ = 4, 1
    applitude_complex = torch.complex(input_arrays[..., 0], input_arrays[..., 1])
    phi_angles, theta_angles = input_arrays[..., 5], input_arrays[..., 6]
    max_antenna_x, max_antenna_z = np.max(antenna_array[:, 0]), np.max(antenna_array[:, 1])
    array_power = np.array([x[0] * x[1] for x in antenna_array])
    array_power = total_power / sum(array_power) * array_power
    batchsize, receiver_num, bs_number, _, _, _ = beamforming_vectors_input.size()
    d = lambda_ / 2  # Half-wavelength spacing
    scalor_factor, noise_power = math.sqrt(10 ** 8), 1

    normalized_user_A_matrix = normalize_user_matrix(user_A_matrix, "abs")
    indices = torch.argmax(normalized_user_A_matrix, dim=-1, keepdim=True)
    hat_user_A_matrix = torch.zeros_like(normalized_user_A_matrix)
    hat_user_A_matrix.scatter_(-1, indices, 1)
    if binary:
        Final_user_A_matrix = (hat_user_A_matrix - normalized_user_A_matrix).detach() + normalized_user_A_matrix #(batchsize,usenum,)
    else:
        Final_user_A_matrix = normalized_user_A_matrix
    # model_output = beamforming_vectors_input
    model_output = beamforming_vectors_input * Final_user_A_matrix.view(*Final_user_A_matrix.shape, 1, 1, 1)
    commitment_loss = -torch.mean(torch.sum(normalized_user_A_matrix ** 2, dim=-1))

    complex_beamforming_vectors = torch.complex(model_output[..., 0], model_output[..., 1]).view(batchsize, receiver_num, bs_number,
                                                                                                 max_antenna_x * max_antenna_z)
    squared_norms = torch.sum(torch.abs(complex_beamforming_vectors) ** 2, dim=-1)
    total_power_per_batch_bs = torch.sum(squared_norms, dim=1, keepdim=True)  # Summing over receivers
    safe_denominator = torch.where(total_power_per_batch_bs > 0, total_power_per_batch_bs, torch.ones_like(total_power_per_batch_bs) * 0.001)
    P_k = torch.from_numpy(np.array(array_power)).view(1, 1, -1).to(device)
    scaling_factors = torch.sqrt(P_k / safe_denominator).to(torch.complex64)
    scaled_beamforming_vectors = (complex_beamforming_vectors * scaling_factors.unsqueeze(-1)).permute([0, 2, 3, 1])
    x_grid, z_grid = generate_position_grid(antenna_array, d, device, bs_num)
    phase_shifts_x = (2 * torch.pi / lambda_) * x_grid.view(1, 1, 1, 1, *x_grid.shape) * torch.sin(
        theta_angles.view(*theta_angles.shape, 1, 1)) * torch.cos(
        phi_angles.view(*phi_angles.shape, 1, 1))
    phase_shifts_z = (2 * torch.pi / lambda_) * z_grid.view(1, 1, 1, 1, *z_grid.shape) * torch.cos(theta_angles.view(*theta_angles.shape, 1, 1))
    phase_shifts = phase_shifts_x + phase_shifts_z
    phase_shifts_e = torch.exp(1j * phase_shifts).view(*phase_shifts.shape[:-2], max_antenna_x * max_antenna_z)  #

    applitude_abs = torch.sum(torch.sum(torch.abs(applitude_complex), dim=3), dim=2)  # (batchsize,receiver_num),
    channel = applitude_complex.to(device).unsqueeze(-1) * phase_shifts_e
    channel_multipath = torch.sum(channel, dim=-2).permute(
        [0, 2, 1, 3])  # batch_size,bs station, user_number , antenna_number

    final = torch.abs(channel_multipath.to(torch.complex64) @ scaled_beamforming_vectors.to(torch.complex64) * scalor_factor)**2
    diagonal_values = torch.diagonal(final, dim1=-2, dim2=-1)
    inference_and_signal = torch.sum(final, dim=-1)
    original_diagonal_values = diagonal_values.permute(0, 2, 1) * Final_user_A_matrix
    originalinference = torch.sum(inference_and_signal.permute(0, 2, 1) - original_diagonal_values, dim=2)
    power_signal = torch.sum(original_diagonal_values, dim=2)
    SINR = power_signal / (originalinference + noise_power)
    rate = torch.log2(1 + SINR)
    return rate, applitude_abs, commitment_loss


def cal_SINR_server3(input_arrays, beamforming_vectors_input, antenna_array,device,total_power,user_A_matrix,binary=0):
    bs_num, lambda_ = 4, 1
    applitude_complex = torch.complex(input_arrays[..., 0], input_arrays[..., 1])
    phi_angles, theta_angles = input_arrays[..., 5], input_arrays[..., 6]
    max_antenna_x, max_antenna_z = np.max(antenna_array[:, 0]), np.max(antenna_array[:, 1])
    array_power = np.array([x[0] * x[1] for x in antenna_array])
    array_power = total_power / sum(array_power) * array_power
    batchsize, receiver_num, bs_number, _, _, _ = beamforming_vectors_input.size()
    d = lambda_ / 2  # Half-wavelength spacing
    scalor_factor, noise_power = math.sqrt(10 ** 8), 1

    choose_method="abs"
    normalized_user_A_matrix = normalize_user_matrix(user_A_matrix, choose_method)
    indices = torch.argmax(normalized_user_A_matrix, dim=-1, keepdim=True)
    hat_user_A_matrix = torch.zeros_like(user_A_matrix)
    hat_user_A_matrix.scatter_(-1, indices, 1)
    if binary:
        Final_user_A_matrix =hat_user_A_matrix  # (batchsize,usenum,)
    else:
        Final_user_A_matrix = normalized_user_A_matrix

    model_output=beamforming_vectors_input*Final_user_A_matrix.view(*Final_user_A_matrix.shape,1,1,1)
    commitment_loss = -torch.mean(torch.sum(Final_user_A_matrix ** 2,dim=-1))

    complex_beamforming_vectors=torch.complex(model_output[...,0],model_output[...,1]).view( batchsize, receiver_num, bs_number, max_antenna_x*max_antenna_z)
    squared_norms = torch.sum(torch.abs(complex_beamforming_vectors) ** 2, dim=-1)
    total_power_per_batch_bs = torch.sum(squared_norms, dim=1, keepdim=True)  # Summing over receivers
    safe_denominator = torch.where(total_power_per_batch_bs > 0, total_power_per_batch_bs,torch.ones_like(total_power_per_batch_bs)*0.001)
    P_k = torch.from_numpy(np.array(array_power)).view(1, 1, -1).to(device)
    scaling_factors = torch.sqrt(P_k / safe_denominator).to(torch.complex64)
    scaled_beamforming_vectors = (complex_beamforming_vectors * scaling_factors.unsqueeze(-1)).permute([0, 2, 3, 1])

    x_grid, z_grid =generate_position_grid(antenna_array, d, device,bs_num)

    phase_shifts_x = (2 * torch.pi / lambda_) *  x_grid.view(1, 1, 1, 1, *x_grid.shape)* torch.sin(theta_angles.view(*theta_angles.shape,1,1)) * torch.cos(
        phi_angles.view(*phi_angles.shape,1,1))
    phase_shifts_z = (2 * torch.pi / lambda_) * z_grid.view(1, 1, 1, 1, *z_grid.shape) * torch.cos(theta_angles.view(*theta_angles.shape,1,1))
    phase_shifts=phase_shifts_x+phase_shifts_z
    phase_shifts_e = torch.exp(1j * phase_shifts).view(*phase_shifts.shape[:-2],max_antenna_x*max_antenna_z)  #

    applitude_abs = torch.sum(torch.sum(torch.abs(applitude_complex), dim=3), dim=2)  # (batchsize,receiver_num),
    channel = applitude_complex.to(device).unsqueeze(-1) * phase_shifts_e
    channel_multipath = torch.sum(channel, dim=-2).permute(
        [0, 2, 1, 3])  # batch_size,bs station, user_number , antenna_number
    # (batchsize,receiver_num, bs_num,number of multipath, 16),
    final = torch.abs(channel_multipath.to(torch.complex64) @ scaled_beamforming_vectors.to(torch.complex64) * scalor_factor)**2
    original_diagonal_values = torch.diagonal(final, dim1=-2, dim2=-1)
    inference_and_signal = torch.sum(torch.sum(final, dim=-1),dim=1,keepdim=True)-original_diagonal_values
    SINR = original_diagonal_values / (inference_and_signal + noise_power)

    rate=torch.sum(torch.log2(1 + SINR)*Final_user_A_matrix.permute(0,2,1),dim=1)
    return rate,applitude_abs,commitment_loss



def cal_SINR_server5(input_arrays, beamforming_vectors_input, antenna_array,device,total_power,user_A_matrix,binary=0):
    bs_num=4
    applitude_complex = torch.complex(input_arrays[..., 0], input_arrays[..., 1])
    phi_angles = input_arrays[..., 5]
    theta_angles = input_arrays[..., 6]

    max_antenna_x,max_antenna_z=np.max(antenna_array[:,0]),np.max(antenna_array[:,1])
    array_power=np.array( [x[0] * x[1] for x in antenna_array])
    array_power = total_power / sum(array_power) * array_power
    batchsize, receiver_num, bs_number, _,_,_=beamforming_vectors_input.size() #.view(batchsize, car_max_num,bs_num, max_antenna_x,max_antenna_z, 2)
    lambda_ = 1  # Wavelength, assuming 1 for simplicity
    d = lambda_ / 2  # Half-wavelength spacing
    scalor_factor = math.sqrt(10 ** 8)
    noise_power =1
    choose_method="abs"
    normalized_user_A_matrix = normalize_user_matrix(user_A_matrix, choose_method)
    indices = torch.argmax(normalized_user_A_matrix, dim=-1, keepdim=True)
    hat_user_A_matrix = torch.zeros_like(normalized_user_A_matrix)
    hat_user_A_matrix.scatter_(-1, indices, 1)
    if binary:
        Final_user_A_matrix = (hat_user_A_matrix - normalized_user_A_matrix).detach() + normalized_user_A_matrix  # (batchsize,usenum,)
    else:
        Final_user_A_matrix = normalized_user_A_matrix

    # model_output=beamforming_vectors_input.to(device)
    model_output=beamforming_vectors_input*Final_user_A_matrix.view(*Final_user_A_matrix.shape,1,1,1)
    commitment_loss = torch.mean(torch.sum((hat_user_A_matrix - normalized_user_A_matrix) ** 2,dim=-1))
    complex_beamforming_vectors=torch.complex(model_output[...,0],model_output[...,1]).view( batchsize, receiver_num, bs_number, max_antenna_x*max_antenna_z)
    squared_norms = torch.sum(torch.abs(complex_beamforming_vectors) ** 2, dim=-1)
    total_power_per_batch_bs = torch.sum(squared_norms, dim=1, keepdim=True)  # Summing over receivers
    safe_denominator = torch.where(total_power_per_batch_bs > 0, total_power_per_batch_bs,torch.ones_like(total_power_per_batch_bs)*0.001)
    P_k = torch.from_numpy(np.array(array_power)).view(1, 1, -1).to(device)
    scaling_factors = torch.sqrt(P_k / safe_denominator).to(torch.complex64)
    scaled_beamforming_vectors = (complex_beamforming_vectors * scaling_factors.unsqueeze(-1)).permute([0, 2, 3, 1])

    x_grid, z_grid =generate_position_grid(antenna_array, d, device,bs_num)

    phase_shifts_x = (2 * torch.pi / lambda_) *  x_grid.view(1, 1, 1, 1, *x_grid.shape)* torch.sin(theta_angles.view(*theta_angles.shape,1,1)) * torch.cos(
        phi_angles.view(*phi_angles.shape,1,1))
    phase_shifts_z = (2 * torch.pi / lambda_) * z_grid.view(1, 1, 1, 1, *z_grid.shape) * torch.cos(theta_angles.view(*theta_angles.shape,1,1))
    phase_shifts=phase_shifts_x+phase_shifts_z
    phase_shifts_e = torch.exp(1j * phase_shifts).view(*phase_shifts.shape[:-2],max_antenna_x*max_antenna_z)  #

    applitude_abs = torch.sum(torch.sum(torch.abs(applitude_complex), dim=3), dim=2)  # (batchsize,receiver_num),
    channel = applitude_complex.to(device).unsqueeze(-1) * phase_shifts_e
    channel_multipath = torch.sum(channel, dim=-2).permute(
        [0, 2, 1, 3])  # batch_size,bs station, user_number , antenna_number
    final = torch.abs(channel_multipath.to(torch.complex64) @ scaled_beamforming_vectors.to(torch.complex64) * scalor_factor)**2
    original_diagonal_values = torch.diagonal(final, dim1=-2, dim2=-1)
    inference_and_signal = torch.sum(torch.sum(final, dim=-1),dim=1,keepdim=True)-original_diagonal_values
    SINR = original_diagonal_values / (inference_and_signal + noise_power)
    rate=torch.sum(torch.log2(1 + SINR),dim=1)
    return rate,applitude_abs,commitment_loss

def cal_SINR_server7(input_arrays, beamforming_vectors_input, antenna_array,device,total_power,user_A_matrix,binary=0):
    bs_num, lambda_ = 4, 1
    applitude_complex = torch.complex(input_arrays[..., 0], input_arrays[..., 1])
    phi_angles, theta_angles = input_arrays[..., 5], input_arrays[..., 6]
    max_antenna_x, max_antenna_z = np.max(antenna_array[:, 0]), np.max(antenna_array[:, 1])
    array_power = np.array([x[0] * x[1] for x in antenna_array])
    array_power = total_power / sum(array_power) * array_power
    batchsize, receiver_num, bs_number, _, _, _ = beamforming_vectors_input.size()
    d = lambda_ / 2  # Half-wavelength spacing
    scalor_factor, noise_power = math.sqrt(10 ** 8), 1

    choose_method="abs"
    normalized_user_A_matrix = normalize_user_matrix(user_A_matrix, choose_method)
    indices = torch.argmax(normalized_user_A_matrix, dim=-1, keepdim=True)
    hat_user_A_matrix = torch.zeros_like(normalized_user_A_matrix)
    hat_user_A_matrix.scatter_(-1, indices, 1)
    if binary:
        Final_user_A_matrix = (hat_user_A_matrix - normalized_user_A_matrix).detach() + normalized_user_A_matrix  # (batchsize,usenum,)
    else:
        Final_user_A_matrix = normalized_user_A_matrix
    # model_output=beamforming_vectors_input.to(device)
    model_output=beamforming_vectors_input*Final_user_A_matrix.view(*Final_user_A_matrix.shape,1,1,1)
    commitment_loss = torch.mean(torch.sum((hat_user_A_matrix - normalized_user_A_matrix) ** 2,dim=-1))

    complex_beamforming_vectors=torch.complex(model_output[...,0],model_output[...,1]).view( batchsize, receiver_num, bs_number, max_antenna_x*max_antenna_z)
    squared_norms = torch.sum(torch.abs(complex_beamforming_vectors) ** 2, dim=-1)
    total_power_per_batch_bs = torch.sum(squared_norms, dim=1, keepdim=True)  # Summing over receivers
    safe_denominator = torch.where(total_power_per_batch_bs > 0, total_power_per_batch_bs,torch.ones_like(total_power_per_batch_bs)*0.001)
    P_k = torch.from_numpy(np.array(array_power)).view(1, 1, -1).to(device)
    scaling_factors = torch.sqrt(P_k / safe_denominator).to(torch.complex64)
    scaled_beamforming_vectors = (complex_beamforming_vectors * scaling_factors.unsqueeze(-1)).permute([0, 2, 3, 1])

    x_grid, z_grid =generate_position_grid(antenna_array, d, device,bs_num)

    phase_shifts_x = (2 * torch.pi / lambda_) *  x_grid.view(1, 1, 1, 1, *x_grid.shape)* torch.sin(theta_angles.view(*theta_angles.shape,1,1)) * torch.cos(
        phi_angles.view(*phi_angles.shape,1,1))
    phase_shifts_z = (2 * torch.pi / lambda_) * z_grid.view(1, 1, 1, 1, *z_grid.shape) * torch.cos(theta_angles.view(*theta_angles.shape,1,1))
    phase_shifts=phase_shifts_x+phase_shifts_z
    phase_shifts_e = torch.exp(1j * phase_shifts).view(*phase_shifts.shape[:-2],max_antenna_x*max_antenna_z)  #

    applitude_abs = torch.sum(torch.sum(torch.abs(applitude_complex), dim=3), dim=2)  # (batchsize,receiver_num),
    channel = applitude_complex.to(device).unsqueeze(-1) * phase_shifts_e
    channel_multipath = torch.sum(channel, dim=-2).permute(
        [0, 2, 1, 3])  # batch_size,bs station, user_number , antenna_number
    final = torch.abs(channel_multipath.to(torch.complex64) @ scaled_beamforming_vectors.to(torch.complex64) * scalor_factor)**2
    original_diagonal_values = torch.diagonal(final, dim1=-2, dim2=-1)
    inference_and_signal = torch.sum(torch.sum(final, dim=-1),dim=1,keepdim=True)-original_diagonal_values
    SINR = original_diagonal_values / (inference_and_signal + noise_power)
    rate=torch.sum(torch.log2(1 + SINR)*Final_user_A_matrix.permute(0,2,1),dim=1)
    return SINR,rate,applitude_abs,commitment_loss





def calculate_received_signal_strength_for_batches2(input_arrays, beamforming_vectors_input, antenna_array,device,total_power,user_A_matrix):
    """
    Calculate the received signal strength for batches of scenarios, where each input array
    corresponds to a specific beamforming vector.

    Parameters:
    - input_arrays:  arrays, each of shape (batchsize,receiver_num, bs_num,number of multipath, 7), where each array corresponds to a specific scenario
    - beamforming_vectors:  PyTorch tensors, the value is (batchsize,receiver_num,bs_number,antenna_number), corresponding one-to-one with the input arrays
    - antenna_vectors: Antenna configuration vectors, representing the physical configuration of each antenna element
    - beamforming_vectors_list size:[batch_size,bs_num,antenna_num]
    -input_arrays size:a list of tensors,len(input_arrays)=batch_size,input_arrays[0].size()=[receiver_num,bs_num,mp_num,7]
    Returns:
    - all_received_signal_strengths: A list of lists, containing the received signal strengths for each receiver and multipath component for each scenario
    """
    # Constants
    # rate,applitude_abs,commitment_loss=cal_SINR_server6(input_arrays, beamforming_vectors_input, antenna_array,device,total_power,user_A_matrix)
    rate1,applitude_abs1,commitment_loss1=cal_SINR_server_new(input_arrays, beamforming_vectors_input, antenna_array,device,total_power,user_A_matrix)
    rate2, applitude_abs2, commitment_loss2=cal_SINR_server_new2(input_arrays, beamforming_vectors_input, antenna_array,device,total_power,user_A_matrix)


    loss= torch.min(rate1, dim=1).values
    rate=rate1

    # Find the minimum value among positive values along dimension 1
    min_values = torch.min(rate, dim=1).values
    # masked_tensor2 = torch.where(applitude_abs > 0, rate2, torch.full_like(rate, float('inf')))
    min_values2 = torch.min(rate2, dim=1).values
    skip_training=~torch.all(torch.isfinite(min_values))
    zero_num=torch.sum(min_values==0)
    zero_num2=torch.sum(min_values2==0)
    return loss,min_values,min_values2,skip_training,commitment_loss1,zero_num,zero_num2


def cal_SINR_server_new(input_arrays, beam, antenna_array,device,total_power,user_A_matrix):
    bs_num=4
    applitude_complex = torch.complex(input_arrays[..., 0], input_arrays[..., 1])
    phi_angles = input_arrays[..., 5]
    theta_angles = input_arrays[..., 6]

    max_antenna_x,max_antenna_z=np.max(antenna_array[:,0]),np.max(antenna_array[:,1])
    array_power=np.array( [x[0] * x[1] for x in antenna_array])
    array_power = total_power / sum(array_power) * array_power
    batchsize, receiver_num, _, _,_,_=beam.size() #.view(batchsize, car_max_num,bs_num, max_antenna_x,max_antenna_z, 2)
    lambda_ = 1  # Wavelength, assuming 1 for simplicity
    d = lambda_ / 2  # Half-wavelength spacing
    scalor_factor = math.sqrt(10 ** 8)
    noise_power =1
    choose_method="abs"

    normalized_user_A_matrix = normalize_user_matrix(user_A_matrix, choose_method)
    indices = torch.argmax(normalized_user_A_matrix, dim=-1, keepdim=True)
    hat_user_A_matrix = torch.zeros_like(normalized_user_A_matrix)
    hat_user_A_matrix.scatter_(-1, indices, 1)
    commitment_loss = torch.mean(torch.sum((hat_user_A_matrix - normalized_user_A_matrix) ** 2,dim=-1))
    Final_user_A_matrix=normalized_user_A_matrix

    complex_beam=torch.complex(beam[...,0],beam[...,1]).view( batchsize, receiver_num, 1, max_antenna_x*max_antenna_z).to(device)
    separate_beam = complex_beam * Final_user_A_matrix.unsqueeze(-1)
    total_power_per_batch_bs =  torch.sum(torch.sum(torch.abs(separate_beam) ** 2, dim=-1), dim=1, keepdim=True)
    safe_denominator = torch.where(total_power_per_batch_bs > 0, total_power_per_batch_bs,torch.ones_like(total_power_per_batch_bs))
    P_k = torch.from_numpy(np.array(array_power)).view(1, 1, -1).to(device)
    scaling_factors = torch.sqrt(P_k / safe_denominator).to(torch.complex64)
    scaled_beamforming_vectors = (separate_beam * scaling_factors.unsqueeze(-1)).permute([0, 2, 3, 1])
    x_grid, z_grid =generate_position_grid(antenna_array, d, device,bs_num)
    phase_shifts_x = (2 * torch.pi / lambda_) *  x_grid.view(1, 1, 1, 1, *x_grid.shape)* torch.sin(theta_angles.view(*theta_angles.shape,1,1)) * torch.cos(
        phi_angles.view(*phi_angles.shape,1,1))
    phase_shifts_z = (2 * torch.pi / lambda_) * z_grid.view(1, 1, 1, 1, *z_grid.shape) * torch.cos(theta_angles.view(*theta_angles.shape,1,1))
    phase_shifts=phase_shifts_x+phase_shifts_z
    phase_shifts_e = torch.exp(1j * phase_shifts).view(*phase_shifts.shape[:-2],max_antenna_x*max_antenna_z)  #
    applitude_abs = torch.sum(torch.sum(torch.abs(applitude_complex), dim=3), dim=2)  # (batchsize,receiver_num),
    channel = applitude_complex.to(device).unsqueeze(-1) * phase_shifts_e
    channel_multipath = torch.sum(channel, dim=-2).permute(
        [0, 2, 1, 3])  # batch_size,bs station, user_number , antenna_number
    # final = torch.abs(channel_multipath.to(torch.complex64) @ scaled_beamforming_vectors.to(torch.complex64) * scalor_factor) ** 2
    # diagonal_values = torch.diagonal(final, dim1=-2, dim2=-1)
    # inference_and_signal = torch.sum(final, dim=-1)
    # # original_diagonal_values = diagonal_values.permute(0, 2, 1) * Final_user_A_matrix
    # original_diagonal_values = torch.sum(diagonal_values,dim=1)
    # originalinference = torch.sum(inference_and_signal, dim=1)-original_diagonal_values
    # SINR =original_diagonal_values / (originalinference + noise_power)
    # rate=torch.log2(1 + SINR)

    final = torch.abs(channel_multipath.to(torch.complex64) @ scaled_beamforming_vectors.to(torch.complex64) * scalor_factor)**2
    original_diagonal_values = torch.diagonal(final, dim1=-2, dim2=-1)
    inference_and_signal = torch.sum(torch.sum(final, dim=-1),dim=1,keepdim=True)-original_diagonal_values
    SINR = original_diagonal_values / (inference_and_signal + noise_power)
    rate=torch.sum(torch.log2(1 + SINR)*Final_user_A_matrix.permute(0,2,1),dim=1)

    return rate,applitude_abs,commitment_loss



def cal_SINR_server_new2(input_arrays, beam, antenna_array,device,total_power,user_A_matrix):
    bs_num=4
    applitude_complex = torch.complex(input_arrays[..., 0], input_arrays[..., 1])
    phi_angles = input_arrays[..., 5]
    theta_angles = input_arrays[..., 6]

    max_antenna_x,max_antenna_z=np.max(antenna_array[:,0]),np.max(antenna_array[:,1])
    array_power=np.array( [x[0] * x[1] for x in antenna_array])
    array_power = total_power / sum(array_power) * array_power
    batchsize, receiver_num, _, _,_,_=beam.size() #.view(batchsize, car_max_num,bs_num, max_antenna_x,max_antenna_z, 2)
    lambda_ = 1  # Wavelength, assuming 1 for simplicity
    d = lambda_ / 2  # Half-wavelength spacing
    scalor_factor = math.sqrt(10 ** 8)
    noise_power =1
    choose_method="abs"

    normalized_user_A_matrix = normalize_user_matrix(user_A_matrix, choose_method)
    indices = torch.argmax(normalized_user_A_matrix, dim=-1, keepdim=True)
    hat_user_A_matrix = torch.zeros_like(normalized_user_A_matrix)
    hat_user_A_matrix.scatter_(-1, indices, 1)
    commitment_loss = torch.mean(torch.sum((hat_user_A_matrix - normalized_user_A_matrix) ** 2,dim=-1))
    Final_user_A_matrix=hat_user_A_matrix

    complex_beam=torch.complex(beam[...,0],beam[...,1]).view( batchsize, receiver_num, 1, max_antenna_x*max_antenna_z).to(device)
    separate_beam = complex_beam * Final_user_A_matrix.unsqueeze(-1)
    total_power_per_batch_bs =  torch.sum(torch.sum(torch.abs(separate_beam) ** 2, dim=-1), dim=1, keepdim=True)
    safe_denominator = torch.where(total_power_per_batch_bs > 0, total_power_per_batch_bs,torch.ones_like(total_power_per_batch_bs))
    P_k = torch.from_numpy(np.array(array_power)).view(1, 1, -1).to(device)
    scaling_factors = torch.sqrt(P_k / safe_denominator).to(torch.complex64)
    scaled_beamforming_vectors = (separate_beam * scaling_factors.unsqueeze(-1)).permute([0, 2, 3, 1])
    x_grid, z_grid =generate_position_grid(antenna_array, d, device,bs_num)
    phase_shifts_x = (2 * torch.pi / lambda_) *  x_grid.view(1, 1, 1, 1, *x_grid.shape)* torch.sin(theta_angles.view(*theta_angles.shape,1,1)) * torch.cos(
        phi_angles.view(*phi_angles.shape,1,1))
    phase_shifts_z = (2 * torch.pi / lambda_) * z_grid.view(1, 1, 1, 1, *z_grid.shape) * torch.cos(theta_angles.view(*theta_angles.shape,1,1))
    phase_shifts=phase_shifts_x+phase_shifts_z
    phase_shifts_e = torch.exp(1j * phase_shifts).view(*phase_shifts.shape[:-2],max_antenna_x*max_antenna_z)  #

    applitude_abs = torch.sum(torch.sum(torch.abs(applitude_complex), dim=3), dim=2)  # (batchsize,receiver_num),
    channel = applitude_complex.to(device).unsqueeze(-1) * phase_shifts_e
    channel_multipath = torch.sum(channel, dim=-2).permute(
        [0, 2, 1, 3])  # batch_size,bs station, user_number , antenna_number
    final = torch.abs(channel_multipath.to(torch.complex64) @ scaled_beamforming_vectors.to(torch.complex64) * scalor_factor)**2
    original_diagonal_values = torch.diagonal(final, dim1=-2, dim2=-1)
    inference_and_signal = torch.sum(torch.sum(final, dim=-1),dim=1,keepdim=True)-original_diagonal_values
    SINR = original_diagonal_values / (inference_and_signal + noise_power)
    rate=torch.sum(torch.log2(1 + SINR)*Final_user_A_matrix.permute(0,2,1),dim=1)
    return rate,applitude_abs,commitment_loss



def uplink_pilot_signal(input_arrays, antenna_array,device,SNR,pilot_num,no_noise=False):
    """
        - input_arrays:  arrays, each of shape (batchsize,receiver_num, bs_num,number of multipath, 9), where each array corresponds to a specific scenario

    """

    lambda_=4
    noise_power, snr_linear = 1, 10 ** (SNR / 10)
    max_antenna_x,max_antenna_z=np.max(antenna_array[:,0]),np.max(antenna_array[:,1])
    batchsize, receiver_num, bs_num, path_num,_=input_arrays.size()
    d = lambda_ / 2  # Half-wavelength spacing
    phi_angles, theta_angles = input_arrays[..., 5], input_arrays[..., 6]
    scalor_factor=math.sqrt(snr_linear)
    antenna_array=np.reshape(antenna_array,[bs_num,-1])
    applitude_complex = torch.complex(input_arrays[..., 0], input_arrays[..., 1])
    x_grid, z_grid = generate_position_grid(antenna_array, d, device, bs_num)

    phi_angles, theta_angles = input_arrays[..., 5], input_arrays[..., 6]
    sin_theta = torch.sin(theta_angles).view(*theta_angles.shape, 1, 1)
    cos_phi = torch.cos(phi_angles).view(*phi_angles.shape, 1, 1)
    cos_theta = torch.cos(theta_angles).view(*theta_angles.shape, 1, 1)

    phase_shifts_x = (2 * torch.pi / lambda_) * x_grid.view(1, 1, 1, 1, *x_grid.shape) * sin_theta * cos_phi
    phase_shifts_z = (2 * torch.pi / lambda_) * z_grid.view(1, 1, 1, 1, *z_grid.shape) * cos_theta
    phase_shifts = phase_shifts_x + phase_shifts_z

    phase_shifts_e = torch.exp(1j * phase_shifts).view(*phase_shifts.shape[:-2],max_antenna_x*max_antenna_z)  #

    channel = applitude_complex.to(device).unsqueeze(-1) * phase_shifts_e
    # sum -2 may sum all the multipath signal?
    # channel_multipath = torch.sum(channel, dim=-2).permute([0, 2, 1, 3])*scalor_factor  # batch_size,bs station, user_number , antenna_number
    channel_multipath = torch.sum(channel, dim=-2)*scalor_factor  # batch_size,user_number,bs station , antenna_number
    channel_multipath_multiple_plot=channel_multipath.unsqueeze(2).repeat_interleave(pilot_num, dim=2) # batch_size,user_number,pilotnum,bs station , antenna_number
    # noise = 1 / math.sqrt(2.0) * (torch.randn(channel_multipath_multiple_plot.size(), device=device) + 1j * torch.randn(channel_multipath_multiple_plot.size(), device=device)) * noise_power
    noise = 1 / math.sqrt(2.0) * (torch.randn(channel_multipath_multiple_plot.size(), device=device) + 1j * torch.randn(channel_multipath_multiple_plot.size(), device=device)) * noise_power

    # channel_noise=channel_multipath_multiple_plot+noise
    channel_noise=channel_multipath_multiple_plot+noise.to(device)
    if no_noise:
        channel_noise=channel_multipath_multiple_plot
    channel_reshape=channel_noise.contiguous().view(batchsize,receiver_num,-1).to(device)
    channel_final = torch.cat([torch.real(channel_reshape), torch.imag(channel_reshape)],dim=-1)
    return channel_final



def uplink_pilot_signal2(input_arrays, antenna_array,device,SNR,pilot_num):
    """
        - input_arrays:  arrays, each of shape (batchsize,receiver_num, bs_num,number of multipath, 9), where each array corresponds to a specific scenario

    """

    lambda_=4
    noise_power, snr_linear = 1, 10 ** (SNR / 10)
    max_antenna_x,max_antenna_z=np.max(antenna_array[:,0]),np.max(antenna_array[:,1])
    batchsize, receiver_num, bs_num, path_num,_=input_arrays.size()
    d = lambda_ / 2  # Half-wavelength spacing
    phi_angles, theta_angles = input_arrays[..., 5], input_arrays[..., 6]
    scalor_factor=math.sqrt(noise_power*snr_linear)
    antenna_array=np.reshape(antenna_array,[bs_num,-1])
    applitude_complex = torch.complex(input_arrays[..., 0], input_arrays[..., 1])


    x_grid, z_grid = generate_position_grid(antenna_array, d, device, bs_num)
    phase_shifts_x = (2 * torch.pi / lambda_) *  x_grid.view(1, 1, 1, 1, *x_grid.shape)* torch.sin(theta_angles.view(*theta_angles.shape,1,1)) * torch.cos(
        phi_angles.view(*phi_angles.shape,1,1))
    phase_shifts_z = (2 * torch.pi / lambda_) * z_grid.view(1, 1, 1, 1, *z_grid.shape) * torch.cos(theta_angles.view(*theta_angles.shape,1,1))
    phase_shifts=phase_shifts_x+phase_shifts_z
    phase_shifts_e = torch.exp(1j * phase_shifts).view(*phase_shifts.shape[:-2],max_antenna_x*max_antenna_z)  #

    channel = applitude_complex.to(device).unsqueeze(-1) * phase_shifts_e
    # sum -2 may sum all the multipath signal?
    # channel_multipath = torch.sum(channel, dim=-2).permute([0, 2, 1, 3])*scalor_factor  # batch_size,bs station, user_number , antenna_number
    channel_multipath = torch.sum(channel, dim=-2)*scalor_factor  # batch_size,user_number,bs station , antenna_number
    channel_multipath_multiple_plot=channel_multipath.unsqueeze(2).repeat_interleave(pilot_num, dim=2) # batch_size,user_number,pilotnum,bs station , antenna_number
    noise = 1 / math.sqrt(2.0) * (torch.randn(channel_multipath_multiple_plot.size(), device=device) + 1j * torch.randn(channel_multipath_multiple_plot.size(), device=device)) * noise_power
    channel_noise=channel_multipath_multiple_plot+noise.to(device)
    channel_reshape=channel_noise.contiguous().view(batchsize,receiver_num,-1).to(device)
    channel_final = torch.cat([torch.real(channel_reshape), torch.imag(channel_reshape)],dim=-1)
    return channel_final





def uplink_pilot_signal_new222(input_arrays, antenna_array,device,SNR,pilot_num):
    """
        - input_arrays:  arrays, each of shape (batchsize,receiver_num, bs_num,number of multipath, 9), where each array corresponds to a specific scenario

    """

    lambda_=4
    noise_power, snr_linear = 1, 10 ** (SNR / 10)
    max_antenna_x,max_antenna_z=np.max(antenna_array[:,0]),np.max(antenna_array[:,1])
    batchsize, receiver_num, bs_num, path_num,_=input_arrays.size()
    d = lambda_ / 2  # Half-wavelength spacing
    phi_angles, theta_angles = input_arrays[..., 5], input_arrays[..., 6]
    scalor_factor=math.sqrt(noise_power*snr_linear)
    antenna_array=np.reshape(antenna_array,[bs_num,-1])
    applitude_complex = torch.complex(input_arrays[..., 0], input_arrays[..., 1])


    x_grid, z_grid = generate_position_grid(antenna_array, d, device, bs_num)
    phase_shifts_x = (2 * torch.pi / lambda_) *  x_grid.view(1, 1, 1, 1, *x_grid.shape)* torch.sin(theta_angles.view(*theta_angles.shape,1,1)) * torch.cos(
        phi_angles.view(*phi_angles.shape,1,1))
    phase_shifts_z = (2 * torch.pi / lambda_) * z_grid.view(1, 1, 1, 1, *z_grid.shape) * torch.cos(theta_angles.view(*theta_angles.shape,1,1))
    phase_shifts=phase_shifts_x+phase_shifts_z
    phase_shifts_e = torch.exp(1j * phase_shifts).view(*phase_shifts.shape[:-2],max_antenna_x*max_antenna_z)  #

    channel = applitude_complex.to(device).unsqueeze(-1) * phase_shifts_e
    # sum -2 may sum all the multipath signal?
    # channel_multipath = torch.sum(channel, dim=-2).permute([0, 2, 1, 3])*scalor_factor  # batch_size,bs station, user_number , antenna_number
    channel_multipath = torch.sum(channel, dim=-2)*scalor_factor  # batch_size,user_number,bs station , antenna_number
    channel_multipath_multiple_plot=channel_multipath.unsqueeze(2).repeat_interleave(pilot_num, dim=2) # batch_size,user_number,pilotnum,bs station , antenna_number
    noise=1/math.sqrt(2.0)*(torch.randn(channel_multipath_multiple_plot.size(),device=device)+1j*torch.randn(channel_multipath_multiple_plot.size(),device=device))*noise_power
    channel_noise=channel_multipath_multiple_plot+noise.to(device)
    channel_reshape=channel_noise.contiguous().view(batchsize,receiver_num,-1).to(device)
    channel_final = torch.cat([torch.real(channel_reshape), torch.imag(channel_reshape)],dim=-1)
    return channel_final


def uplink_pilot_signal_new(input_arrays, antenna_array, device, SNR, pilot_num,pilot_tensor , no_noise=0):
    """
        - input_arrays:  arrays, each of shape (batchsize,receiver_num, bs_num,number of multipath, 9), where each array corresponds to a specific scenario

    """

    lambda_ = 4
    if SNR == 60:
        noise_power, snr_linear = 1, 10 ** (SNR / 10)
    else:
        snr_linear = 10 ** (60 / 10)
        noise_power = 1 / 10 ** ((SNR - 60) / 10)
    max_antenna_x, max_antenna_z = np.max(antenna_array[:, 0]), np.max(antenna_array[:, 1])
    batchsize, receiver_num, bs_num, path_num, _ = input_arrays.size()
    d = lambda_ / 2  # Half-wavelength spacing
    phi_angles, theta_angles = input_arrays[..., 5], input_arrays[..., 6]
    pilot_length = pilot_num *receiver_num
    scalor_factor = math.sqrt(snr_linear)

    antenna_array = np.reshape(antenna_array, [bs_num, -1])
    applitude_complex = torch.complex(input_arrays[..., 0], input_arrays[..., 1])

    x_grid, z_grid = generate_position_grid(antenna_array, d, device, bs_num)
    phase_shifts_x = (2 * torch.pi / lambda_) * x_grid.view(1, 1, 1, 1, *x_grid.shape) * torch.sin(
        theta_angles.view(*theta_angles.shape, 1, 1)) * torch.cos(
        phi_angles.view(*phi_angles.shape, 1, 1))
    phase_shifts_z = (2 * torch.pi / lambda_) * z_grid.view(1, 1, 1, 1, *z_grid.shape) * torch.cos(theta_angles.view(*theta_angles.shape, 1, 1))
    phase_shifts = phase_shifts_x + phase_shifts_z
    phase_shifts_e = torch.exp(1j * phase_shifts).view(*phase_shifts.shape[:-2], max_antenna_x * max_antenna_z)  #

    channel = applitude_complex.to(device).unsqueeze(-1) * phase_shifts_e
    # sum -2 may sum all the multipath signal?
    # channel_multipath = torch.sum(channel, dim=-2).permute([0, 2, 1, 3])*scalor_factor  # batch_size,bs station, user_number , antenna_number
    channel_multipath = torch.sum(channel, dim=-2) * scalor_factor  # batch_size,user_number,bs station , antenna_number
    # channel_multipath_multiple_plot = channel_multipath.unsqueeze(-1).repeat_interleave(pilot_num*6,
    #                                                                                    dim=-1)  # batch_size,user_number,pilotnum,bs station , antenna_number
    noise_size=[batchsize ,receiver_num, bs_num,max_antenna_x*max_antenna_z,pilot_length]
    noise = 1 / math.sqrt(2.0) * ( torch.randn(noise_size, device=device) + 1j * torch.randn(noise_size, device=device)) * noise_power
    added_noise= noise@pilot_tensor /pilot_length
    if no_noise:
        channel_noise = channel_multipath
    else:
        channel_noise = channel_multipath + added_noise.squeeze(-1).to(device)
    channel_reshape = channel_noise.contiguous().view(batchsize, receiver_num, -1).to(device)
    channel_final = torch.cat([torch.real(channel_reshape), torch.imag(channel_reshape)], dim=-1)
    return channel_final


def Cuplink_pilot_signal_new21(input_arrays, antenna_array, device, SNR, pilot_num,pilot_tensor , no_noise=0):
    """
        - input_arrays:  arrays, each of shape (batchsize,receiver_num, bs_num,number of multipath, 9), where each array corresponds to a specific scenario

    """

    lambda_ = 4
    if SNR == 60:
        noise_power, snr_linear = 1, 10 ** (SNR / 10)
    else:
        snr_linear = 10 ** (60 / 10)
        noise_power = 1 / 10 ** ((SNR - 60) / 10)
    max_antenna_x, max_antenna_z = np.max(antenna_array[:, 0]), np.max(antenna_array[:, 1])
    batchsize, receiver_num, bs_num, path_num, _ = input_arrays.size()
    d = lambda_ / 2  # Half-wavelength spacing
    phi_angles, theta_angles = input_arrays[..., 5], input_arrays[..., 6]
    pilot_length = pilot_num
    scalor_factor = math.sqrt(snr_linear)

    antenna_array = np.reshape(antenna_array, [bs_num, -1])
    applitude_complex = torch.complex(input_arrays[..., 0], input_arrays[..., 1])

    x_grid, z_grid = generate_position_grid(antenna_array, d, device, bs_num)
    phase_shifts_x = (2 * torch.pi / lambda_) * x_grid.view(1, 1, 1, 1, *x_grid.shape) * torch.sin(
        theta_angles.view(*theta_angles.shape, 1, 1)) * torch.cos(
        phi_angles.view(*phi_angles.shape, 1, 1))
    phase_shifts_z = (2 * torch.pi / lambda_) * z_grid.view(1, 1, 1, 1, *z_grid.shape) * torch.cos(theta_angles.view(*theta_angles.shape, 1, 1))
    phase_shifts = phase_shifts_x + phase_shifts_z
    phase_shifts_e = torch.exp(1j * phase_shifts).view(*phase_shifts.shape[:-2], max_antenna_x * max_antenna_z)  #
    channel = applitude_complex.to(device).unsqueeze(-1) * phase_shifts_e
    channel_multipath = torch.sum(channel, dim=-2) * scalor_factor  # batch_size,user_number,bs station , antenna_number
    noise_size=[batchsize ,receiver_num, bs_num,max_antenna_x*max_antenna_z,pilot_length]
    noise = 1 / math.sqrt(2.0) * ( torch.randn(noise_size, device=device) + 1j * torch.randn(noise_size, device=device)) * noise_power
    added_noise= noise@pilot_tensor /pilot_length
    if no_noise:
        channel_noise = channel_multipath
    else:
        channel_noise = channel_multipath + added_noise.squeeze(-1).to(device)
    channel_reshape = channel_noise.contiguous().view(batchsize, receiver_num, -1).to(device)
    channel_final = torch.cat([torch.real(channel_reshape), torch.imag(channel_reshape)], dim=-1)
    return channel_final
    #pilot=torch.conj(pilot_tensor.transpose(-1,-2))
    #pilot_T=pilot_tensor
    #sum_hjxj_add_noise=torch.sum(channel_multipath@pilot,dim=1,keep_dims=True)+noise
    #Y_hat=sum_hjxj_add_noise@pilot

def uplink_pilot_signal_new2(input_arrays, antenna_array, device, SNR, pilot_num,pilot_tensor , no_noise=0):
    """
        - input_arrays:  arrays, each of shape (batchsize,receiver_num, bs_num,number of multipath, 9), where each array corresponds to a specific scenario

    """

    lambda_ = 4
    if SNR == 60:
        noise_power, snr_linear = 1, 10 ** (SNR / 10)
    else:
        snr_linear = 10 ** (60 / 10)
        noise_power = 1 / 10 ** ((SNR - 60) / 10)
    max_antenna_x, max_antenna_z = np.max(antenna_array[:, 0]), np.max(antenna_array[:, 1])
    batchsize, receiver_num, bs_num, path_num, _ = input_arrays.size()
    d = lambda_ / 2  # Half-wavelength spacing
    phi_angles, theta_angles = input_arrays[..., 5], input_arrays[..., 6]
    pilot_length = pilot_num
    scalor_factor = math.sqrt(snr_linear)

    antenna_array = np.reshape(antenna_array, [bs_num, -1])
    applitude_complex = torch.complex(input_arrays[..., 0], input_arrays[..., 1])

    x_grid, z_grid = generate_position_grid(antenna_array, d, device, bs_num)
    phase_shifts_x = (2 * torch.pi / lambda_) * x_grid.view(1, 1, 1, 1, *x_grid.shape) * torch.sin(
        theta_angles.view(*theta_angles.shape, 1, 1)) * torch.cos(
        phi_angles.view(*phi_angles.shape, 1, 1))
    phase_shifts_z = (2 * torch.pi / lambda_) * z_grid.view(1, 1, 1, 1, *z_grid.shape) * torch.cos(theta_angles.view(*theta_angles.shape, 1, 1))
    phase_shifts = phase_shifts_x + phase_shifts_z
    phase_shifts_e = torch.exp(1j * phase_shifts).view(*phase_shifts.shape[:-2], max_antenna_x * max_antenna_z)  #
    channel = applitude_complex.to(device).unsqueeze(-1) * phase_shifts_e
    channel_multipath = torch.sum(channel, dim=-2) * scalor_factor  # batch_size,user_number,bs station , antenna_number
    noise_size=[batchsize ,receiver_num, bs_num,max_antenna_x*max_antenna_z,pilot_length]
    noise = 1 / math.sqrt(2.0) * ( torch.randn(noise_size, device=device) + 1j * torch.randn(noise_size, device=device)) * noise_power
    #added_noise= noise@pilot_tensor /pilot_length
    if no_noise:
        channel_noise = channel_multipath
    else:
        pilot=torch.conj(pilot_tensor.transpose(-1,-2))
        pilot_T=pilot_tensor
        sum_hjxj_add_noise=torch.sum(channel_multipath.to(torch.complex64).unsqueeze(-1)@pilot,dim=1,keepdim=True)+noise
        channel_noise=sum_hjxj_add_noise@pilot_T/pilot_length
    channel_reshape = channel_noise.contiguous().view(batchsize, receiver_num, -1).to(device)
    channel_final = torch.cat([torch.real(channel_reshape), torch.imag(channel_reshape)], dim=-1)
    return channel_final