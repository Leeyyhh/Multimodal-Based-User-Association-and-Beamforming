import math

import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

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




def construct_car_size2():
    data=[[0, 4.181210041046143, 1.9941171407699585, 1.385296106338501],
     [1, 4.1928300857543945, 1.8161858320236206, 1.4738311767578125],
    [2, 5.0078253746032715, 1.8816219568252563, 1.5347249507904053],
     [3, 3.987684965133667, 1.8508483171463013, 1.6171095371246338],
    [4, 4.513522624969482, 2.006814479827881, 1.5248334407806396],
    [5, 4.673638820648193, 1.8118125200271606, 1.441947340965271],
     [6, 4.717525005340576, 1.894826889038086, 1.300939917564392],
     [7, 3.805800199508667, 1.970275640487671, 1.4750301837921143],
     [8, 5.0267767906188965, 2.1515462398529053, 1.6506516933441162],
     [9, 5.357479572296143, 2.033202886581421, 1.4106587171554565],
     [10, 3.705369472503662, 1.788678526878357, 1.549050211906433],
     [11, 4.552699089050293, 2.097072124481201, 1.7671663761138916]]
    vehicle_dimensions = np.array(data)
    vehicle_dimensions[:,1:]=vehicle_dimensions[:,1:]

    return vehicle_dimensions
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



def calculate_received_signal_strength_for_batches(input_arrays, beamforming_vectors_input, antenna_array, device, power_array, user_A_matrix,epoch):

    rate, amplitude_abs, commitment_loss = cal_SINR_server(input_arrays, beamforming_vectors_input, antenna_array, device, power_array, user_A_matrix,epoch)
    binary_rate, _, _ = cal_SINR_server(input_arrays, beamforming_vectors_input, antenna_array, device, power_array, user_A_matrix,epoch, 1)

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
    x_positions = (torch.arange(x_max_antenna, dtype=torch.float32) * d).to(device)
    x_positions_new = x_positions - torch.mean(x_positions)

    # Generate and center z positions
    z_positions = (torch.arange(z_max_antenna, dtype=torch.float32) * d).to(device)
    z_positions_new = z_positions - torch.mean(z_positions)

    # Create a mesh grid using the centered positions
    x_grid, z_grid = torch.meshgrid(x_positions_new, z_positions_new, indexing='ij')

    return x_grid, z_grid



def cal_SINR_server(input_arrays, beamforming_vectors_input, antenna_array,device,power_arrary,user_A_matrix,epoch,binary=0):
    discrete_rate_start = 0  # Starting discrete rate
    discrete_rate_end = 1.0  # Final discrete rate
    annealing_epochs = 100  # Number of epochs for full annealing
    discrete_rate = min(discrete_rate_start + (discrete_rate_end - discrete_rate_start) * (epoch / annealing_epochs), discrete_rate_end)
    bs_num, lambda_ = 4, 1
    applitude_complex = torch.complex(input_arrays[..., 0], input_arrays[..., 1])
    phi_angles, theta_angles = input_arrays[..., 5], input_arrays[..., 6]
    max_antenna_x, max_antenna_z = np.max(antenna_array[:, 0]), np.max(antenna_array[:, 1])
    array_power = np.array([x[0] * x[1] for x in antenna_array])
    array_power = power_arrary / sum(array_power) * array_power
    # array_power=np.array(power_arrary)
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
        Final_user_A_matrix =discrete_rate* (hat_user_A_matrix - normalized_user_A_matrix).detach() + normalized_user_A_matrix #(batchsize,usenum,)
        # Final_user_A_matrix = (hat_user_A_matrix - normalized_user_A_matrix).detach() + normalized_user_A_matrix #(batchsize,usenum,)

    # model_output=beamforming_vectors_input*Final_user_A_matrix.view(*Final_user_A_matrix.shape,1,1,1)
    model_output=beamforming_vectors_input
    commitment_loss = torch.mean(torch.sum(torch.abs(Final_user_A_matrix -1/bs_number)** 2,dim=-1))
    complex_beamforming_vectors=torch.complex(model_output[...,0],model_output[...,1]).view( batchsize, receiver_num, bs_number, max_antenna_x*max_antenna_z)
    squared_norms = torch.sum(torch.abs(complex_beamforming_vectors) ** 2, dim=-1)
    power_arrary_per_batch_bs = torch.sum(squared_norms, dim=1, keepdim=True)  # Summing over receivers
    safe_denominator = torch.where(power_arrary_per_batch_bs > 0, power_arrary_per_batch_bs,torch.ones_like(power_arrary_per_batch_bs)*0.001)
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




def uplink_pilot_signal(input_arrays, antenna_array,device,SNR,pilot_num,no_noise=0):
    """
        - input_arrays:  arrays, each of shape (batchsize,receiver_num, bs_num,number of multipath, 9), where each array corresponds to a specific scenario

    """

    lambda_=4
    if SNR==60:
        noise_power, snr_linear = 1, 10 ** (SNR / 10)
    else:
        snr_linear=10 ** (60 / 10)
        noise_power=1/10**((SNR-60)/10)
    max_antenna_x,max_antenna_z=np.max(antenna_array[:,0]),np.max(antenna_array[:,1])
    batchsize, receiver_num, bs_num, path_num,_=input_arrays.size()
    d = lambda_ / 2  # Half-wavelength spacing
    phi_angles, theta_angles = input_arrays[..., 5], input_arrays[..., 6]

    scalor_factor=math.sqrt(snr_linear)

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
    if no_noise:
        channel_noise=channel_multipath_multiple_plot
    else:
        channel_noise = channel_multipath_multiple_plot + noise.to(device)
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
