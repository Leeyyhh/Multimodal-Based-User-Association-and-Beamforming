# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from utils.beam_design_related import calculate_received_signal_strength_for_batches, beam_Logger, deduct_new_model_input2, construct_car_size,uplink_pilot_signal_new2
import numpy as np
import sys
from tqdm import tqdm
import os
import torch
from utils.loader import CustomDataset,InfiniteDataLoader,CustomDataset_RF
from torch.utils.data import Dataset, DataLoader
from utils.netork_model import Beampredict_GNN as Beampredict
import torch.optim as optim
from utils.some_functions import create_directory, preprocess_data
from utils.camera_for_location import obtain_K_w2c
import argparse
from pathlib import Path
from utils.camera_for_location import get_bounding_box_centers_matrix, construct_projection_matrix, obtain_K_w2c
from prefetch_generator import BackgroundGenerator
import json

class BeamTraining:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.save_dir = opt.save_dir
    def save_setup_parameters(self):
        # Automatically capture and save all attributes of the instance
        with open(self.save_dir / 'setup_parameters.json', 'w') as f:
            json.dump(vars(self), f, indent=4, default=str)  # Using default=str to handle non-serializable objects like 'device'
        print("Setup parameters saved to:", self.save_dir / 'setup_parameters.json')
    def obtain_Phi(self):
        omega_tau_p = np.exp(-1j * 2 * np.pi / self.pilot_num)
        indices = np.arange(self.pilot_num)
        Phi = omega_tau_p ** np.outer(indices, indices)
        if self.car_num > self.pilot_num:
            np.random.seed(1)
            Phi_extended = np.exp(1j * 2 * np.pi * np.random.rand(self.car_num, self.pilot_num))
            Phi_extended[:self.pilot_num, :] = Phi
        else:
            Phi_extended = Phi[:self.car_num, :]
        pilot_tensor = torch.tensor(Phi_extended, dtype=torch.complex64).to(device).unsqueeze(-2).unsqueeze(-1).unsqueeze(0)  # (receiver_num,pilot_length)
        return pilot_tensor
    def setup(self,training=True):
        self.training=training
        self.uplink_SNR=60
        self.pilot_num = 256
        self.car_num = 6
        self.no_noise = False
        self.pilot_tensor = self.obtain_Phi()
        self.dropout_prob = 0
        # Load weights, setup directories, loggers, etc.
        self.layer_num=2
        self.save_dir =Path( self.save_dir + f"_SNR_{self.uplink_SNR}_pilot{self.pilot_num}")
        if self.training:
            self.save_dir=Path(self.create_directory(self.save_dir))
        self.log_file_path = self.save_dir / "train.log"
        self.weights_dir = self.save_dir / "weights"
        if self.training:
            sys.stdout = beam_Logger(self.log_file_path)
            self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.last_weight, self.best_weight = self.weights_dir / "last_beam.pt", self.weights_dir / "best_beam.pt"
        self.max_ant_x, self.max_ant_z = np.max(self.opt.antenna_array[:, 0]), np.max(self.opt.antenna_array[:, 1])
        self.max_ant = self.max_ant_x * self.max_ant_z
        self.model_inputs_size = self.opt.bs_num*self.max_ant*2
        self.clip_gradient_label = True
        self.clip_gradient_value = 2
        #data_set and data_loader
        self.train_dataset = CustomDataset_RF(self.opt.train_file, seed=2, cache=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        #Car configuration
        if training:
            self.valid_dataset = CustomDataset_RF(self.opt.valid_file, seed=1, cache=True)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        else:
            self.valid_dataset = CustomDataset_RF(self.opt.test_file, seed=0, cache=True)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        self.car_size_array = construct_car_size()
        self.height_car = self.car_size_array[:, 3] / 2
        self.w2c_list, self.K_list, self.location_list = obtain_K_w2c()

        self.calculate_function = calculate_received_signal_strength_for_batches
        self.chosen_calculation_function_index=2
        self.divide_num = torch.tensor([1280, 720, 1280, 720, 1, 1]).view(1, 1, 1, 6)
        self.nbs = 256*2*2  # nominal batch size
        self.accumulate = max(round(self.nbs / self.opt.batch_size), 1)
        self.nb = len(self.train_loader)
        self.last_opt_step = -1
        self.min_loss = float('inf')
        self.nn_outputsize=self.max_ant * 2* self.opt.bs_num+ self.opt.bs_num
        self.learning_rate = 0.0002
        #model setup
        self.feature_size = 1024
        self.hidden_layers = [1024,1024]
        self.output_size = 1024
        self.linear=torch.nn.Linear(self.feature_size, self.nn_outputsize).to(self.device).float()
        self.ones_d=torch.ones([1, self.car_num,self.opt.bs_num, self.max_ant_x,self.max_ant_z, 2]).to(self.device).float()
        self.set_up_models()
        if self.training:
            self.save_setup_parameters()
            # self.create_directory(self.save_dir)
    def set_up_models(self):

        # Create instances using the helper function
        self.feature_models = Beampredict(self.model_inputs_size, self.hidden_layers, self.opt.Batch_normal,  self.feature_size,self.dropout_prob).to(self.device).float()
        self.f_a0 = Beampredict( self.feature_size, self.hidden_layers, self.opt.Batch_normal,  self.output_size,self.dropout_prob).to(self.device).float()
        self.f_s0 = Beampredict( self.feature_size, self.hidden_layers, self.opt.Batch_normal,  self.output_size,self.dropout_prob).to(self.device).float()
        self.f_c0 = Beampredict( self.output_size * 2, self.hidden_layers, self.opt.Batch_normal,  self.output_size,self.dropout_prob).to(self.device).float()
        self.f_a1 = Beampredict( self.output_size, self.hidden_layers, self.opt.Batch_normal,  self.output_size,self.dropout_prob).to(self.device).float()
        self.f_s1 = Beampredict( self.output_size, self.hidden_layers, self.opt.Batch_normal,  self.output_size,self.dropout_prob).to(self.device).float()
        self.f_c1 = Beampredict( self.output_size * 2, self.hidden_layers, self.opt.Batch_normal,  self.output_size,self.dropout_prob).to(self.device).float()
        self.models=[self.feature_models, self.f_a0, self.f_s0, self.f_c0, self.f_a1, self.f_s1, self.f_c1,self.linear]

    def aggregation_block(self, feature, f_s,f_a,f_c):
        batchsize=feature.size(0)
        s_output = f_s(feature.view(batchsize * self.car_num, -1)).view(batchsize, self.car_num, -1)
        a_output = f_a(feature.view(batchsize * self.car_num, -1)).view(batchsize, self.car_num, -1)

        a_output = torch.mean(a_output, dim=1, keepdim=True)-1/self.car_num*a_output
        c_output = f_c(
            torch.cat([s_output.view(batchsize * self.car_num, -1), a_output.view(batchsize * self.car_num, -1)],
                      dim=1)).view(
            batchsize, self.car_num, -1)
        return c_output
    def train_epoch(self, epoch):
        losses = []  # Initialize list to store losses
        RSS=[]
        for model in self.models:
            model.train()
        pbar = tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.opt.epochs}", leave=True)
        for pbar_index, ( targets, paths) in enumerate(pbar):
            targets = targets[:, :self.car_num, :, :, :].to(self.device)
            commitment_loss_ratio =1-min(epoch,10)/10
            ni = pbar_index + self.nb * epoch  # number integrated batches (since train start)
            self.train_batch = len(targets)

            RF_signal=uplink_pilot_signal_new2(input_arrays=targets,antenna_array=self.opt.antenna_array,device=self.device,SNR=self.uplink_SNR,pilot_num=self.pilot_num,pilot_tensor=self.pilot_tensor,no_noise=self.no_noise).float()#(batchsize,carnumber,bs*antenna_num*2)
            feature = self.feature_models(RF_signal.reshape(self.train_batch * self.car_num, -1)).view(self.train_batch, self.car_num, -1)
            fc0_output = self.aggregation_block(feature, self.f_s0, self.f_a0, self.f_c0)
            fc1_output = self.aggregation_block(fc0_output, self.f_s1, self.f_a1, self.f_c1)
            raw_output=self.linear(fc1_output.view(self.train_batch * self.car_num, -1)).view(self.train_batch, self.car_num, -1)
            user_A_matrix = raw_output[:,:, :self.opt.bs_num].view(self.train_batch, self.car_num, self.opt.bs_num)
            beam_output = raw_output[:, :,self.opt.bs_num:].view(self.train_batch, self.car_num,self.opt.bs_num, self.max_ant_x,self.max_ant_z, 2)*self.ones_d

            if torch.any(torch.isnan(beam_output)):
                print("Output is NaN")
            loss, received_strength, min_values2, skip_training, commitment_loss, zero_num, zero_num2 = self.calculate_function(
                targets, beam_output,
                self.opt.antenna_array, self.device,
                self.opt.total_power, user_A_matrix,epoch=epoch)
            if skip_training:
                continue
            train_loss = -torch.mean(received_strength) + commitment_loss * commitment_loss_ratio
            losses.append(train_loss.item())  # Store the current batch loss
            RSS.append(received_strength.mean().item())
            train_loss.backward()
            if ni - self.last_opt_step >= self.accumulate:
                if self.clip_gradient_label:
                    for model in self.models:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_gradient_value)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.last_opt_step = ni
        mean_loss = sum(losses) / len(losses)
        mean_RSS = sum(RSS) / len(RSS)
        print(f"Mean Loss for Epoch {epoch+1}: {mean_loss:.4f}")
        print(f"Mean Received Strength for Epoch {epoch+1}: {mean_RSS:.4f}")
    def train(self):
        # Set all models to training mode
        all_parameters = [param for model in self.models for param in model.parameters()]
        self.optimizer = torch.optim.Adam(all_parameters,lr=self.learning_rate)
        self.optimizer.zero_grad()
        for epoch in range(self.opt.epochs):
            self.epoch = epoch
            self.train_epoch( epoch)
            self.validate()
    def validate(self):
        for model in self.models:
            model.eval()
        valid_loss = []
        valid_loss2 = []
        with torch.no_grad():  # Disable gradient calculation
            # for imgs, targets, paths in self.valid_loader:
            for targets, paths in self.valid_loader:
                targets = targets[:, :self.car_num, :, :, :].to(self.device)
                self.valid_batch = len(targets)
                RF_signal = uplink_pilot_signal_new2(input_arrays=targets, antenna_array=self.opt.antenna_array, device=self.device,
                                                SNR=self.uplink_SNR,pilot_num=self.pilot_num,pilot_tensor=self.pilot_tensor,no_noise=self.no_noise).float()  # (batchsize,carnumber,bs*antenna_num*2)
                feature = self.feature_models(RF_signal.reshape(self.valid_batch * self.car_num, -1)).view(self.valid_batch, self.car_num, -1)
                fc0_output = self.aggregation_block(feature, self.f_s0, self.f_a0, self.f_c0)
                fc1_output = self.aggregation_block(fc0_output, self.f_s1, self.f_a1, self.f_c1)
                raw_output = self.linear(fc1_output.view(self.valid_batch * self.car_num, -1)).view(self.valid_batch, self.car_num, -1)
                user_A_matrix = raw_output[:, :, :self.opt.bs_num].view(self.valid_batch, self.car_num, self.opt.bs_num)
                beam_output = raw_output[:, :, self.opt.bs_num:].view(self.valid_batch, self.car_num, self.opt.bs_num, self.max_ant_x, self.max_ant_z,
                                                                      2) * self.ones_d
                loss, received_strength, min_values2, skip_training, commitment_loss, zero_num, zero_num2 = self.calculate_function(
                    targets, beam_output, self.opt.antenna_array, self.device, self.opt.total_power, user_A_matrix,epoch=0)

                valid_loss.append(received_strength)
                valid_loss2.append(min_values2)

        flat_tensor = torch.cat([t.flatten() for t in valid_loss])
        flat_tensor2 = torch.cat([t.flatten() for t in valid_loss2])
        new_loss = -torch.mean(flat_tensor.float())
        new_loss2 = -torch.mean(flat_tensor2.float())
        print('epoch:',self.epoch,'loss:',new_loss,"normalization_loss", new_loss2)
        if new_loss2 < self.min_loss and self.training:
            self.min_loss = new_loss2
            ckpt = {
                "RF_epoch": self.epoch,
                "RF_models": {f'model_{i}': model.state_dict() for i, model in enumerate(self.models)},
                "RF_optimizer": self.optimizer.state_dict()}
            torch.save(ckpt, self.best_weight)

    def restore_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.epoch = checkpoint['RF_epoch']
        for i, model in enumerate(self.models):
            model_state_dict = checkpoint['RF_models'][f'model_{i}']
            model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint['RF_optimizer'])
        print(f"Model and optimizer states have been restored from epoch {self.epoch}")
    def test(self):
        # Set all models to training mode
        all_parameters = [param for model in self.models for param in model.parameters()]
        self.optimizer = torch.optim.Adam(all_parameters,lr=self.learning_rate)
        self.optimizer.zero_grad()

        self.restore_model("runs/Pilot/exp_SNR_60_pilot256_1/weights/best_beam.pt")
        self.validate()
    def create_directory(self, directory):
        base_directory = directory
        i = 1
        while os.path.exists(directory):
            directory = f"{base_directory}_{i}"
            i += 1
        if self.training:
            os.makedirs(directory)
            print(f"Directory created: {directory}")
        return directory

    def create_layer(self, input_size, output_sizes, batch_norm):
        return Beampredict(input_size, [1024], batch_norm, output_sizes[-1], dropout_prob=self.dropout_prob).to(self.device).float()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import time
    # time.sleep(5*3600+1800)
    loss_function_name = "exp"
    training_label=0
    train_save_dir = "runs/Pilot/"
    train_save_dir= os.path.join(train_save_dir, loss_function_name)
    opt = argparse.Namespace(
        bs_num=4,
        batch_size=256,
        total_power=4,
        car_max_num=12,
        car_num=6,
        antenna_array=np.reshape([[4, 4], [4, 4], [4, 4], [4, 4]], [4, -1]),
        save_dir=train_save_dir,
        epochs=800,
        Batch_normal='BN',  # 'GN', 'LN', 'BN', 'None'
        weights=None,  # Specify path to weights if neededc
        data='path_to_data',
        train_file='E:/PHD/train_data_0423/train_tensor.txt',
        valid_file='E:/PHD/train_data_0423/valid_tensor.txt',
        test_file='E:/PHD/train_data_0423/test_tensor.txt'
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = BeamTraining(opt, device)
    training.setup(training=training_label)
    if training_label==1:
        training.train()
    else:
        training.test()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
