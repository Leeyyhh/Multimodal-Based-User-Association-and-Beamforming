# This is a sample Python script.
import time
# time.sleep(3600*4)
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from utils.beam_design_related import calculate_received_signal_strength_for_batches, beam_Logger, deduct_new_model_input2, construct_car_size,uplink_pilot_signal_new2
import numpy as np
import sys
from tqdm import tqdm
import os
import torch
from utils.loader import CustomDataset,InfiniteDataLoader
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
import random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# Set seed globally
set_seed(2323)
# Define the directory as a Path object
class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class BeamTraining:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.save_dir = opt.save_dir
    def obtain_Phi(self):
        pilot_length = self.pilot_num
        omega_tau_p = np.exp(-1j * 2 * np.pi / pilot_length)
        indices = np.arange(pilot_length)
        Phi = omega_tau_p ** np.outer(indices, indices)
        if self.car_num > self.pilot_num:
            np.random.seed(1)
            Phi_extended = np.exp(1j * 2 * np.pi * np.random.rand(self.car_num, pilot_length))
            Phi_extended[:pilot_length, :] = Phi
        else:
            Phi_extended = Phi[:self.car_num, :]
        pilot_tensor = torch.tensor(Phi_extended, dtype=torch.complex64).to(device).unsqueeze(-2).unsqueeze(-1).unsqueeze(0)  # (receiver_num,pilot_length)
        return pilot_tensor
    def save_setup_parameters(self):
        # Automatically capture and save all attributes of the instance
        with open(self.save_dir / 'setup_parameters.json', 'w') as f:
            json.dump(vars(self), f, indent=4, default=str)  # Using default=str to handle non-serializable objects like 'device'
        print("Setup parameters saved to:", self.save_dir / 'setup_parameters.json')
    def setup(self, training =True):
        self.training=training
        self.uplink_SNR=60
        self.pilot_num = 256
        self.no_noise=False
        self.car_num = 6
        self.dropout_prob = 0
        self.RF_scalor=4
        self.frozen_layers = 0
        self.RF_layer_num = 2
        self.pilot_tensor = self.obtain_Phi()

        self.transfer_learning = True
        self.restore_rf_paths= "./runs/Pilot/exp_SNR_60_pilot256_1/weights/best_beam.pt"
        self.restore_im_paths="./runs/Image/max_min_picture2/weights/best_beam.pt"

        # Load weights, setup directories, loggers, etc.
        self.test_restore_path='D:\LYH\GNN_0507\\runs\image_RF_new\max_min1_SNR_60_pilot256\weights\\best_beam.pt'

        self.save_dir = self.save_dir + f"_SNR_{self.uplink_SNR}_pilot{self.pilot_num}"
        if self.training:
            self.save_dir = Path(self.create_directory(self.save_dir))
        else:
            self.save_dir = Path((self.save_dir))

        self.log_file_path = self.save_dir / "train.log"
        self.weights_dir = self.save_dir / "weights"
        if self.training:
            sys.stdout = beam_Logger(self.log_file_path)
            self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.last_weight, self.best_weight = self.weights_dir / "last_beam.pt", self.weights_dir / "best_beam.pt"

        self.max_ant_x, self.max_ant_z = np.max(self.opt.antenna_array[:, 0]), np.max(self.opt.antenna_array[:, 1])
        self.max_ant = self.max_ant_x * self.max_ant_z
        self.model_inputs_size =  3 * self.opt.bs_num
        self.model_inputs_size_RF = self.opt.bs_num*self.max_ant*2

        self.train_dataset = CustomDataset(self.opt.train_file, seed=0, cache=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        if training:
            self.valid_dataset = CustomDataset(self.opt.valid_file, seed=0, cache=True)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        else:
            self.valid_dataset = CustomDataset(self.opt.test_file, seed=0, cache=True)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.opt.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        #Car configuration
        self.car_size_array = construct_car_size()
        self.height_car = self.car_size_array[:, 3] / 2
        self.w2c_list, self.K_list, self.location_list = obtain_K_w2c()
        self.calculate_function = calculate_received_signal_strength_for_batches
        self.chosen_calculation_function_index=2
        self.divide_num = torch.tensor([1280, 720, 1280, 720, 1, 1]).view(1, 1, 1, 6)
        self.nbs = 256 * 2*2  # nominal batch size
        self.accumulate = max(round(self.nbs / self.opt.batch_size), 1)
        self.nb = len(self.train_loader)
        self.last_opt_step = -1
        self.min_loss = float('inf')
        self.nn_outputsize=self.max_ant * 2*self.opt.bs_num+ self.opt.bs_num
        self.learning_rate = 0.0002


        self.new_feature_IM = 512
        self.f_input_size_IM = 512
        self.hidden_layers_IM =  [512,512]
        self.output_size_IM = 512
        #model setup
        self.new_feature_RF = 512*2
        self.hidden_layers_RF = [1024,1024]
        self.output_size_RF = 512*2


        self.new_feature_IMRF= self.output_size_RF + self.output_size_IM
        self.hidden_layers_IMRF = [1024*2]
        self.output_size_IMRF = 1024

        self.linear=torch.nn.Linear(self.output_size_IMRF, self.nn_outputsize).to(self.device).float()
        self.ones_d=torch.ones([1, self.car_num,self.opt.bs_num, self.max_ant_x,self.max_ant_z, 2]).to(self.device).float()

        self.RF_models=self.set_up_models()
        self.clip_gradient_label = True
        self.clip_gradient_value =2
        self.IM_models=self.set_up_image_models()
        self.RF_IM_models=self.set_up_imageRF_models()
        self.models=[self.IM_models,self.RF_models,self.RF_IM_models]
        if self.training:
            self.save_setup_parameters()




    def set_up_models(self):
        # Create instances using the helper function
        self.feature_models = Beampredict(self.model_inputs_size_RF, self.hidden_layers_RF, self.opt.Batch_normal, self.new_feature_RF).to(self.device).float()
        # self.feature_models_RF = Beampredict(self.model_inputs_size_RF, self.hidden_layers, self.opt.Batch_normal,  self.new_feature).to(self.device).float()
        self.f_a0 = Beampredict(self.new_feature_RF, self.hidden_layers_RF, self.opt.Batch_normal, self.output_size_RF).to(self.device).float()
        self.f_s0 = Beampredict(self.new_feature_RF, self.hidden_layers_RF, self.opt.Batch_normal, self.output_size_RF).to(self.device).float()
        self.f_c0 = Beampredict(self.output_size_RF * 2, self.hidden_layers_RF, self.opt.Batch_normal, self.output_size_RF, ).to(self.device).float()
        self.f_a1 = Beampredict(self.output_size_RF, self.hidden_layers_RF, self.opt.Batch_normal, self.output_size_RF).to(self.device).float()
        self.f_s1 = Beampredict(self.output_size_RF, self.hidden_layers_RF, self.opt.Batch_normal, self.output_size_RF).to(self.device).float()
        self.f_c1 = Beampredict(self.output_size_RF * 2, self.hidden_layers_RF, self.opt.Batch_normal, self.output_size_RF).to(self.device).float()
        models = [self.feature_models, self.f_a0, self.f_s0, self.f_c0, self.f_a1, self.f_s1, self.f_c1]

        return models

    def set_up_image_models(self):

        # Create instances using the helper function
        self.feature_models_img = Beampredict(self.model_inputs_size, self.hidden_layers_IM, self.opt.Batch_normal, self.new_feature_IM).to(self.device).float()
        # self.feature_models_RF_img = Beampredict(self.model_inputs_size_RF, self.hidden_layers, self.opt.Batch_normal,  self.new_feature).to(self.device).float()
        self.f_a0_img = Beampredict( self.new_feature_IM, self.hidden_layers_IM, self.opt.Batch_normal,  self.output_size_IM).to(self.device).float()
        self.f_s0_img = Beampredict( self.new_feature_IM, self.hidden_layers_IM, self.opt.Batch_normal,  self.output_size_IM).to(self.device).float()
        self.f_c0_img = Beampredict( self.f_input_size_IM* 2, self.hidden_layers_IM, self.opt.Batch_normal,  self.output_size_IM,).to(self.device).float()
        self.f_a1_img = Beampredict( self.f_input_size_IM, self.hidden_layers_IM, self.opt.Batch_normal,  self.output_size_IM).to(self.device).float()
        self.f_s1_img = Beampredict( self.f_input_size_IM, self.hidden_layers_IM, self.opt.Batch_normal,  self.output_size_IM).to(self.device).float()
        self.f_c1_img = Beampredict( self.f_input_size_IM * 2, self.hidden_layers_IM, self.opt.Batch_normal,  self.output_size_IM).to(self.device).float()
        models = [self.feature_models_img, self.f_a0_img, self.f_s0_img, self.f_c0_img, self.f_a1_img, self.f_s1_img, self.f_c1_img]
        return models


    def set_up_imageRF_models(self):

        # Create instances using the helper function
        input= self.output_size_RF + self.output_size_IM
        output= self.output_size_RF + self.output_size_IM
        self.f_a_imgRF = Beampredict( self.new_feature_IMRF, self.hidden_layers_IMRF, self.opt.Batch_normal, self.output_size_IMRF).to(self.device).float()
        self.f_s_imgRF = Beampredict( self.new_feature_IMRF, self.hidden_layers_IMRF, self.opt.Batch_normal,  self.output_size_IMRF).to(self.device).float()
        self.f_c_imgRF = Beampredict( self.output_size_IMRF* 2, self.hidden_layers_IMRF, self.opt.Batch_normal,  self.output_size_IMRF).to(self.device).float()
        self.f_a1_imgRF = Beampredict( self.output_size_IMRF, self.hidden_layers_IMRF, self.opt.Batch_normal,  self.output_size_IMRF).to(self.device).float()
        self.f_s1_imgRF = Beampredict( self.output_size_IMRF, self.hidden_layers_IMRF, self.opt.Batch_normal,  self.output_size_IMRF).to(self.device).float()
        self.f_c1_imgRF = Beampredict( self.output_size_IMRF * 2, self.hidden_layers_IMRF, self.opt.Batch_normal,  self.output_size_IMRF).to(self.device).float()

        models = [self.f_a_imgRF, self.f_s_imgRF, self.f_c_imgRF,self.f_a1_imgRF,self.f_s1_imgRF,self.f_c1_imgRF, self.linear]
        return models



    def aggregation_block(self, feature, f_s,f_a,f_c):
        outputs = []
        batchsize=feature.size(0)
        s_output = f_s(feature.view(batchsize * self.car_num, -1)).view(batchsize, self.car_num, -1)
        a_output = f_a(feature.view(batchsize * self.car_num, -1)).view(batchsize, self.car_num, -1)
        a_output = torch.mean(a_output, dim=1, keepdim=True)-1/self.car_num*a_output
        # Concatenate s_output and a_output, then process through 'c' layer
        c_output = f_c(
            torch.cat([s_output.view(batchsize * self.car_num, -1), a_output.view(batchsize * self.car_num, -1)],
                      dim=1)).view(
            batchsize, self.car_num, -1)
        return c_output
    def train_block(self, feature_RF,feature_img):
        feature_RF = feature_RF / self.RF_scalor
        fc0_output = self.aggregation_block(feature_RF, self.f_s0, self.f_a0, self.f_c0)
        fc1_output = self.aggregation_block(fc0_output, self.f_s1, self.f_a1, self.f_c1)
        fc2_output = fc1_output

        fc0_output_img = self.aggregation_block(feature_img, self.f_s0_img, self.f_a0_img, self.f_c0_img)
        fc1_output_img = self.aggregation_block(fc0_output_img, self.f_s1_img, self.f_a1_img, self.f_c1_img)
        fc2_output_img = fc1_output_img

        combine_feature = torch.cat([fc2_output, fc2_output_img], dim=-1)

        fc_output_img_RF = self.aggregation_block(combine_feature, self.f_s_imgRF, self.f_a_imgRF, self.f_c_imgRF)
        fc_output_img_RF = self.aggregation_block(fc_output_img_RF, self.f_s1_imgRF, self.f_a1_imgRF, self.f_c1_imgRF)
        return fc_output_img_RF
    def train_epoch(self, epoch):
        losses = []  # Initialize list to store losses
        RSS=[]
        for model in self.models:
            if isinstance(model, list):  # Check if the item is a list of models
                for sub_model in model:
                    sub_model.train()
            else:
                model.train()
        pbar = tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.opt.epochs}", leave=True)
        for pbar_index, (imgs, targets, paths) in enumerate(pbar):
            targets = targets[:, :self.car_num, :, :, :].to(self.device)
            commitment_loss_ratio =1-min(epoch,10)/10
            ni = pbar_index + self.nb * epoch  # number integrated batches (since train start)
            self.train_batch = len(targets)
            imgs_new, remove_loc_tensor, detect_label_tensor = self.preprocess_data(imgs, targets)
            remove_loc_tensor = remove_loc_tensor.to(self.device).permute(0, 2, 1, 3).reshape(self.train_batch, self.car_num, -1)
            #(input_arrays, antenna_array,device,SNR):
            RF_signal=uplink_pilot_signal_new2(input_arrays=targets,antenna_array=self.opt.antenna_array,device=self.device,SNR=self.uplink_SNR,pilot_num=self.pilot_num,pilot_tensor=self.pilot_tensor,no_noise=self.no_noise).float()#(batchsize,carnumber,bs*antenna_num*2)
            # combine_signal = torch.cat([RF_signal, remove_loc_tensor], dim=2)
            feature_img = self.feature_models_img(remove_loc_tensor.reshape(self.train_batch * self.car_num, -1)).view(self.train_batch, self.car_num, -1)
            feature_RF = self.feature_models(RF_signal.reshape(self.train_batch * self.car_num, -1)).view(self.train_batch, self.car_num, -1)
            fc_output_img_RF=self.train_block(feature_RF,feature_img)
            raw_output=self.linear(fc_output_img_RF.view(self.train_batch * self.car_num, -1)).view(self.train_batch, self.car_num, -1)
            user_A_matrix = raw_output[:,:, :self.opt.bs_num].view(self.train_batch, self.car_num, self.opt.bs_num)
            beam_output = raw_output[:, :,self.opt.bs_num:].view(self.train_batch, self.car_num,self.opt.bs_num, self.max_ant_x,self.max_ant_z, 2)*self.ones_d

            if torch.any(torch.isnan(beam_output)):
                print("Output is NaN")
            loss, received_strength, min_values2, skip_training, commitment_loss, zero_num, zero_num2 = self.calculate_function(
                targets, beam_output,
                self.opt.antenna_array, self.device,
                self.opt.total_power, user_A_matrix,epoch=epoch)

            train_loss = -torch.mean(received_strength) + commitment_loss * commitment_loss_ratio
            losses.append(train_loss.item())  # Store the current batch loss
            RSS.append(received_strength.mean().item())
            train_loss.backward()
            if ni - self.last_opt_step >= self.accumulate:

                if self.clip_gradient_label:
                    for model in self.models:
                        if isinstance(model, list):  # Check if the item is a list of models
                            for sub_model in model:
                                torch.nn.utils.clip_grad_norm_(sub_model.parameters(), self.clip_gradient_value)
                        else:
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
        # Flatten the list of models if there are nested lists
        if self.frozen_layers==0:
            all_models = [model for sublist in self.models for model in (sublist if isinstance(sublist, list) else [sublist])]
        else:
            all_models=self.RF_IM_models
        # Gather all parameters from all models
        all_parameters = [param for model in all_models for param in model.parameters()]

        # Initialize the optimizer with all parameters
        self.optimizer = torch.optim.Adam(all_parameters, lr=self.learning_rate)

        self.optimizer.zero_grad()
        if self.transfer_learning:
        # self.restore_model("E:\PHD\\assiociation_0507\\runs\\train_noise1_lr\max_min_gnn5\weights\\best_beam.pt")
            self.restore_IM_model(self.restore_im_paths)
            self.restore_RF_model(self.restore_rf_paths)
            # self.restore_model(self.test_restore_path)
        for epoch in range(self.opt.epochs):
            self.epoch = epoch
            self.train_epoch( epoch)
            self.validate()
    def test(self):
        # Set all models to training mode
        all_models = [model for sublist in self.models for model in
                      (sublist if isinstance(sublist, list) else [sublist])]

        all_parameters = [param for model in all_models for param in model.parameters()]
        self.optimizer = torch.optim.Adam(all_parameters,lr=self.learning_rate)
        self.optimizer.zero_grad()
        self.restore_model(self.test_restore_path)
        self.validate()
    def validate(self):
        for model in self.models:
            if isinstance(model, list):  # Check if the item is a list of models
                for sub_model in model:
                    sub_model.eval()
            else:
                model.eval()
        valid_loss = []
        valid_loss2 = []

        with torch.no_grad():  # Disable gradient calculation
            for imgs, targets, paths in self.valid_loader:
                targets = targets[:, :self.car_num, :, :, :].to(self.device)
                self.valid_batch = len(targets)
                imgs_new, remove_loc_tensor, detect_label_tensor = self.preprocess_data(imgs, targets)
                remove_loc_tensor = remove_loc_tensor.to(self.device).permute(0, 2, 1, 3).reshape(self.valid_batch, self.car_num, -1)
                # (input_arrays, antenna_array,device,SNR):
                RF_signal = uplink_pilot_signal_new2(input_arrays=targets, antenna_array=self.opt.antenna_array, device=self.device,
                                                     SNR=self.uplink_SNR, pilot_num=self.pilot_num, pilot_tensor=self.pilot_tensor,
                                                     no_noise=self.no_noise).float()  # (batchsize,carnumber,bs*antenna_num*2)

                feature_img = self.feature_models_img(remove_loc_tensor.reshape(self.valid_batch * self.car_num, -1)).view(self.valid_batch,
                                                                                                                           self.car_num, -1)
                feature_RF = self.feature_models(RF_signal.reshape(self.valid_batch * self.car_num, -1)).view(self.valid_batch, self.car_num, -1)
                # combine_feature = torch.cat([feature, feature_RF], dim=2)

                fc_output_img_RF = self.train_block(feature_RF, feature_img)

                raw_output = self.linear(fc_output_img_RF.view(self.valid_batch * self.car_num, -1)).view(self.valid_batch, self.car_num, -1)

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

        if self.training and new_loss2 < self.min_loss:
            self.min_loss = new_loss2
            # Flatten the list of models if there are nested lists
            all_models = [model for sublist in self.models for model in (sublist if isinstance(sublist, list) else [sublist])]
            # Create the checkpoint
            ckpt = {
                "RF_epoch": self.epoch,
                "RF_models": {f'model_{i}': model.state_dict() for i, model in enumerate(all_models)},
                "RF_optimizer": self.optimizer.state_dict()
            }

            # Save the checkpoint
            torch.save(ckpt, self.best_weight)
    def preprocess_data(self, imgs, targets):
        batchsize = np.size(imgs, 0)
        imgs_stack = imgs.view(batchsize, self.opt.bs_num, self.opt.car_max_num, -1)
        imgs2 = imgs_stack[:, :, :self.car_num, :]
        height2 = self.height_car[:self.car_num]
        center_matrix = get_bounding_box_centers_matrix(imgs2, self.opt.bs_num)
        estimated_loc = construct_projection_matrix(center_matrix, self.K_list, np.stack(self.w2c_list), height2, np.stack(self.location_list))
        undetect_label = np.sum(center_matrix, axis=-1) < 0
        detect_label = np.sum(center_matrix, axis=-1) > 0
        detect_label_tensor = torch.tensor(detect_label, device=self.device).float().permute(0, 2, 1)
        estimated_loc[undetect_label, :] = -56
        location = targets[:, :self.car_num, 0, 0, 7:10]
        remove_loc_tensor = torch.tensor(estimated_loc, device=self.device).float() / 56
        imgs_new = (imgs2 / self.divide_num).view(batchsize, -1).to(self.device).float()
        imgs_new[imgs_new == 0] = -1
        return imgs_new, remove_loc_tensor, detect_label_tensor
    def restore_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.epoch = checkpoint['RF_epoch']

        all_models = [model for sublist in self.models for model in (sublist if isinstance(sublist, list) else [sublist])]

        for i, model in enumerate(all_models):
            model_state_dict = checkpoint['RF_models'][f'model_{i}']
            model.load_state_dict(model_state_dict)
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model and optimizer states have been restored from epoch {self.epoch}")

    def restore_RF_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['RF_epoch']

        # Flatten the list of models if there are nested lists
        all_models = [model for sublist in self.RF_models for model in (sublist if isinstance(sublist, list) else [sublist])]

        # Restore state dictionaries for each model
        for i, model in enumerate(all_models):
            model_state_dict = checkpoint['RF_models'][f'model_{i}']
            model.load_state_dict(model_state_dict)

        # Restore the optimizer state
        # self.optimizer.load_state_dict(checkpoint['RF_optimizer'])

        print(f"rf Model and optimizer states have been restored from epoch {epoch}")
    def restore_IM_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        # Flatten the list of models if there are nested lists
        all_models = [model for sublist in self.IM_models for model in (sublist if isinstance(sublist, list) else [sublist])]
        # Restore state dictionaries for each model
        for i, model in enumerate(all_models):
            model_state_dict = checkpoint['models'][f'model_{i}']
            model.load_state_dict(model_state_dict)

        print(f"iM Model and optimizer states have been restored from epoch {epoch}")
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
    train_save_dir = "./runs/Multimodal/"
    train_save_dir= os.path.join(train_save_dir, loss_function_name)
    training_label=0

    opt = argparse.Namespace(
        bs_num=4,
        batch_size=256,
        total_power=4,
        car_max_num=12,
        car_num=6,
        antenna_array=np.reshape([[4, 4], [4, 4], [4, 4], [4, 4]], [4, -1]),
        save_dir=train_save_dir,
        epochs=500,
        Batch_normal='BN',  # 'GN', 'LN', 'BN', 'None'
        weights=None,  # Specify path to weights if needed
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
