import numpy as np
import os
import torch

from Model import CNN
from generate_images import generate_jet_image_memmaps
from training import CNNTrainer, train_model
from evaluate_model import down_quark_efficiency_roc

def run_all_kappa(raw_data_dir:str, num_seeds:int, 
                    energy_gev:int,  
                    output_data_root_dir:str):
    
    KAPPAS = np.arange(0,1)
    for k in KAPPAS:
        modified_out_dir = os.path.join(output_data_root_dir, "saved-data-for-{}GeV-{}-tenth-kappa".format(energy_gev, k))

        k /= 10

        generate_and_train(raw_data_dir, num_seeds, energy_gev, k, modified_out_dir)
        down_quark_efficiency_roc(modified_out_dir, "best_model.pth", modified_out_dir, energy_gev, k)

def generate_and_train(raw_data_dir:str, num_seeds:int, 
                        energy_gev:int, kappa:float, 
                        output_data_root_dir:str):

    print("Beginning a new image generation and training process for {} GeV and kappa = {}".format(energy_gev, kappa))
    generate_jet_image_memmaps(raw_data_dir, energy_gev, kappa, num_seeds, output_data_root_dir)

    NUM_CHANNELS = 2
    model    = CNN(NUM_CHANNELS)

    training_dir   = os.path.join(output_data_root_dir, "training")
    augmented_dir  = os.path.join(output_data_root_dir, "augmented")
    validation_dir = os.path.join(output_data_root_dir, "validation")

    best_model, training_loss_x, training_loss, val_loss_x, val_loss = \
        train_model(model, kappa, training_dir, augmented_dir, validation_dir, output_data_root_dir, num_epochs=1)



    

    