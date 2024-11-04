import os
from Model import CNN
from generate_images import generate_jet_image_memmaps
import torch
from training import CNNTrainer, train_model


def generate_and_train(raw_data_dir:str, num_seeds:int, 
                        energy_gev:int, kappa:float, 
                        output_data_root_dir:str):

    print("Beginning a new image generation and training process for {} GeV and kappa = {}".format(energy_gev, kappa))
    modified_out_dir = os.path.join(output_data_root_dir, "{}GeV-{}kappa".format(energy_gev, kappa))
    generate_jet_image_memmaps(raw_data_dir, energy_gev, kappa, num_seeds, modified_out_dir)

    NUM_CHANNELS = 2
    model    = CNN(NUM_CHANNELS)

    training_dir = os.path.join(modified_out_dir, "training")
    augmented_dir = os.path.join(modified_out_dir, "augmented")
    validation_dir = os.path.join(modified_out_dir, "validation")

    best_model, training_loss_x, training_loss, val_loss_x, val_loss = train_model(model, training_dir, augmented_dir, validation_dir, modified_out_dir, num_epochs=1)



    

    