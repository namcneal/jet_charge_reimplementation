from Model import CNN
import matplotlib.pyplot as plt
from mmap_ninja import RaggedMmap
import numpy as np
import torch
from torchvision import tv_tensors 


PERCENT_TRAINING   = 80

class CNNTrainer(object):
    def __init__(self, model):
        # self.batch_size = batch_size
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.model = model

    def loss_from_batch(self, batch, use_gpu=True):
        inputs, labels = batch
        if use_gpu:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

        outputs = self.model(inputs)
        loss = self.loss_function(outputs, labels)
        return loss
    
import os

# from images_from_seed import output_group_file_name

def load_memmaps_for_seed(base_dir, seed_no):
    image_memmap_filename = "seed{}-images".format(seed_no)
    label_memmap_filename = "seed{}-labels".format(seed_no)

    images = RaggedMmap(os.path.join(base_dir, image_memmap_filename))
    labels = RaggedMmap(os.path.join(base_dir, label_memmap_filename))

    return images,labels

def verify_all_memmap_entries(memmap:RaggedMmap, expected_num_entries:int=-1):
    if expected_num_entries > 0:
        if not len(memmap) == expected_num_entries:
            raise ValueError("The number of entries in the memmap {} did not match the expected {}".format(len(memmap), expected_num_entries))
        
    for (idx, _) in enumerate(memmap):
        try:
            memmap[idx] 
        except ValueError:
            print("Error on index {}".format(idx))
            print(memmap.shapes[idx])
            print(memmap.sizes[idx])
            raise ValueError("Error loading entry {} from memmap.".format(idx))

import gc
from math import floor
import torch.nn as nn
import random

def main():
    num_channels = 2
    model = CNN(num_channels)

    trainer   = CNNTrainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    use_gpu = True
    if use_gpu:
        torch.cuda.set_device(0) # Set the GPU to use
        model = nn.DataParallel(model) # Wrap the model with DataParallel
        model = model.to('cuda') # Move the model to the GPU
    print("Model created: ", model)


    # energy_gev = 100
    # kappa      = 0.2
    # print("\n== Training model for energy {} GeV and kappa {} ==".format(energy_gev, kappa))

    base_dir = os.path.join("d:", "up_down_batch_data")
    num_seeds = 100
    memmap_image_label_pairs = [load_memmaps_for_seed(base_dir, seed_no) for seed_no in range(1,num_seeds+1)]

    ENTRIES_PER_SEED = 313
    for (images, labels) in memmap_image_label_pairs:
        verify_all_memmap_entries(images,ENTRIES_PER_SEED)
        verify_all_memmap_entries(labels,ENTRIES_PER_SEED)

    num_batches = ENTRIES_PER_SEED
    PERCENT_TRAINING = 80

    num_training_batches   = floor(num_batches * PERCENT_TRAINING / 100)
    num_validation_batches = floor((num_batches - num_training_batches) / 2)
    num_testing_batches    = num_batches - num_training_batches - num_validation_batches

    print(f"Training groups: {num_training_batches}, Validation groups: {num_validation_batches}, Testing groups: {num_testing_batches}")

    shuffled_batch_idxx = list(range(0, num_batches))
    random.Random(0).shuffle(shuffled_batch_idxx)
    training_idxx   = shuffled_batch_idxx[:num_training_batches]
    validation_idxx = shuffled_batch_idxx[num_training_batches:num_batches-num_testing_batches]

    logs_dir = "./training_logs/"
    each_batch_training_losses = []

    WINDOW = 20
    windowed_training_losses = []
    windowed_training_loss_recorded_at_epoch = []

    validation_loss_recorded_at_epoch = []
    validation_losses = []
    best_validation_loss = float('inf')
    best_model = None

    BATCHES_PER_EPOCH = num_seeds * num_training_batches
    num_epochs = 35
    for epoch in range(1,num_epochs+1):
        print("\nStarting epoch {}/{}:".format(epoch+1, num_epochs))

        num_batches_seen_this_epoch = 0
        for idx in range(0, num_seeds):
            print("Working on Epoch: {} | Seed No. {}:".format(epoch, idx+1))

            images_batches, labels_batches = memmap_image_label_pairs[idx]

            avg_percent_down_quarks = 0
            for (i, training_batch_idx) in enumerate(training_idxx):
                num_batches_seen_this_epoch += 1

                images = torch.tensor(images_batches[training_batch_idx]).float()
                labels = torch.nn.functional.one_hot(torch.tensor(labels_batches[training_batch_idx]).long(), 2).float()
                avg_percent_down_quarks += torch.mean(labels[:,1]).item() 
         
                if use_gpu:
                    images = images.to('cuda')
                    labels = labels.to('cuda')

                optimizer.zero_grad()
                output = model(images)
                loss   = trainer.loss_function(output, labels)

                each_batch_training_losses.append(loss.item()) 
                windowed_training_losses.append(np.mean(each_batch_training_losses[-WINDOW:]))
                percent_through_epoch = num_batches_seen_this_epoch / BATCHES_PER_EPOCH
                windowed_training_loss_recorded_at_epoch.append(epoch - 1 + percent_through_epoch) 

                if i % 50 == 0:
                    print(f"\t\tEpoch {epoch} | Seed {idx+1} | Batch {i}/{num_training_batches} | Loss (bits): {loss / np.log(2)} | Last {WINDOW} Avg Loss: {windowed_training_losses[-1] / np.log(2)}")

                loss.backward()
                optimizer.step()
            # End of training for this seed   

            avg_percent_down_quarks /= num_training_batches

            cumulative_validation_loss = 0
            print("\tEstimating validation loss...")
            for validation_batch_idx in validation_idxx:

                images = torch.tensor(images_batches[validation_batch_idx]).float()
                labels = torch.nn.functional.one_hot(torch.tensor(labels_batches[validation_batch_idx]).long(), 2).float()

                with torch.no_grad():
                    validation_batch_loss = trainer.loss_from_batch((images, labels))
                    cumulative_validation_loss += validation_batch_loss.item()

            average_validation_loss_per_batch = cumulative_validation_loss / len(validation_idxx)
            validation_losses.append(average_validation_loss_per_batch)
            validation_loss_recorded_at_epoch.append(epoch -1 + percent_through_epoch)

            is_model_better = average_validation_loss_per_batch < best_validation_loss
            if is_model_better:
                best_validation_loss = average_validation_loss_per_batch    
                best_model = model

                torch.save(best_model.state_dict(), logs_dir+"best_model.pth")

            print(f"\n\tValidation loss: {average_validation_loss_per_batch / np.log(2)}")
            print(f"\tBest validation loss: {best_validation_loss / np.log(2)}")
            print("\tAverage percent of down quarks in training samples: {}".format(round(avg_percent_down_quarks, 3)))

            # Save images of intermediate training and validation losses
            plt.grid()

            plt.plot(windowed_training_loss_recorded_at_epoch,  windowed_training_losses   / np.log(2), label="Training Loss")
            plt.plot(validation_loss_recorded_at_epoch, validation_losses / np.log(2), label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Cross-Entropy Loss (bits)")
            plt.legend()

            plt.savefig(logs_dir + "intermediate_loss.png".format(epoch+1))
            plt.clf()

        # Save images of intermediate training and validation losses
        plt.plot(windowed_training_loss_recorded_at_epoch,   windowed_training_losses   / np.log(2), label="Training Loss")
        plt.plot(validation_loss_recorded_at_epoch, validation_losses / np.log(2), label="Validation Loss")

        plt.xlabel("Epochs")
        plt.ylabel("Cross-Entropy Loss (bits)")
        plt.legend()

        plt.savefig(logs_dir + "losses_after_epoch_{}.png".format(epoch+1))
        plt.clf()


    # Save the training and validation losses to files as numpy arrays 
    np.save(logs_dir+"training_losses.npy",   each_batch_training_losses / np.log(2))
    np.save(logs_dir+"validation_losses.npy", validation_losses / np.log(2))
    
    # Save a plot of the training and validation losses over each epoch, 

    plt.plot(windowed_training_loss_recorded_at_epoch, windowed_training_losses, label="Training Loss")
    plt.plot(validation_loss_recorded_at_epoch, validation_losses, label="Validation Loss")

    # Subdivide the tick axis into twenty per seed group, and major ticks should be the epochs
    # plt.xticks(np.linspace(0, num_epochs*num_training_groups, num_seed_groups+1), np.arange(num_seed_groups+1))

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(logs_dir+"losses_5e-4.png")

    print("Training complete.")

        
        




