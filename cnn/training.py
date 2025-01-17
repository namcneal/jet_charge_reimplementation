import matplotlib.pyplot as plt
from mmap_ninja import RaggedMmap
import numpy as np
import sys 
import torch

sys.path.append("../")
from Model import CNN
from JetsFromFile import JetsFromFile
from JetImages import JetImage

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
from torch.utils.data import Dataset
class DatasetFromMemmap(Dataset):
    def __init__(self, image_memmap, label_memmap):
        self.images = image_memmap
        self.labels = label_memmap

        if len(self.images) != len(self.labels):
            raise ValueError("Number of images and labels must be the same.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx])

    
import gc
from math import floor
import random
import mmap_ninja
from torch.utils.data import DataLoader, TensorDataset

def train_model(model:CNN, kappa:float,
                base_training_dir:str, augmented_training_dir:str, validation_dir:str,
                results_dir:str,
                num_epochs:int=35, batch_size:int=512, use_gpu:bool=True):
    
    trainer   = CNNTrainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1/1000)

    use_gpu = True
    if use_gpu:
        torch.cuda.set_device(0) # Set the GPU to use
        model = torch.nn.DataParallel(model) # Wrap the model with DataParallel
        model = model.to('cuda') # Move the model to the GPU

    # Load the base training images and their augmented counterparts
    base_training_dataset = DatasetFromMemmap(
        mmap_ninja.np_open_existing(os.path.join(base_training_dir, "images")),
        mmap_ninja.np_open_existing(os.path.join(base_training_dir, "labels"))
    )

    augmented_dataset = DatasetFromMemmap(
        mmap_ninja.np_open_existing(os.path.join(augmented_training_dir, "images")),
        mmap_ninja.np_open_existing(os.path.join(augmented_training_dir, "labels"))
    )

    # Combine the two datasets to create the full dataset used for training
    full_training_dataset = torch.utils.data.ConcatDataset([base_training_dataset, augmented_dataset])

    validation_dataset = DatasetFromMemmap(
        mmap_ninja.np_open_existing(os.path.join(validation_dir, "images")),
        mmap_ninja.np_open_existing(os.path.join(validation_dir, "labels"))
    )

    print("Training with {} total images.".format(len(full_training_dataset)))
    print("Of these, {} are in the base training set and {} are in the augmented set.".format(len(base_training_dataset), len(augmented_dataset)))
    print("Validating with {} images.".format(len(validation_dataset)))

    BATCH_SIZE = 512 
    training_dataloader   = DataLoader(full_training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset,    batch_size=BATCH_SIZE, shuffle=True)

    each_batch_training_losses = []

    WINDOW = 20
    windowed_training_losses = []
    windowed_training_loss_recorded_at_epoch = []

    validation_loss_recorded_at_epoch = []
    validation_losses    = []
    best_validation_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        print("\nStarting epoch {}/{}:".format(epoch+1, num_epochs))

        for (training_batch_idx, (image_batch, label_batch)) in enumerate(training_dataloader):
            images = image_batch.float()
            labels = label_batch.float()

            if use_gpu:
                images = image_batch.to('cuda')
                labels = labels.to('cuda')

            optimizer.zero_grad()
            output = model(images)
            loss   = trainer.loss_function(output, labels)

            loss.backward()
            optimizer.step()

            each_batch_training_losses.append(loss.item()) 
            windowed_training_losses.append(np.mean(each_batch_training_losses[-WINDOW:]))

            percent_through_epoch = training_batch_idx / len(training_dataloader)
            windowed_training_loss_recorded_at_epoch.append(epoch + percent_through_epoch)

            if training_batch_idx % 200 == 0:
                print(f"\tEpoch {epoch} | Batch {training_batch_idx}/{len(training_dataloader)} | Loss (bits): {loss / np.log(2)} | Last {WINDOW} Avg Loss: {windowed_training_losses[-1] / np.log(2)}")

                total_validation_loss = 0
                for (validation_images, validation_labels) in validation_dataloader:
                    with torch.no_grad():
                        validation_images = validation_images.float()
                        validation_labels = validation_labels.float()

                        if use_gpu:
                            validation_images = validation_images.to('cuda')
                            validation_labels = validation_labels.to('cuda')

                        validation_output = model(validation_images)
                        validation_batch_loss = trainer.loss_function(validation_output, validation_labels)
                        total_validation_loss += validation_batch_loss.item()
                # End of validation loop

                avg_validation_loss = total_validation_loss / len(validation_dataloader)
                validation_losses.append(avg_validation_loss)
                validation_loss_recorded_at_epoch.append(epoch + percent_through_epoch)

                is_model_better = avg_validation_loss < best_validation_loss
                if is_model_better:
                    best_validation_loss = avg_validation_loss
                    best_model = model
                    torch.save(best_model.state_dict(), os.path.join(results_dir, "best_model.pth".format(kappa)))
                
                ### Save an intermediate plot each time a group is finished
                plt.plot(windowed_training_loss_recorded_at_epoch, windowed_training_losses / np.log(2), label="Training Loss", color='blue')
                plt.plot(validation_loss_recorded_at_epoch, validation_losses / np.log(2), label="Validation Loss", color='orange')
                plt.xlabel("Epochs")
                plt.ylabel("Cross-Entropy Loss (bits)")
                plt.grid()
                plt.legend()
                plt.savefig(os.path.join(results_dir, "intermediate_losses.png".format(training_batch_idx, epoch+1)))
                plt.clf()
            # End of training loop

        print("Finished Epoch {} of {}.".format(epoch+1, num_epochs))
        print("\tValidation loss (bits): ", avg_validation_loss / np.log(2))
        print("\tBest validation loss so far: ", best_validation_loss/np.log(2)) 

        ### Save a plot of the training and validation losses after each epoch
        plt.plot(windowed_training_loss_recorded_at_epoch, windowed_training_losses / np.log(2), label="Training Loss", color='blue')
        plt.plot(validation_loss_recorded_at_epoch, validation_losses / np.log(2), label="Validation Loss", color='orange')
        plt.xlabel("Epochs")
        plt.ylabel("Cross-Entropy Loss (bits)")
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(results_dir, "losses_after_epoch_{}_kappa_{}.png".format(epoch+1, kappa)))
        plt.clf()

    # Save the training and validation losses to files as numpy arrays 
    np.save(os.path.join(results_dir, "windowed_training_loss_recorded_at_these_epochs.npy"), windowed_training_loss_recorded_at_epoch)
    np.save(os.path.join(results_dir, "windowed_training_losses.npy"), windowed_training_losses / np.log(2))
    np.save(os.path.join(results_dir, "validation_loss_recorded_at_these_epochs.npy"), validation_loss_recorded_at_epoch)
    np.save(os.path.join(results_dir, "validation_losses.npy"), validation_losses / np.log(2))

    print("Training complete.")
    return best_model, windowed_training_loss_recorded_at_epoch, windowed_training_losses, validation_loss_recorded_at_epoch, validation_losses

        
        




