from Model import CNN
import numpy as np
import torch
from torchvision import tv_tensors 


PERCENT_TRAINING   = 80

class CNNTrainer(object):
    def __init__(self, model ,batch_size):
        self.batch_size = batch_size
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.model = model

    def loss_from_batch(self, batch):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, labels)
        return loss
    
    @classmethod
    def get_loss(cls, trainer, batch):
        return trainer.loss_from_batch(trainer.model, batch)
    

# from multiprocessing.dummy import Pool
def parallel_trainer_batch_losses(model, batches, trainers:list[CNNTrainer]):
    assert len(batches) == len(trainers), "The number of batches must match the number of trainers"

    # pool   = Pool(len(trainers))
    losses = [trainer.loss_from_batch(batch) for trainer, batch in zip(trainers, batches)]

    return losses

import sys
sys.path.append("image_data")
from images_from_seed import seed_group_file_name

def batch_loader_from_seed_group(energy_gev, kappa, seed_group, batch_size):
    # load the images from the seed group file 
    data_dir = 'image_data'
    images, is_up_labels = torch.load(seed_group_file_name(data_dir, energy_gev, kappa, seed_group))

    # Split the data we just loaded into as many batches as possible, plus one with the remainders 
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, is_up_labels),batch_size=batch_size, shuffle=True) 

    return loader

def main():
    num_channels = 2
    model = CNN(num_channels)

    num_seed_groups = 20
    num_training_groups   = 16

    num_trainers = 4
    batch_size   = 512
    trainers = [CNNTrainer(model, batch_size) for _ in range(num_trainers)]

    energy_gev = 100
    kapps      = 0.2
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_losses   = []
    validation_losses = []
    validation_loaders = [batch_loader_from_seed_group(energy_gev, kapps, seed_group, batch_size) for seed_group in range(num_training_groups, num_seed_groups)]
        
    num_epochs = 35

    best_validation_loss = float('inf')
    best_model = None
    for epoch in range(num_epochs):
        for seed_group in range(0, num_training_groups):
            print("\nEpoch {}, seed group {} of {}...".format(epoch, seed_group, num_training_groups))
            running_training_loss   = 0

            training_batches = iter(batch_loader_from_seed_group(energy_gev, kapps, seed_group, batch_size))

            num_training_rounds    = len(training_batches) // num_trainers
            for training_round in range(num_training_rounds):
                batches = [next(training_batches) for _ in range(num_trainers)]
                losses = parallel_trainer_batch_losses(model, batches, trainers)

                # Update the model using the losses
                optimizer.zero_grad()
                sum(losses).backward()
                optimizer.step()

                running_training_loss += sum(losses).item()

                if training_round % 10 == 9:
                    print(f"\tRound {training_round+1} of {num_training_rounds}, training loss: {running_training_loss / 10}")
                    running_training_loss = 0.

            average_training_loss_per_batch = running_training_loss / len(training_batches)
            training_losses.append(average_training_loss_per_batch)

            # Calculate the validation loss
            validation_losses_per_seed_group = []
            for i in range(num_trainers):
                trainer = trainers[i]
                validation_loader = validation_loaders[i]
                for batch in validation_loader:
                    validation_losses_per_seed_group.append(trainer.loss_from_batch(batch).item())

            average_validation_loss_per_batch = sum(validation_losses_per_seed_group) / len(validation_losses_per_seed_group)
            print(f"\tSeed group {seed_group}, validation loss: {average_validation_loss_per_batch}")
            validation_losses.append(average_validation_loss_per_batch)

            is_model_better = average_validation_loss_per_batch < best_validation_loss
            if is_model_better:
                best_validation_loss = average_validation_loss_per_batch
                best_model = model

                torch.save(best_model.state_dict(), "best_model.pth")

        
        # Save the training and validation losses to files as numpy arrays 
        np.save("training_losses.npy", training_losses)
        np.save("validation_losses.npy", validation_losses)
        
        # Save a plot of the training and validation losses over each epoch, 
        import matplotlib.pyplot as plt

        plt.plot(training_losses, label="Training Loss")
        plt.plot(validation_losses, label="Validation Loss")

        # Subdivide the tick axis into twenty per seed group, and major ticks should be the epochs
        plt.xticks(np.linspace(0, num_epochs*num_training_groups, num_seed_groups+1), np.arange(num_seed_groups+1))

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("losses.png")


    print("Training complete.")

        
        




