import sys
import os

# Get the absolute path of the higher directory
higher_directories = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))]

# Append the higher directory to sys.path
for directory in higher_directories:
    if directory not in sys.path: sys.path.append(directory)

from JetsFromFile import JetsFromFile
from JetImages import JetImage

def generate_images_from_seeds(kappa:float, energy_gev:int, seeds:list[int], data_dir:str):
    all_tuples = [generate_images_from_seed(kappa, energy_gev, seed, data_dir) for seed in seeds]

    up_images   = np.concatenate([t[0] for t in all_tuples])
    down_images = np.concatenate([t[1] for t in all_tuples])

    return up_images, down_images


def generate_images_from_seed(kappa:float, energy_gev:int, seed:int, data_dir:str):
    JET_EVENTS_PER_FILE = 10_000
 
    up_jet_datafile   = JetsFromFile(energy_gev,   "up", seed, data_dir)
    down_jet_datafile = JetsFromFile(energy_gev, "down", seed, data_dir)

    up_jets   = up_jet_datafile.from_txt()
    down_jets = down_jet_datafile.from_txt()

    up_images   = [JetImage.two_channel_image(jet, kappa) for jet in up_jets]
    down_images = [JetImage.two_channel_image(jet, kappa) for jet in down_jets]

    return up_images, down_images


import numpy as np
import torch
from torchvision import tv_tensors

def seed_group_file_name(image_dir:str, energy_gev:int, kappa:float, seed_group:int):
    return "{}/{}GEV-kappa{}-seedgroup{}.npy".format(image_dir, energy_gev, round(kappa, 1), seed_group)

def main():
    """
    Main function to generate and save images from a seed value.
    This function takes command line arguments to specify the data directory, image directory,
    seed value, energy in GeV, and kappa value. It generates images using these parameters,
    concatenates the images, and saves them to a file.
    Command Line Arguments:
    - data_dir (str): Path to the data directory.
    - image_dir (str): Path to the image directory.
    - seed (int): Seed value for random number generation.
    - energy_gev (int): Energy in GeV.
    - kappa (float): Kappa value.
    The generated images are saved in the specified image directory with a filename that includes
    the energy in GeV, kappa value (rounded to one decimal place), and seed value.
    """
    data_dir  = sys.argv[1]
    image_dir = sys.argv[2]
    energy_gev = int(sys.argv[3])
    kappa      = float(sys.argv[4])

    # Split the 100 seeds into 20 groups of 5 seeds each
    start_seeds = [i for i in range(1, 101, 5)]
    end_seeds   = [i + 4 for i in start_seeds]

    for i, (start_seed, end_seed) in enumerate(zip(start_seeds, end_seeds)):        
        print("Generating images for seed group {}. Seeds {} to {}...".format(i, start_seed, end_seed))
        up_images, down_images = generate_images_from_seeds(kappa, energy_gev, range(start_seed, end_seed+1), data_dir)

        all_images   = JetImage.preprocess_many_images(
                            np.concatenate([up_images, down_images])
                        )
        
        is_up_labels = np.concatenate([np.ones(len(up_images)), np.zeros(len(down_images))])

        all_images = tv_tensors.Image(all_images).float()
        is_up_labels = torch.tensor(is_up_labels).long()

        # Save all the images with their labels
        filename = seed_group_file_name(energy_gev, kappa, i)
        torch.save((all_images, is_up_labels), filename)


    # training_images = tv_tensors.Image(standardized[:last_training_idx]).float()
    # training_labels = torch.tensor(is_up[:last_training_idx]).long()

    # Concatenate and save the image arrays, stacking up above down images along the first dimension
    # images = np.concatenate([up_images, down_images])

    # Save the images to a file in the data directory with the seed, the energy in GeV, 
    # and the value for kappa to one decimal place in the filename
    # filename = "{}/{}GEV-kappa{}-seeds{}to{}.npy".format(image_dir, energy_gev, round(kappa, 1), start_seed, end_seed)
    # np.save(filename, images)
