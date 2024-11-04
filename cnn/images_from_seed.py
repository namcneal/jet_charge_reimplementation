import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import random
from random import shuffle

import matplotlib.image as mpimg
from pathlib import Path
import mmap_ninja
from mmap_ninja import RaggedMmap


# Get the absolute path of the higher directory
higher_directories = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))]

# Append the higher directory to sys.path
for directory in higher_directories:
    if directory not in sys.path: sys.path.append(directory)

from JetsFromFile import JetsFromFile
from JetImages import JetImage

from torch.utils.data import Dataset
class DatasetFromGroups(Dataset):
    
    def __init__(self, base_dir:str, group_folder_names:str, augment:bool=False):
        super(DatasetFromGroups, self).__init__()
        self.num_groups = 0
        self.image_memmaps = [RaggedMmap(os.path.join(base_dir, folder_name)) for folder_name in group_folder_names]
        self.label_memmaps = [RaggedMmap(os.path.join(base_dir, folder_name)) for folder_name in group_folder_names]

        if augment:
            NUM_AUGMENTATIONS = 7
            self.
        self.num_groups = len(group_folder_names)
    

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
    defaults = ["../data/up_down_2017", os.path.join('d:/', "up_down_2017_data"), 1000, 0.2]
    if len(sys.argv) < len(defaults) + 1:
        print("The number of arguments passed to the script was less than the required number of arguments.")
        print("Using default values for the arguments: ", defaults)

        in_dir, out_dir, energy_gev, kappa = defaults
    else:
        print("Using provided arguments: ", sys.argv[1:])
        in_dir   = sys.argv[1]
        out_dir  = sys.argv[2]
        energy_gev = int(sys.argv[3])
        kappa      = float(sys.argv[4])

    # Multiplied by two for up and down
    num_seeds = 10
    total_num_images_per_seed = 2 * JetsFromFile.JET_EVENTS_PER_FILE

    num_data_groups = 10    
    is_group_image_memmap_created = [False for _ in range(num_data_groups)]
    is_group_label_memmap_created = [False for _ in range(num_data_groups)]
    group_image_memmap = [None for _ in range(num_data_groups)]
    group_label_memmap = [None for _ in range(num_data_groups)]
    group_image_out_dirs = [os.path.join(out_dir, "group{}-images".format(i)) for i in range(num_data_groups)]
    group_label_out_dirs = [os.path.join(out_dir, "group{}-labels".format(i)) for i in range(num_data_groups)]
    for dir in group_image_out_dirs + group_label_out_dirs:
        if not os.path.exists(dir):
            print("Directory {} does not exist. Creating it.".format(dir))
            os.makedirs(dir)

    for seed_no in range(1, num_seeds+1):
        print("Generating images for seed {} of {}".format(seed_no, num_seeds))
        up_images, down_images = generate_images_from_seed(kappa, energy_gev, seed_no, in_dir)
        all_images, is_down    = combine_up_down_images_create_labels(up_images, down_images)

        print("\tAll jet images have been generated and labelled. Up and down images interlaced.".format(seed_no))
        print("\tPreprocessing the images' first (i.e. momentum) channel ".format(seed_no))
        all_images = JetImage.preprocess_a_channel(all_images, channel_idx=0)
        
        print("\tAll jet images for this seed are preprocessed. Now shuffling, partitioning into groups, and saving images.")
        
        permutation = np.arange(total_num_images_per_seed)
        random.Random(0).shuffle(permutation)
        
        for group_no in range(num_data_groups):
            print("\t\tWorking on group {} of 10".format(group_no))
            indices_for_group = permutation % num_data_groups == group_no
            group_images = all_images[indices_for_group]
            group_labels = is_down[indices_for_group]

            """ Save the images and labels to memmap """
            # If the memmap has not been created, create it and ensure we don't create it again
            if not is_group_image_memmap_created[group_no]:
                group_image_memmap[group_no] = RaggedMmap.from_lists(group_image_out_dirs[group_no], [group_images])
                is_group_image_memmap_created[group_no] = True
            
            # If the memmap has been created, append the images to it
            else:
                group_image_memmap[group_no].extend(group_images)

            # Do the same for the labels
            if not is_group_label_memmap_created[group_no]:
                group_label_memmap[group_no] = RaggedMmap.from_lists(group_label_out_dirs[group_no], [group_labels])
                is_group_label_memmap_created[group_no] = True
            else:
                group_label_memmap[group_no].extend(group_labels)
        
        

def combine_up_down_images_create_labels(up_images:np.array, down_images:np.array):
    if len(up_images.shape) != 4:
        raise ValueError("The up images must have shape (num_images, num_channels, num_pixels, num_pixels)")
    if len(down_images.shape) != 4:
        raise ValueError("The down images must have shape (num_images, num_channels, num_pixels, num_pixels)")
    if up_images.shape[1] != down_images.shape[1]:
        raise ValueError("The number of channels in the up and down images must be the same")
    
    num_up   = up_images.shape[0]
    num_down = down_images.shape[0]

    all_images = np.concatenate([up_images, down_images], axis=0)
    if all_images.dtype != np.float32:
        all_images = all_images.astype(np.float32)

    is_down = np.concatenate([np.zeros(num_up), np.ones(num_down)])

    # Interleaf the up and down images
    interleafed_idx = list(range(num_up + num_down))
    interleafed_idx[::2]  = range(num_up)
    interleafed_idx[1::2] = range(num_up, num_up + num_down)
    
    all_images = all_images[interleafed_idx,:,:,:]
    is_down    = is_down[interleafed_idx]

    return all_images, is_down

def generate_images_from_seed(kappa:float, energy_gev:int, seed:int, data_dir:str, num_channels:int=2):
    
    year = 2017
    up_jet_datafile   = JetsFromFile(energy_gev,   "up", seed, data_dir, year)
    down_jet_datafile = JetsFromFile(energy_gev, "down", seed, data_dir, year)

    up_jets   = up_jet_datafile.from_txt()
    down_jets = down_jet_datafile.from_txt()

    ## Find any jets with no particles
    no_particles = [jet for jet in up_jets + down_jets if jet.get_num_particles() == 0]
    if len(no_particles) > 0:
        raise ValueError("Some jets have no particles")
    
    if num_channels == 2:
        up_images   = np.stack([JetImage.two_channel_image_from_jet(jet, kappa) for jet in up_jets],   axis=0)
        down_images = np.stack([JetImage.two_channel_image_from_jet(jet, kappa) for jet in down_jets], axis=0)
    else:   
        raise ValueError("Only two channels are supported")
    
    return up_images, down_images


