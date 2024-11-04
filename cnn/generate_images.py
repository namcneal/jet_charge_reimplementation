import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor

import numpy as np
import random
from random import shuffle

import matplotlib.image as mpimg
from pathlib import Path
import mmap_ninja
import torch

# Get the absolute path of the higher directory
higher_directories = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))]

# Append the higher directory to sys.path
for directory in higher_directories:
    if directory not in sys.path: sys.path.append(directory)

from JetsFromFile import JetsFromFile
from JetImages import JetImage

def generate_jet_image_memmaps(raw_data_dir:str, energy_gev:str, kappa:float, num_seeds:int, output_data_root_dir:str):
    if not os.path.exists(output_data_root_dir):
        os.makedirs(output_data_root_dir)

    generate_images_from_all_seeds(raw_data_dir, energy_gev, kappa, num_seeds, output_data_root_dir)
    augment_training_data(output_data_root_dir)
    verify_all_memmaps(output_data_root_dir)

def verify_all_memmaps(base_dir:str):
    training_image_dir = os.path.join(base_dir, "training/images")
    training_label_dir = os.path.join(base_dir, "training/labels")
    validation_image_dir = os.path.join(base_dir, "validation/images")
    validation_label_dir = os.path.join(base_dir, "validation/labels")
    testing_image_dir = os.path.join(base_dir, "testing/images")
    testing_label_dir = os.path.join(base_dir, "testing/labels")

    for dir in [training_image_dir, training_label_dir, validation_image_dir, validation_label_dir, testing_image_dir, testing_label_dir]:
        verify_all_memmap_elements( dir )

def generate_images_from_all_seeds(in_data_dir:str, energy_gev:int, kappa:float, num_seeds:int, 
                                    out_data_dir:str,
                                    num_channels:int=2):
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

    # Multiplied by two for up and down
    total_num_images_per_seed = 2 * JetsFromFile.JET_EVENTS_PER_FILE

    training_data_dir   = os.path.join(out_data_dir, "training")
    validation_data_dir = os.path.join(out_data_dir, "validation")
    testing_data_dir    = os.path.join(out_data_dir, "testing")
    
    train_val_test_image_dirs = [os.path.join(dir, "images") for dir in [training_data_dir, validation_data_dir, testing_data_dir]]
    train_val_test_label_dirs = [os.path.join(dir, "labels") for dir in [training_data_dir, validation_data_dir, testing_data_dir]]
    for dir in train_val_test_image_dirs + train_val_test_label_dirs:
        if not os.path.exists(dir):
            print("Creating directory: ", dir)
            os.makedirs(dir)

    train_val_test_image_memmaps = [None, None, None]
    train_val_test_label_memmaps = [None, None, None]

    first_seed = 1        
    for seed_no in range(first_seed, num_seeds + 1):
        print("Generating images for seed {} of {}".format(seed_no, num_seeds))
        up_images, down_images = generate_images_from_seed(kappa, energy_gev, seed_no, in_data_dir, num_channels=num_channels)
        all_images, is_down    = combine_up_down_images_create_labels(up_images, down_images)

        print("\tAll jet images have been generated and labelled. Up and down images interlaced.".format(seed_no))
        print("\tPreprocessing the images' first (i.e. momentum) channel ".format(seed_no))
        all_images = JetImage.preprocess_a_channel(all_images, channel_idx=0)
        all_labels = torch.nn.functional.one_hot(torch.from_numpy(is_down).long(), 2).numpy()

        print("\tAll jets for this seed are processed. Saving 80 pct to training, 10 pct to validation, and 10 pct to testing.")
        TRAIN_PCT = 80
        VALIDATION_PCT = 10
        num_training   = int(TRAIN_PCT/100 * total_num_images_per_seed)
        num_validation = int(VALIDATION_PCT/100 * total_num_images_per_seed)
        num_testing    = total_num_images_per_seed - num_training - num_validation

        all_indices = np.arange(total_num_images_per_seed, dtype=np.int32)
        training_indices   = all_indices[:num_training]
        validation_indices = all_indices[num_training:num_training+num_validation]
        testing_indices    = all_indices[-num_testing:] 
        
        train_val_test_indices = [training_indices, validation_indices, testing_indices]

        if seed_no == first_seed:
            for i, (image_dir, label_dir) in enumerate(zip(train_val_test_image_dirs, train_val_test_label_dirs)):
                train_val_test_image_memmaps[i] = mmap_ninja.np_from_ndarray(image_dir, all_images[train_val_test_indices[i]])
                train_val_test_label_memmaps[i] = mmap_ninja.np_from_ndarray(label_dir, all_labels[train_val_test_indices[i]]) 
        else:
            for i, (image_dir, label_dir) in enumerate(zip(train_val_test_image_dirs, train_val_test_label_dirs)):
                mmap_ninja.np_extend(train_val_test_image_memmaps[i], all_images[train_val_test_indices[i]])
                mmap_ninja.np_extend(train_val_test_label_memmaps[i], all_labels[train_val_test_indices[i]])

        print("\tImages and labels saved to memmap files for seed {}.".format(seed_no))
        print("\tSaved in total: {} training, {} validation, {} testing images.".format(num_training, num_validation, num_testing))

def augment_training_data(output_data_root_dir:str):
    image_memmap_dir = os.path.join(output_data_root_dir, "training/images")
    label_memmap_dir = os.path.join(output_data_root_dir, "training/labels")

    print("Augmenting the training data.")
    loaded_images = mmap_ninja.np_open_existing(image_memmap_dir)
    loaded_labels = mmap_ninja.np_open_existing(label_memmap_dir)

    if len(loaded_images) != len(loaded_labels):
        raise ValueError("The number of images and labels do not match.")
    
    augmented_data_dir  = os.path.join(output_data_root_dir, "augmented")
    augmented_image_dir = os.path.join(augmented_data_dir, "images")
    augmented_label_dir = os.path.join(augmented_data_dir, "labels")
    for dir in [augmented_image_dir, augmented_label_dir]:
        if not os.path.exists(dir):
            print("Creating directory: ", dir)
            os.makedirs(dir)

    augmented_image_memmap = None
    augmented_label_memmap = None

    num_images_at_a_time = 5000
    num_chunks = ceil(len(loaded_images) / num_images_at_a_time) 
    split_images = np.array_split(loaded_images, num_chunks)
    split_labels = np.array_split(loaded_labels, num_chunks)
    for idx, (images, labels) in enumerate(zip(split_images, split_labels)):
        if idx % 4 == 0:
            print("Augmenting chunk {} of {}.".format(idx, num_chunks))

        augmented_images, augmented_labels = JetImage.augment_many_images(images, labels)

        if idx == 0:
            augmented_image_memmap = mmap_ninja.np_from_ndarray(augmented_image_dir, augmented_images)
            augmented_label_memmap = mmap_ninja.np_from_ndarray(augmented_label_dir, augmented_labels)
        else:
            mmap_ninja.np_extend(augmented_image_memmap, augmented_images)
            mmap_ninja.np_extend(augmented_label_memmap, augmented_labels)
    
def verify_all_memmap_elements(memmap_dir:str):
    loaded = mmap_ninja.np_open_existing(memmap_dir)
    for idx in range(len(loaded)):
        try:
            loaded[idx]
        except:
            print("Error at index ", idx)
            break
    print("All elements in the memmap at {} are accessible.".format(memmap_dir))

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


