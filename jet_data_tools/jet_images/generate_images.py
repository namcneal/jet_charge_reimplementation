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

higher_directories = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 
                      os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
                    ]

# Append the higher directory to sys.path
for directory in higher_directories:
    if directory not in sys.path: sys.path.append(directory)

from FileSystemNavigation import Directories, Filenames
from JetsFromFile import JetsFromFile
from JetImages import JetImage


class JetChargeAttributes(object):
    def __init__(self, year:int, seed:int, energy_gev:int, jet_charge_kappa:float):
        self.year = year
        self.seed = seed
        self.energy_gev = energy_gev
        self.jet_charge_kappa = jet_charge_kappa


def generate_and_save_all_images(directories:Directories, filenames:Filenames, seeds:range, kappa):
    generate_jet_image_memmaps(directories, filenames, seeds, kappa)

##


def generate_images_from_seed(directories:Directories, filenames:Filenames, jet_charge_data_attributes:JetChargeAttributes):
    seed   = jet_charge_data_attributes.seed
    energy_gev = jet_charge_data_attributes.energy_gev
    kappa  = jet_charge_data_attributes.jet_charge_kappa
    year   = jet_charge_data_attributes.year

    up_jet_datafile   = JetsFromFile(energy_gev,   "up", seed, directories.raw_data_directory, year)
    down_jet_datafile = JetsFromFile(energy_gev, "down", seed, directories.raw_data_directory, year)


    up_jets   = up_jet_datafile.from_txt()
    down_jets = down_jet_datafile.from_txt()

    ## Find any jets with no particles
    no_particles = [jet for jet in up_jets + down_jets if jet.get_num_particles() == 0]
    if len(no_particles) > 0:
        raise ValueError("Some jets have no particles")
    
    up_images   = np.stack([JetImage.two_channel_image_from_jet(jet, kappa) for jet in up_jets],   axis=0)
    down_images = np.stack([JetImage.two_channel_image_from_jet(jet, kappa) for jet in down_jets], axis=0)
    
    return up_images, down_images

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

def generate_images_from_all_seeds(directories:Directories, filenames:Filenames, seeds:range, kappa:float):
    # Multiplied by two for up and down
    total_num_images_per_seed = 2 * JetsFromFile.JET_EVENTS_PER_FILE

    # In each of the three directories, create two subdirectories for the images and labels
    train_test_image_dirs = [directories.training_image_directory(), directories.testing_image_directory()]
    train_test_label_dirs = [directories.training_label_directory(), directories.testing_label_directory()]

    for dir in train_test_image_dirs + train_test_label_dirs:
        if not os.path.exists(dir):
            print("Creating directory: ", dir)
            os.makedirs(dir)

    # These will be the memmap files for all the images and labels, for every seed combined.
    # There are two for each of the training, validation, and testing data, so six in total.
    # Each one of the pair lives in the subdirectories created above.
    train_test_image_memmaps = [None, None]
    train_test_label_memmaps = [None, None]

    num_seeds = len(seeds)
    for seed_no in seeds:
        print("Generating images for seed {} of {}".format(seed_no, num_seeds))

        jet_charge_data_attributes = JetChargeAttributes(directories.dataset_details.data_year, seed_no, directories.dataset_details.energy_gev, kappa)

        # Generate the up and down images for this seed as a numpy array of shape (num_images, num_channels, num_pixels, num_pixels)
        up_images, down_images = generate_images_from_seed(directories, filenames, jet_charge_data_attributes)

        # Combine the up and down images into a single array of shape (2*num_images, num_channels, num_pixels, num_pixels)
        # The up and down images are interlaced in the array, alternating between up and down
        # In the process the labels are created as well
        all_images, is_down  = combine_up_down_images_create_labels(up_images, down_images)

        assert all_images.shape[0] == total_num_images_per_seed
        assert is_down.shape[0]    == total_num_images_per_seed

        print("\tAll jet images have been generated and labelled. Up and down images interlaced.".format(seed_no))
        print("\tPreprocessing the images' first (i.e. momentum) channel ".format(seed_no))

        # Only the first channel is preprocessed (i.e. zero-centered and scaled by the standard deviation)
        all_images = JetImage.preprocess_a_channel(all_images, channel_idx=0)

        # Convert the labels to a one-hot encoding
        all_labels = torch.nn.functional.one_hot(torch.from_numpy(is_down).long(), 2).numpy()

        print("\tAll jets for this seed are processed. Saving 80 pct to training, 10 pct to validation, and 10 pct to testing.")

        # Compute the number of samples that will go into each of the training, validation, and testing sets
        # The training-esting split is 80-20
        TRAIN_PCT = 80
        num_training   = int(TRAIN_PCT/100 * total_num_images_per_seed)
        num_testing    = total_num_images_per_seed - num_training 
        # Assign each image to the training, validation, or testing set based on the computed split
        all_indices = np.arange(total_num_images_per_seed, dtype=np.int32)
        training_indices   = all_indices[:num_training]
        testing_indices    = all_indices[num_training:] 
        
        assigned_train_test_indices = [training_indices, testing_indices]

        # Iterate through image-label directory pairs for each of the training, validation, and testing sets
        for loop_idx, (image_dir, label_dir, assigned_indices) in enumerate(zip(train_test_image_dirs,
                                                                                train_test_label_dirs,  
                                                                                assigned_train_test_indices)):


            images_from_assigned_indices = all_images[assigned_indices,:,:,:]

            # Given the first seed, create the memmap files for each of the training, validation, and testing sets
            if train_test_image_memmaps[loop_idx] is None:
                train_test_image_memmaps[loop_idx] = mmap_ninja.np_from_ndarray(image_dir, images_from_assigned_indices)
                train_test_label_memmaps[loop_idx] = mmap_ninja.np_from_ndarray(label_dir, all_labels[assigned_indices,:])
            

            # Otherwise we append the new images and labels to the existing memmap files
            else:
                mmap_ninja.np_extend(train_test_image_memmaps[loop_idx], images_from_assigned_indices)
                mmap_ninja.np_extend(train_test_label_memmaps[loop_idx], all_labels[assigned_indices,:])

        print("\tImages and labels saved to memmap files for seed {}.".format(seed_no))
        print("\tSaved in total: {} training, {} testing images.".format(num_training, num_testing))

def generate_jet_image_memmaps(directories:Directories, filenames:Filenames, seeds:range, kappa:float): 
    # Generate the training, validation, and testing images and labels for all seeds
    # These are saved as memory-mapped files in the output_data_root_dir
    generate_images_from_all_seeds(directories, filenames, seeds, kappa)

    # Take the training images and labels and augment them to create more training data
    augment_training_data(directories)

    # Verify that all the memory-mapped files can be accessed 
    verify_all_memmaps(directories)

def augment_training_data(directories:Directories):

    # The directories for the training images and labels 
    # that are created in the function 'generate_images_from_all_seeds'
    training_image_memmap_dir = directories.training_image_directory()
    training_label_memmap_dir = directories.training_label_directory()

    print("Augmenting the training data.")
    # Load the existing images and labels memory-mapped files that were created
    # in the function 'generate_images_from_all_seeds'
    # These will not be modified, but will be used to create the augmented data
    training_image_memmap = mmap_ninja.np_open_existing(training_image_memmap_dir)
    training_label_memmap = mmap_ninja.np_open_existing(training_label_memmap_dir)

    # The training data is augmented in chunks rather than the full set at once
    num_images_at_a_time = 5000
    num_chunks = ceil(len(loaded_images) / num_images_at_a_time) 
    split_images = np.array_split(loaded_images, num_chunks)
    split_labels = np.array_split(loaded_labels, num_chunks)

    for idx, (images, labels) in enumerate(zip(split_images, split_labels)):
        if idx % 4 == 0:
            print("Augmenting chunk {} of {}.".format(idx, num_chunks))

        # The actual image augmentation. It includes: 3 reflections, 4 translations
        augmented_images, augmented_labels = JetImage.augment_many_images(images, labels)

        memmap_ninja.np_extend(training_image_memmap, augmented_images)
        memmap_ninja.np_extend(training_label_memmap, augmented_labels)

def verify_all_memmap_elements(memmap_dir:str):
    loaded = mmap_ninja.np_open_existing(memmap_dir)
    for idx in range(len(loaded)):
        try:
            loaded[idx]
        except:
            print("Error at index ", idx)
            break

    print("All elements in the memmap at {} are accessible.".format(memmap_dir))

def verify_all_memmaps(base_dir:str):
    for dir in [directories.training_image_directory(), 
                directories.training_label_directory(), 

                directories.testing_image_directory(), 
                directories.testing_label_directory()]:
                
        verify_all_memmap_elements( dir )