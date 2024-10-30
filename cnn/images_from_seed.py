import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Get the absolute path of the higher directory
higher_directories = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))]

# Append the higher directory to sys.path
for directory in higher_directories:
    if directory not in sys.path: sys.path.append(directory)

from JetsFromFile import JetsFromFile
from JetImages import JetImage

# def generate_images_from_seeds(kappa:float, energy_gev:int, seeds:list[int], data_dir:str):
#     all_tuples = [generate_images_from_seed(kappa, energy_gev, seed, data_dir) for seed in seeds]

#     up_images   = np.concatenate([t[0] for t in all_tuples])
#     down_images = np.concatenate([t[1] for t in all_tuples])

#     return up_images, down_images

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
 
    up_jet_datafile   = JetsFromFile(energy_gev,   "up", seed, data_dir)
    down_jet_datafile = JetsFromFile(energy_gev, "down", seed, data_dir)

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


import numpy as np
import random
import torch
from torchvision import tv_tensors
from random import shuffle


# def remove_directory_slash(directory:str):
#     if directory[-1] == '/':
#         return directory[:-1]
#     return directory

# def input_file_name_from_seed(image_dir:str, energy_gev:int, kappa:float, seed:int):
#     image_dir = remove_directory_slash(image_dir)

#     return os.path.join(image_dir, "{}GEV-kappa{}-seed{}.npy".format(energy_gev, round(kappa, 1), seed))

# def batch_images_filename(batch_outdata_dir:str, energy_gev:int, kappa:float, seed:int, batch_id:int):
#     batch_outdata_dir = remove_directory_slash(batch_outdata_dir)

#     return "batch{}--generated--from--{}GEV-kappa{}-seed{}.npy".format(batch_id, energy_gev, round(kappa, 1), seed)

# def batch_labels_filename(batch_outdata_dir:str, energy_gev:int, kappa:float, seed:int, batch_id:int):
#     batch_outdata_dir = remove_directory_slash(batch_outdata_dir)

#     return "batch{}--generated--from--{}GEV-kappa{}-seed{}-labels(up=0,dn=1).npy".format(batch_id, energy_gev, round(kappa, 1), seed)


import matplotlib.image as mpimg
from pathlib import Path
import mmap_ninja
from mmap_ninja import RaggedMmap


# def mmap_image_batches():
#     batched_data_dir = sys.argv[1]
#     print("Batched data directory: ", batched_data_dir)

#     all_files = os.listdir(batched_data_dir)
#     all_image_files = [f for f in all_files if f.startswith("batch") and f.endswith(".npy") and "labels" not in f]

#     all_image_locations = [os.path.join(batched_data_dir, f) for f in all_image_files]
#     print(all_image_locations[:5])
    
#     RaggedMmap.from_generator(
#         out_dir = Path(batched_data_dir),
#         sample_generator=map(np.load, all_image_locations),
#         batch_size=1000,
#         verbose=True
#     )


import shutil

# def mmap_label_batches():
#     batched_data_dir = sys.argv[1]
#     print("Batched data directory: ", batched_data_dir)

#     all_files = os.listdir(batched_data_dir)
#     if len(all_files) == 0:
#         print("No files in the directory")
#         return
    
#     all_image_files = [f for f in all_files if f.startswith("batch") and f.endswith(".npy") and "labels" in f]

#     all_image_locations = [os.path.join(batched_data_dir, f) for f in all_image_files]

#     RaggedMmap.from_generator(
#         out_dir = Path(batched_data_dir),
#         sample_generator=map(np.load, all_image_locations),
#         batch_size=1000,
#         verbose=True
#     )

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
    defaults = ["../data/up_down", os.path.join('d:/', "up_down_batch_data"), 100, 0.2]
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

    # Base image, four translations, and three reflections
    num_translations = 4
    num_reflections  = 3
    num_versions_per_image = 1 + num_translations + num_reflections

    # Multiplied by two for up and down
    num_total_images_per_seed = 2 * num_versions_per_image * JetsFromFile.JET_EVENTS_PER_FILE
    eventual_batch_size = 512 
    num_full_batches_per_seed = num_total_images_per_seed // eventual_batch_size
    total_num_batches_per_seed = num_full_batches_per_seed + 1
    print("Total number of batches per seed: ", total_num_batches_per_seed)

    num_seeds = 100
    # Approximate total size is ~130 GB
    # chosen_num_final_files = 50

    # num_total_images  = num_total_images_per_seed * num_seeds
    num_total_batches = total_num_batches_per_seed * num_seeds

    batch_id_numbers = list(range(1, num_total_batches+1))
    shuffle(batch_id_numbers)
    batch_id_numbers = iter(batch_id_numbers)

    image_map_created = False
    image_memmap = None
    label_map_created = False
    label_memmap = None
    
    # so_far = 0
    return
    for seed_no in range(1, num_seeds+1):
        image_outdata_dir = os.path.join(out_dir, "seed{}-images".format(seed_no))
        label_outdata_dir = os.path.join(out_dir, "seed{}-labels".format(seed_no))

        if not os.path.exists(image_outdata_dir):
            os.makedirs(image_outdata_dir)

            image_map_created = False
            image_memmap = None

        if not os.path.exists(label_outdata_dir):
            os.makedirs(label_outdata_dir)      
            label_map_created = False
            label_memmap = None

        print("Generating images for seed {} of 100".format(seed_no))
        up_images, down_images = generate_images_from_seed(kappa, energy_gev, seed_no, in_dir)
        all_images, is_down    = combine_up_down_images_create_labels(up_images, down_images)

        print("\tAll jet images have been generated and labelled for seed {} of 100".format(seed_no))
        print("\tPreprocessing the images' first (i.e. momentum) channel ".format(seed_no))
        all_images = JetImage.preprocess_a_channel(all_images, channel_idx=0)
        
        print("\tAugmenting the image data set")
        all_images = JetImage.augment_many_images(all_images)
        is_down    = np.concatenate([is_down for _ in range(num_versions_per_image)])

        print("\tAll jet images for this seed are augmented and preprocessed. Now shuffling the images and labels.")
        permutation = list(range(num_total_images_per_seed))
        random.Random(0).shuffle(permutation)
        all_images = all_images[permutation,:,:,:]
        is_down    = is_down[permutation]

        batched_images = np.array_split(all_images, total_num_batches_per_seed , axis=0)
        batched_labels = np.array_split(is_down,    total_num_batches_per_seed , axis=0)
        assert len(batched_images) == len(batched_labels)

        print("\tAll data creation finished and results split among the final {} batches.\n\tSaving each batch to disk.".format(total_num_batches_per_seed))
        for (idx, (images, labels)) in enumerate(zip(batched_images, batched_labels)):
            # so_far += 1
            if idx % 50 == 0:
                print("\t\tSaved {} batches of {}.".format(idx, total_num_batches_per_seed))

            if images.shape[0] > eventual_batch_size:
                raise ValueError("The number of images in a batch is greater than the eventual batch size")
            if labels.shape[0] > eventual_batch_size:
                raise ValueError("The number of labels in a batch is greater than the eventual batch size")
            

            # batch_images_filename = "batch{}--generated--from--{}GEV-kappa{}-seed{}.npy".format(batch_id, energy_gev, round(kappa, 1), i)
            # batch_labels_filename = "batch{}--generated--from--{}GEV-kappa{}-seed{}-labels(up=0,dn=1).npy".format(batch_id, energy_gev, round(kappa, 1), i)

            if not image_map_created:
                image_memmap = RaggedMmap.from_lists(image_outdata_dir, [images])
                image_map_created = True
            else:
                image_memmap.append(images)

            if not label_map_created:
                label_memmap = RaggedMmap.from_lists(label_outdata_dir, [labels])
                label_map_created = True
            else:
                label_memmap.append(labels)
                
