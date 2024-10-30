import os
import sys
# sys
# 
# Get the absolute path of the higher directory
higher_directories = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))]

# Append the higher directory to sys.path
for directory in higher_directories:
    if directory not in sys.path: sys.path.append(directory)

from Jets import Jet


import matplotlib.pyplot as plt
import numpy as np

"""
This class is responsible for generating images from jets.
"""
class JetImage(object):
    jet_radius = 0.4

    pixels_per_dim     = 33

    variable_range = np.linspace(-2*jet_radius, 2*jet_radius, pixels_per_dim)

    @classmethod 
    def augment_many_images(cls, image_set:np.array):
        assert len(image_set.shape) == 4, "The input has shape: {}. It must be a 4D array".format(image_set.shape)

        all_augments = [np.zeros_like(image_set) for _ in range(7)] 

        # The three reflections of the image: horizontal, vertical, and both
        all_augments[0] = image_set[:,:, :, ::-1]
        all_augments[1] = image_set[:,:, ::-1, :]
        all_augments[2] = image_set[:,:, ::-1, ::-1]

        # The four translations of the image
        all_augments[3][:, :, 1:,  :] = image_set[:, :, 1:, :] 
        all_augments[4][:, :, :-1, :] = image_set[:,:, :-1, :]
        all_augments[5][:, :, :,  1:] = image_set[:,:, :, :-1]
        all_augments[6][:, :, :, :-1] = image_set[:,:, :, 1:]

        return np.concatenate([image_set] + all_augments, axis=0)
    
    @classmethod
    def preprocess_a_channel(cls, many_images:np.array, channel_idx:int):
        if len(many_images.shape) == 4:
            ...
            # print("The input has shape: ", many_images.shape)   
        else:
            raise ValueError("The input has shape: {}. It must be a 4D array".format(many_images.shape))
        
        # L1 normalization
        sums = np.sum(many_images[:,channel_idx,:,:], axis=(1,2), keepdims=True)
        many_images[:, channel_idx, :,:] = np.divide(many_images[:,channel_idx,:,:], sums, where= np.abs(sums) > 0)

        # Zero centering
        mean = np.mean(many_images[:,channel_idx,:,:], axis=(0), keepdims=True)
        many_images[:, channel_idx, :,:] -= mean

        # Rescaling by the standard deviation
        for_numerical_stability = 1e-6
        std  = np.std(many_images[:,channel_idx,:,:],  axis=(0), keepdims=True) + for_numerical_stability  
        many_images[:, channel_idx, :,:] = np.divide(many_images[:,channel_idx,:,:], std, where= np.abs(std) > 0)

        return many_images

    @classmethod
    def two_channel_image_from_jet(cls, jet:Jet, kappa:float):

        first_channel  = cls.empty_channel()
        second_channel = cls.empty_channel()

        for particle_idx, bin_edges in enumerate(cls.sort_particles_into_bins(jet)):
            eta_bin, phi_bin = bin_edges

            first_channel[eta_bin,  phi_bin] += jet.get_particle_pts()[particle_idx]

            second_channel[eta_bin, phi_bin] += jet.get_particle_charges()[particle_idx] * jet.get_particle_pts()[particle_idx] ** kappa 

        second_channel /= jet.get_pt() ** kappa

        image = np.stack([first_channel, second_channel])

        return image
    
    @classmethod
    def one_channel_image(cls, jet:Jet, kappa:float):
        first_channel = cls.empty_channel()

        for particle_idx, bin_edges in enumerate(cls.sort_particles_into_bins(jet)):
            eta_bin, phi_bin = bin_edges

            first_channel[eta_bin, phi_bin] += jet.get_particle_pts()[particle_idx]

        image = np.stack([first_channel]) 

        return image
    
    

    @classmethod
    def empty_channel(cls, dtype=np.float32):
        return np.zeros((cls.pixels_per_dim, cls.pixels_per_dim), dtype=dtype)

    @classmethod
    def sort_particles_into_bins(cls, jet:Jet):
        if jet.get_num_particles() == 0:
            raise ValueError("The jet has no particles to sort into bins")
        
        eta_range = JetImage.variable_range + jet.get_eta()
        phi_range = JetImage.variable_range + jet.get_phi()

        # The -1 is to make the bins start at 0, and therefore 
        # be compatible with the indexing of the image
        eta_bins = np.digitize(jet.get_particle_etas(), eta_range) - 1
        phi_bins = np.digitize(jet.get_particle_phis(), phi_range) - 1

        particle_bins = zip(eta_bins, phi_bins)

        return particle_bins

        

