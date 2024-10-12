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


import numpy as np

class JetImage(object):
    jet_radius = 0.4
    image_width = 2*jet_radius

    pixels_per_dim = 33

    variable_range = np.linspace(-image_width, image_width, pixels_per_dim)

    @classmethod
    def two_channel_image(cls, jet:Jet, kappa:float):

        first_channel  = cls.empty_channel()
        second_channel = cls.empty_channel()

        for particle_idx, bin_edges in enumerate(cls.sort_particles_into_bins(jet)):
            eta_bin, phi_bin = bin_edges

            first_channel[eta_bin, phi_bin]          += jet.particles[particle_idx].pt
            
            second_channel[eta_bin, phi_bin] += jet.particles[particle_idx].charge() * jet.particles[particle_idx].pt**kappa

        # Complete the pt-weighted jet charge per bin 
        # by dividing by the total pt in that bin (to the power of kappa)
        second_channel /= jet.total_pt**kappa

        image = np.stack([first_channel, second_channel])

        return image
    
    @classmethod
    def preprocess_many_images(cls, many_images:np.array):
        assert len(many_images.shape) == 4, "The input must be a 4D array of shape (num_images, num_channels, pixels_per_dim, pixels_per_dim)"
        
        # Normalize the whole image by the sum of its pixels in each channel
        sums = np.sum(many_images, axis=(2,3), keepdims=True)
        normalized = np.divide(many_images, sums, where=np.abs(sums) > 0)

        # Zero-center each channel's pixels by the corresponding average over all images
        zero_centered = normalized - np.mean(normalized, axis=(0), keepdims=True)

        # Standardize each channel by the channel-standard-deviation over all images
        for_noise_reduction = 1e-6
        standardized = zero_centered /  (np.std(normalized, axis=(0), keepdims=True) + for_noise_reduction)

        return standardized
    
    @classmethod
    def normalize_channel(cls, channel, channel_name):
        if np.abs(np.sum(channel, axis=None)) < 1e-16:
            raise ValueError("The {} channel has  summed to {}. Cannot normalize to 1.".format(channel_name, np.sum(channel, axis=None)))
        
        return channel / np.sum(channel, axis=None)

    @classmethod
    def empty_channel(cls):
        return np.zeros((cls.pixels_per_dim, cls.pixels_per_dim))

    @classmethod
    def sort_particles_into_bins(cls, jet:Jet):
        eta_range = JetImage.variable_range + jet.eta
        phi_range = JetImage.variable_range + jet.phi

        all_eta = np.array([p.eta for p in jet.particles])
        all_phi = np.array([p.phi for p in jet.particles])

        # The -1 is to make the bins start at 0, and therefore 
        # be compatible with the indexing of the image
        eta_bins = np.digitize(all_eta, eta_range) - 1
        phi_bins = np.digitize(all_phi, phi_range) - 1

        particle_bins = zip(eta_bins, phi_bins)

        return particle_bins

        

