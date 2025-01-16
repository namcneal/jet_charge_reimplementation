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
    variable_width = 0.8 # for both eta and phi
    image_width_pixels     = 33  # pixels

    pixel_width = variable_width / image_width_pixels

    # variable_range = np.linspace(-2*jet_radius, 2*jet_radius, pixels_per_dim)

    @classmethod
    def empty_channel(cls, dtype=np.float32):
        return np.zeros((cls.image_width_pixels, cls.image_width_pixels), dtype=dtype)

    @classmethod
    def convert_position_variables_to_indices(cls, jet:Jet):
        etas = jet.get_particle_etas()
        phis = jet.get_particle_phis()
        pts  = jet.get_particle_pts()

        # Deal with split images 
        reference_phi = phis[np.argmax(pts)]
        phis[phis - reference_phi >  cls.variable_width] -= 2*np.pi
        phis[phis - reference_phi < -cls.variable_width] += 2*np.pi

        # print(etas / cls.pixel_width)
        # print(phis / cls.pixel_width)

        # Convert the eta and phi continuous variables to pixel indices
        eta_pixel_indices = np.ceil(etas / cls.pixel_width - 0.5)
        phi_pixel_indices = np.ceil(phis / cls.pixel_width - 0.5)

        # print(eta_pixel_indices, phi_pixel_indices)

        # Find the centroid of the jet and its pixel location so that we can center the image
        centroid_eta = np.average(etas, weights=pts)
        centroid_phi = np.average(phis, weights=pts)

        centroid_eta_pixel_idx = np.ceil(centroid_eta / cls.pixel_width - 0.5) - np.floor(cls.image_width_pixels / 2)
        centroid_phi_pixel_idx = np.ceil(centroid_phi / cls.pixel_width - 0.5) - np.floor(cls.image_width_pixels / 2)

        # Center the eta and phi pixel indices around the centroid
        eta_pixel_indices -= centroid_eta_pixel_idx
        phi_pixel_indices -= centroid_phi_pixel_idx

        return eta_pixel_indices.astype(int), phi_pixel_indices.astype(int)

    @classmethod
    def get_valid_pixel_mask(cls, eta_pixel_indices, phi_pixel_indices):
        # Keep only the pixels that are within the correct range of
        # zero to the pixel width of the image
        valid_eta_pixels = (eta_pixel_indices >= 0) & (eta_pixel_indices < cls.image_width_pixels) 
        valid_phi_pixels = (phi_pixel_indices >= 0) & (phi_pixel_indices < cls.image_width_pixels)
        valid_pixels = valid_eta_pixels & valid_phi_pixels

        return valid_pixels

    @classmethod
    def zipped_position_indices_pt_and_charge(cls, jet:Jet):
        eta_indices, phi_indices = cls.convert_position_variables_to_indices(jet)
        pts     = jet.get_particle_pts()
        charges = jet.get_particle_charges()

        valid_pixels = cls.get_valid_pixel_mask(eta_indices, phi_indices)

        return zip(eta_indices[valid_pixels], phi_indices[valid_pixels], pts[valid_pixels], charges[valid_pixels])
    @classmethod
    def two_channel_image_from_jet(cls, jet:Jet, kappa:float):
        first_channel  = cls.empty_channel()
        second_channel = cls.empty_channel()

        for (eta_idx, phi_idx, pt, charge) in cls.zipped_position_indices_pt_and_charge(jet):
            # Transverse momentum  
            first_channel[eta_idx, phi_idx]  += pt

            # Pt-weighted jet charge
            second_channel[eta_idx, phi_idx] += charge * pt ** kappa

        second_channel /= jet.get_total_pt() ** kappa

        image = np.stack([first_channel, second_channel])

        return image

    @classmethod
    def one_channel_image_from_jet(cls, jet:Jet, kappa:float):
        first_channel = cls.empty_channel()

        for (eta_idx, phi_idx, pt, _) in zip(cls.position_indices_pt_and_charge(jet)):             
            # Transverse momentum
            first_channel[eta_idx, phi_idx] += pt

        return image

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
    def augment_many_images(cls, image_set:np.array, label_set:np.array):
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

        return (np.concatenate([image_set] + all_augments, axis=0),
                np.concatenate([label_set for _ in range(8)], axis=0))
    



    

    


        

