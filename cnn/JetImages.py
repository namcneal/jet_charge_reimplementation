import sys
sys.path.append('../data')
from data_loading import Jet, Particle

import numpy as np


class JetImage(object):
    jet_radius = 0.4
    image_width = 2*jet_radius

    pixels_per_dim = 33

    variable_range = np.linspace(-image_width, image_width, pixels_per_dim)

    @classmethod
    def normalized_two_channel_image(cls, jet:Jet, kappa:float):
        particle_bins = list(cls.sort_particles_into_bins(jet))

        first_channel  = cls.cumululative_pt_channel(jet, particle_bins)
        second_channel = cls.pt_weighted_jetcharge_channel(jet, particle_bins, first_channel, kappa)

        return np.stack([first_channel, second_channel])
    
    @classmethod
    def normalize_channel(cls, channel, channel_name):
        if np.abs(np.sum(channel, axis=None)) < 1e-5:
            raise ValueError("The {} channel has  summed to {}. Cannot normalize to 1.".format(channel_name, np.sum(channel, axis=None)))
        
        return channel / np.sum(channel, axis=None)

    @classmethod
    def pt_weighted_jetcharge_channel(cls, jet:Jet, particle_bins, cumulative_pt_channel:np.ndarray, kappa:float):
        channel = JetImage.empty_channel()

        for particle_idx, particle in enumerate(jet.particles):
            eta_bin, phi_bin = particle_bins[particle_idx]

            channel[eta_bin, phi_bin] += particle.charge() * particle.pt**kappa

            # TODO: check if this is the correct way to normalize the pt-weighted jet charge for each bin
            # One could also normalize using the jet's total pt
            channel[eta_bin, phi_bin] /= cumulative_pt_channel[eta_bin, phi_bin]

        return channel
    
        # return cls.normalize_channel(channel, "Charge")

    @classmethod
    def cumululative_pt_channel(cls, jet:Jet, particle_bins:list):
        channel = JetImage.empty_channel()

        for particle_idx, particle in enumerate(jet.particles):
            eta_bin, phi_bin = particle_bins[particle_idx]

            channel[eta_bin, phi_bin] += particle.pt

        channel /= jet.total_pt

        return channel
        # return cls.normalize_channel(channel, "Pt")
    
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

        

