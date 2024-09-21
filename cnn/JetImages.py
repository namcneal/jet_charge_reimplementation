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
    def empty_channel(cls):
        return np.zeros((cls.pixels_per_dim, cls.pixels_per_dim))
    
    def __init__(self, jet: Jet):
        self.jet = jet

    def create_image(self, kappa):
        particle_bins = list(self.sort_particles_into_bins())

        first_channel = self.cumululative_pt_channel(particle_bins)
        second_channel = self.pt_weighted_jetcharge_channel(particle_bins, kappa)

        return np.stack([first_channel, second_channel])

    def pt_weighted_jetcharge_channel(self, particle_bins:list, kappa):
        channel = JetImage.empty_channel()

        for particle_idx, particle in enumerate(self.jet.particles):
            eta_bin, phi_bin = particle_bins[particle_idx]

            channel[eta_bin, phi_bin] += particle.charge() * particle.pt**kappa

        return channel / self.jet.total_pt**kappa

    def cumululative_pt_channel(self, particle_bins:list):
        channel = JetImage.empty_channel()

        for particle_idx, particle in enumerate(self.jet.particles):
            eta_bin, phi_bin = particle_bins[particle_idx]

            channel[eta_bin, phi_bin] += particle.pt

        return channel / self.jet.total_pt

    def sort_particles_into_bins(self):
        eta_range = JetImage.variable_range + self.jet.eta
        phi_range = JetImage.variable_range + self.jet.phi

        all_eta = np.array([p.eta for p in self.jet.particles])
        all_phi = np.array([p.phi for p in self.jet.particles])

        # The -1 is to make the bins start at 0, and therefore 
        # be compatible with the indexing of the image pixels
        eta_bins = np.digitize(all_eta, eta_range) - 1
        phi_bins = np.digitize(all_phi, phi_range) - 1
        print(eta_bins)

        particle_bins = zip(eta_bins, phi_bins)

        return particle_bins

        

