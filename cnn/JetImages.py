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

    def sort_particles_into_bins(self):
        eta_range = JetImage.variable_range + self.jet.eta
        phi_range = JetImage.variable_range + self.jet.phi

        all_eta = np.array([p.eta for p in self.jet.particles])
        all_phi = np.array([p.phi for p in self.jet.particles])


        eta_bins = np.digitize(all_eta, eta_range)
        phi_bins = np.digitize(all_phi, phi_range)

        particle_bins = zip(eta_bins, phi_bins)

        return particle_bins

        

