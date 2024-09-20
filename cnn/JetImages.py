import sys
sys.path.append('../data')
from data_loading import Jet, Particle

import numpy as np


class JetImage(object):
    jet_radius = 0.4
    image_width = 2*jet_radius

    pixels_per_dim = 33

    @classmethod
    def empty_channel(cls):
        return np.zeros((cls.num_pixels, cls.num_pixels))

    @classmethod
    def sort_particles_into_bins(cls, jet: Jet):
        pass

