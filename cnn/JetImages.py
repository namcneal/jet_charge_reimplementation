import sys
sys.path.append('../data')
from data_loading import Jet, Particle

import numpy as np

def sort_particles_into_bins(jet: Jet):
    all_eta = np.array([part.eta for part in jet.particles])
    all_phi = np.array([part.phi for part in jet.particles])

    delta_eta  = 0.024
    delta_phi = delta_eta

    eta_bins = np.arange(min(all_eta), max(all_eta), delta_eta)
    phi_bins = np.arange(min(all_phi), max(all_phi), delta_phi)

    particle_eta_bin_indices = np.digitize(all_eta, eta_bins)
    particle_phi_bin_indices = np.digitize(all_phi, phi_bins)

    return eta_bins, phi_bins, zip(particle_eta_bin_indices, particle_phi_bin_indices)




# def pt_histogram_from_jet(jet: Jet):
#     all_eta = np.array([part.eta for part in jet.particles])
#     all_phi = np.array([part.phi for part in jet.particles])
#     all_pt  = np.array([part.pt  for part in jet.particles])

#     delta_eta  = 0.024
#     delta_phi = delta_eta

#     eta_bins = np.arrange(min(all_eta), max(all_eta), delta_eta)
#     phi_bins = np.arrange(-np.pi, np.pi, delta_phi)

#     hist, eta_bins, phi_bins = np.histogram2d(all_eta, all_phi, bins=(eta_bins, phi_bins), weights=all_pt)

# def jetcharge_pt_histogram_from_jet(jet: Jet):
#     pass