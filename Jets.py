import energyflow


class Particle(object):
    def __init__(self, eta, phi, pt, E, id):
        self.id  = id
        
        self.eta = eta
        self.phi = phi
        self.pt  = pt
        # self.E   = E

        # self.p   =  energyflow.p4s_from_ptyphipids([pt, eta, phi, id])

    def charge(self):
        # charge_map = {11: -1, -11:  1, 13: -1, -13:  1, 22:  0, -22:  0, 
        #       111:  0, -111:  0, 130:  0, -130:  0, 211:  1, -211: -1, 
        #       321:  1, -321: -1, 2112:  0, -2112:  0, 2212:  1, -2212: -1}
        # print("id: ", self.id)
        # print("charge: ", energyflow.pids2chrgs([self.id]))
        # assert energyflow.pids2chrgs([self.id]) == charge_map[self.id], "Charge map is incorrect"
        return energyflow.pids2chrgs([self.id])[0]
    
import numpy as np

class Jet(object):
    def __init__(self, origin:str):
        self.origin    = origin

        # self.particles     : list[Particle] = None
        self.num_particles = None
        self.particle_data : np.array       = None

        self.centroid_eta = None
        self.centroid_phi = None
        self.total_pt     = None

    def get_num_particles(self):
        return self.num_particles

    def get_eta(self):
        return self.centroid_eta
    
    def get_phi(self):
        return self.centroid_phi
    
    def get_pt(self):
        return self.total_pt
    
    def get_particle_etas(self):
        return self.particle_data[0,:].view()
    
    def get_particle_phis(self):
        return self.particle_data[1,:].view()
    
    def get_particle_pts(self):
        return self.particle_data[2,:].view()
    
    def get_particle_charges(self):
        return self.particle_data[3,:].view()

    @classmethod
    def from_particles(cls, particles:list[Particle], origin:str):
        jet = cls(origin)
        jet.num_particles = len(particles)
        
        particle_etas = [p.eta for p in particles]
        particle_phis = [p.phi for p in particles]
        particle_pts  = [p.pt  for p in particles]
        particle_charges = [p.charge() for p in particles]

        jet.particle_data = np.stack([particle_etas, particle_phis, particle_pts, particle_charges])

        jet.centroid_eta = np.average(particle_etas, weights=particle_pts)
        jet.centroid_phi = np.average(particle_phis, weights=particle_pts)
        jet.total_pt     = sum(particle_pts)

        return jet

