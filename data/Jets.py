import energyflow

class Particle(object):
    def __init__(self, eta, phi, pt, E, id):
        self.id  = id
        
        self.eta = eta
        self.phi = phi
        self.pt  = pt
        self.E   = E

        self.p   =  energyflow.p4s_from_ptyphipids([pt, eta, phi, id])

    def charge(self):
        energyflow.pids2chrgs([self.id])

class Jet(object):
    def __init__(self, phi, eta, total_pt):
        self.particles = []
        self.phi       = phi
        self.eta       = eta
        self.total_pt  = total_pt   

    def add_particle_from_pythia_line(self, line):
        self.particles.append()