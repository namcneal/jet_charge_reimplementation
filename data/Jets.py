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
        return energyflow.pids2chrgs([self.id])

class Jet(object):
    def __init__(self, eta, phi, total_pt):
        self.particles = []
        self.eta       = eta
        self.phi       = phi
        self.total_pt  = total_pt