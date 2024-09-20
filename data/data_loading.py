from energyflow import p4s_from_ptyphipids


class Particle(object):
    def __init__(self, pt, eta, phi, pid):
        self.id = pid
        self.p  = self.momentum(pt, eta, phi)

        self.pt  = pt
        self.eta = eta
        self.phi = phi

    def momentum(self, pt, eta, phi):
        return p4s_from_ptyphipids(pt, eta, phi, self.id)
    

class Jet(object):
    def __init__(self):
        self.particles = []

    def __len__(self):
        return len(self.particles)

    def __getitem__(self, i):
        return self.particles[i]

    def __iter__(self):
        return iter(self.particles)
