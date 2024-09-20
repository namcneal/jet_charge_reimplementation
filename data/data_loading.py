from energyflow import p4s_from_ptyphipids
import numpy as np


class Particle(object):
    @classmethod
    def from_pythia_line(self, line):
        line = line.split(',')

        data = np.array([float(x) for x in line[0:-1]])
        pid  = int(line[-1])

        return Particle(*data, pid)

    def __init__(self, eta, phi, pt, E, id):
        self.id  = id
        
        self.eta = eta
        self.phi = phi
        self.pt  = pt
        self.E   = E

        self.p   =  p4s_from_ptyphipids([pt, eta, phi, id])

    def charge(self):
        energyflow.pids2chrgs([self.id])

class Jet(object):
    def __init__(self):
        self.particles = []

    def add_particle_from_pythia_line(self, line):
        self.particles.append(Particle.from_pythia_line(line))


def jets_from_pythia_txt(filename):
    jets = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    inside_jet = False
    for line in lines:
        if line[0:5] == "Event":
            jets.append(Jet())
            inside_jet = True

        elif line in ('\n', '\r\n'): 
            inside_jet = False
            
        elif inside_jet:
            jets[-1].add_particle_from_pythia_line(line)


    return jets 

        
            
            


        

