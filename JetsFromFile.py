import numpy as np

from Jets import Particle, Jet 

class JetsFromFile(object):
    JET_EVENTS_PER_FILE = 10_000

    def __init__(self, energy_gev, origin_particle, seed_no, dir):
        if dir[-1] != '/':
            dir += '/'

        if energy_gev not in (100,1000):
            raise ValueError("Energy in GeV must be either 100 or 1000")
        if origin_particle not in ("up", "down"):
            raise ValueError("origin_particle must be either 'up' or 'down'")
        if not type(seed_no) == int:
            raise ValueError("seed_no must be an integer")
        
        self.origin_particle = origin_particle
        self.filename = "{}{}GEV-{}quark-seed{}.txt".format(dir, energy_gev, origin_particle, seed_no)
    
    def from_txt(self):
        return self.jets_from_fastjet_txt()

    def jets_from_fastjet_txt(self):
        particles_by_jet : list[Particle] = []

        with open(self.filename, 'r') as f:
            lines = f.readlines()
            # print(lines)

        inside_jet = False
        for line in lines:
            if line[0:5] == "Event":
                inside_jet = True

                line = line.split(',')
                # eta, phi, pt = [float(num) for num in line[1:4]] # Not currently using the pt from the file
                # jets.append(Jet(self.origin_particle, eta, phi))

                particles_by_jet.append([])

            elif line in ('\n', '\r\n'): 
                inside_jet = False
                
            elif inside_jet:
                particles_by_jet[-1].append(self.particle_from_line(line))

        return [Jet.from_particles(particles, self.origin_particle) for particles in particles_by_jet]
    
    @classmethod
    def particle_from_line(cls, line):
        line = line.split(',')

        data = np.array([float(x) for x in line[0:-1]])
        pid  = int(line[-1])

        return Particle(*data, pid)

        
            
            


        

