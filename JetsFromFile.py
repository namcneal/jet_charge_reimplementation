import numpy as np

from Jets import Particle, Jet 

class JetsFromFile(object):
    JET_EVENTS_PER_FILE = 10_000

    def add_backslash_to_dir(self, dir:str):
        if dir[-1] != '/':
            dir += '/'
        return dir

    def __init__(self, energy_gev, origin_particle, seed_no, raw_data_dir, year=2024):
        raw_data_dir = self.add_backslash_to_dir(raw_data_dir)

        raw_data_dir += "up_down"
        if year == 2017:
            raw_data_dir += "_2017"

        raw_data_dir = self.add_backslash_to_dir(raw_data_dir)

        if energy_gev not in (100,1000):
            raise ValueError("Energy in GeV must be either 100 or 1000")
        if origin_particle not in ("up", "down"):
            raise ValueError("origin_particle must be either 'up' or 'down'")
        if not type(seed_no) == int:
            raise ValueError("seed_no must be an integer")
        
        self.origin_particle = origin_particle
        if year == 2024:
            self.filename = "{}{}GeV-{}quark-seed{}.txt".format(raw_data_dir, energy_gev, origin_particle, seed_no)
        elif year == 2017:
            self.filename = "{}{}GEV-{}quark-event-seed{}.txt".format(raw_data_dir, energy_gev, origin_particle, seed_no)
        else:
            raise ValueError("year must be either 2024 or 2017")

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

        pid  = int(line[-1])
        data = np.array([float(x) for x in line[0:-1]])

        # Check for the old vs new data format
        rapidity = data[0]
        phi      = data[1]
        pt       = data[2]

        return Particle(rapidity, phi, pt, pid)

        
            
            


        

