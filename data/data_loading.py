import numpy as np

from Jets import Particle, Jet 

def particle_from_pythia_line(line):
    line = line.split(',')

    data = np.array([float(x) for x in line[0:-1]])
    pid  = int(line[-1])

    return Particle(*data, pid)

def jets_from_pythia_txt(filename):
    jets = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    inside_jet = False
    for line in lines:
        if line[0:5] == "Event":
            inside_jet = True

            line = line.split(',')
            jet_data = [float(num) for num in line[1:4]]
            jets.append(Jet(*jet_data))



        elif line in ('\n', '\r\n'): 
            inside_jet = False
            
        elif inside_jet:
            jets[-1].particles.append(particle_from_pythia_line(line))


    return jets 

        
            
            


        

