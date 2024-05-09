"""
This script contains the class to simulate the forward operator.
"""

import numpy as np
import torch
from utils import generate_projections, generate_FBP
from utils import find_sigma_noise


class Simulator():
    def __init__(self, config):
        """
        Class to simulate the forward operator
        """
        self.config = config
        self.n1 = config.n1
        self.n2 = config.n2
        self.n3 = config.n3
        self.n_projections = config.n_projections
        self.angle_max = config.angle_max
        self.angle_min = config.angle_min
        self.simulate_noise = config.simulate_noise
        self.noise_level = config.noise_level # TODO: make this as list of limits and take linspace bettween them
        self.pix = config.pix
        self.fixed_angles = config.fixed_angles
        if self.fixed_angles:
            self.angles = np.array(config.angles)
        else:
            self.angles = np.linspace(self.angle_min, self.angle_max, self.n_projections)
            self.angle_delta = (self.angle_max  - self.angle_min)/(self.n_projections-1)

        # Parameters used by the dataset    
        self.defocus_list = config.defocus_list # TODO: make it linspace between min and max
        self.dose_list = config.dose_list

        # Parameters used by the simulator
        # TODO: This can be made better
        # Current only some parametes can be change
        self.simulatar_parm = {'voltage': 300, 
                  'aberration':2.7, 
                  'sigma':0.3,
                  'defocus':-3,
                  'tilt':0,
                  'dose': 300,
                  'tiltscheme':0,
                  'pix':self.pix,
                  'tiltax':'Y',
                  'raddamage':0,
                  'scatter':1,
                  'ctfoverlap':20,
                  'tilterr':0, }
        
    
    def simulate(self,volume):
        """
        Simulate the forward operator
        """

        angles = self.angles.copy()

        if self.fixed_angles is False:
            # Add random noise to the angles
            angles = angles + self.angle_delta*(np.random.rand(self.n_projections)-0.5)

        # Simulate the projections
            
        # angles = torch.FloatTensor(angles).to(volume.device)
        proj = generate_projections(volume, angles)   

        # Simulate the CTF
        #sample some of the parameters

        # sim_param = self.simulatar_parm
        # dose = np.random.choice(self.dose_list)
        # sim_param['dose'] = dose
        # defocus = np.random.uniform(low=min(self.defocus_list), high=max(self.defocus_list), size=1)[0]
        # sim_param['defocus'] = defocus

        # sim_param['tilt'] = angles*180/torch.pi


        # proj = helper_ctf(proj, sim_param)

        # proj =  (proj - proj.mean())/proj.std()
        # proj = proj/proj.max()

        if self.simulate_noise:
            # Add noise
            # if noise level is a list, then sample from it
            if isinstance(self.noise_level, list):
                noise_level = np.random.uniform(low = min(self.noise_level),
                                                 high = max(self.noise_level), size=1)[0]
            else:
                noise_level = self.noise_level
            # TODO : include Gaussian approximation of poisson noise
            sigma_value = find_sigma_noise(noise_level,proj)
            proj = proj + sigma_value*torch.randn_like(proj) #+  abs(proj)*torch.randn_like(proj)*alpha

        return proj, angles
    
    def FBP(self, proj):

        angles = self.angles.copy()
        # angles = torch.FloatTensor(angles).to(proj.device)
        fbp = generate_FBP(proj, angles)

        return fbp



    
    # def deform_projection(self,proj):
    #     """
    #     This includes global shifts
    #     """

    #     n_angles = proj.shape[0]

    #     # Add global shifts
    #     for 




        
        
        

