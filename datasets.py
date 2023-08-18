import torch
from utils import *
import numpy as np
import os
import config_funknn as config




class CT_sinogram(torch.utils.data.Dataset):

    def __init__(self, directory, noise_snr = 30, self_supervised = False, test_set = False):

        self.directory = directory
        self.name_list = os.listdir(self.directory)[:config.num_training]
        self.noise_snr = noise_snr
        self.theta = np.linspace(0.0, 180.0, config.c, endpoint=False)
        self.self_supervised = self_supervised
        self.test_set = test_set


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.name_list[idx]
        data = np.load(os.path.join(self.directory,file_name))
        image = data['image']
        sinogram = data['sinogram']

        noise_sigma = 10**(-self.noise_snr/20.0)*np.sqrt(np.mean(np.sum(
            np.square(np.reshape(sinogram, (1 , -1))) , -1)))
        noise = np.random.normal(loc = 0,
                                 scale = noise_sigma,
                                 size = np.shape(sinogram))/np.sqrt(np.prod(np.shape(sinogram)))
        sinogram += noise

        if self.self_supervised:
            fbp = iradon(sinogram, theta=self.theta, circle= False)
            fbp_sinogram = radon(fbp, theta=self.theta, circle= False)
            fbp_sinogram = torch.tensor(fbp_sinogram, dtype = torch.float32)
            fbp = torch.tensor(fbp, dtype = torch.float32)
            fbp = fbp.reshape(-1, 1)
            return fbp , fbp_sinogram
        
        sinogram = torch.tensor(sinogram, dtype = torch.float32)
        image = torch.tensor(image, dtype = torch.float32)
        image = image.reshape(-1, 1)

        if config.max_scale > 1 and self.test_set:
            image_high = data['image_high']
            image_high = torch.tensor(image_high, dtype = torch.float32)
            image_high = image_high.reshape(-1, 1)

            return image, sinogram, image_high

        return image, sinogram
    


class CT_FBP(torch.utils.data.Dataset):

    def __init__(self, directory, noise_snr = 30):

        self.directory = directory
        self.name_list = os.listdir(self.directory)[:config.num_training]
        self.noise_snr = noise_snr
        self.theta = np.linspace(0.0, 180.0, config.c, endpoint=False)


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.name_list[idx]
        data = np.load(os.path.join(self.directory,file_name))
        image = data['image']
        sinogram = data['sinogram']

        noise_sigma = 10**(-self.noise_snr/20.0)*np.sqrt(np.mean(np.sum(
            np.square(np.reshape(sinogram, (1 , -1))) , -1)))
        noise = np.random.normal(loc = 0,
                                 scale = noise_sigma,
                                 size = np.shape(sinogram))/np.sqrt(np.prod(np.shape(sinogram)))
        sinogram += noise

        fbp = iradon(sinogram, theta=self.theta, circle= False)

        image = torch.tensor(image, dtype = torch.float32)[None,...]
        fbp = torch.tensor(fbp, dtype = torch.float32)[None,...]

        return image, fbp