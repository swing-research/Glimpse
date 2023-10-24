import torch
from utils import *
import numpy as np
import os
import config_funknn as config
from skimage.transform import radon
import imageio


class CT_images(torch.utils.data.Dataset):

    def __init__(self, directory, noise_snr = 30, unet = False, ood = False):

        self.directory = directory
        self.name_list = os.listdir(self.directory)[:config.num_training]
        self.noise_snr = noise_snr
        self.unet = unet
        self.ood = ood


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.name_list[idx]
        if self.ood:
            image = imageio.imread(os.path.join(self.directory,file_name))
            image = image/255.0
        else:
            image = np.load(os.path.join(self.directory,file_name))

        image = torch.tensor(image, dtype = torch.float32)[None,None]
        image = F.interpolate(image, size = config.image_size,
                                  mode = 'bilinear',
                                  antialias= True,
                                  align_corners= True)
        
        image_np = image.detach().cpu().numpy()[0,0]
        sinogram = radon(image_np, theta=config.theta, circle= False)

        noise_sigma = 10**(-self.noise_snr/20.0)*np.sqrt(np.mean(np.sum(
            np.square(np.reshape(sinogram, (1 , -1))) , -1)))
        noise = np.random.normal(loc = 0,
                                 scale = noise_sigma,
                                 size = np.shape(sinogram))/np.sqrt(np.prod(np.shape(sinogram)))
        sinogram += noise

        if self.unet:
            fbp = iradon(sinogram, theta=config.theta, circle= False)
            fbp = torch.tensor(fbp, dtype = torch.float32)[None,...]
            return image[0], fbp

        sinogram = torch.tensor(sinogram, dtype = torch.float32)
        image = image[0,0].reshape(-1, 1)

        return image, sinogram
    




class CT_odl(torch.utils.data.Dataset):

    def __init__(self, directory):

        self.directory = directory
        self.name_list = os.listdir(self.directory)[:config.num_training]



    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.name_list[idx]
        data = np.load(os.path.join(self.directory,file_name))
        image = data['image']
        sinogram = data['sinogram']
        fbp = data['fbp']

        image = torch.tensor(image, dtype = torch.float32)[None,...]
        fbp = torch.tensor(fbp, dtype = torch.float32)[None,...]
        sinogram = torch.tensor(sinogram, dtype = torch.float32)[None,...]

        return image, sinogram, fbp

    
