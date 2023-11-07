import torch
from utils import *
import numpy as np
import os
import config_funknn as config
from skimage.transform import radon
import imageio


class CT_images(torch.utils.data.Dataset):

    def __init__(self, directory, noise_snr = 30, unet = False, ood = False, train=True):

        self.directory = directory
        if train:
            name_list = os.listdir(self.directory)[:config.num_training]
            self.name_list = name_list * (30000//config.num_training)
        else:
            self.name_list = os.listdir(self.directory)


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

        # fbp = iradon(sinogram, theta=config.theta, circle= False)
        # fbp = torch.tensor(fbp, dtype = torch.float32)[None,...]
        if self.unet:
            fbp = iradon(sinogram, theta=config.theta, circle= False)
            fbp = torch.tensor(fbp, dtype = torch.float32)[None,...]
            return image[0], fbp

        sinogram = torch.tensor(sinogram, dtype = torch.float32)
        image = image[0,0].reshape(-1, 1)

        return image, sinogram#, fbp
    


class CT_dataset(torch.utils.data.Dataset):

    def __init__(self, directory, unet = False, train=True):

        self.directory = directory
        # if train:
        #     name_list = os.listdir(self.directory)[:config.num_training]
        #     self.name_list = name_list * (30000//config.num_training)
        # else:
        #     self.name_list = os.listdir(self.directory)

        self.name_list = sorted(os.listdir(self.directory))
        self.unet = unet


    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.name_list[idx]
        file = np.load(os.path.join(self.directory,file_name))
        image = file['image']
        image = torch.tensor(image, dtype = torch.float32)

        if config.memory_analysis:
            image = F.interpolate(image[None,None], size = config.image_size,
                                mode = 'nearest')[0]

        # print(image.shape)

        if self.unet:

            if config.uncalibrated:
                sinogram = file['sinogram']
                fbp = iradon(sinogram, theta=config.theta, circle= False)

            else:
                fbp = file['fbp']

            fbp = torch.tensor(fbp, dtype = torch.float32)[None,...]
            if config.memory_analysis:
                fbp = F.interpolate(fbp[None,...], size = config.image_size,
                                    mode = 'nearest')[0]

            return image[None,...], fbp
        
        else:
            sinogram = file['sinogram']
            sinogram = torch.tensor(sinogram, dtype = torch.float32)
            if config.memory_analysis:
                n = int(np.ceil((config.image_size) * np.sqrt(2)))
                sinogram = F.interpolate(sinogram[None,None], size = (n,config.n_angles),
                                    mode = 'nearest')[0,0]

            image = image.reshape(-1, 1)
            return image, sinogram
        
    
    


class CT_odl(torch.utils.data.Dataset):

    def __init__(self, directory):

        self.directory = directory
        self.name_list = sorted(os.listdir(self.directory))#[:config.num_training]



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

        if config.memory_analysis:
            image = F.interpolate(image[None,...], size = config.image_size,
                                mode = 'nearest')[0]
            
            fbp = F.interpolate(fbp[None,...], size = config.image_size,
                                mode = 'nearest')[0]
            
            n = int(np.ceil((config.image_size) * np.sqrt(2)))
            sinogram = F.interpolate(sinogram[None,...], size = (config.n_angles,n),
                                     mode = 'nearest')[0]
            
        return image, sinogram, fbp

    
